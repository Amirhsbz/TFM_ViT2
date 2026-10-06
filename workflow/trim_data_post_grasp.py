"""Writes a copy of a dataset with each episode cut to start once the object is in the gripper.

A policy trained on the full episodes can learn the approach as one fixed motion when the
object sits in nearly the same place in every demo, instead of locating it visually. Cutting
the approach removes that part of the problem: at deployment the object is placed in the gripper
by hand and the policy takes over from there.

Where each episode is cut:
  * at grasp-complete -- the first frame, after the grasp found by grasp_index(), where the
    gripper has reached --settle-fraction of the closed level it holds the object at. Earlier
    frames have a half-closed gripper and "keep closing" action labels, neither of which occurs
    at deployment, where the gripper is already closed when inference starts. No pre-grasp frames
    are kept: pi0 conditions on the current frame only and its action chunks look forward, so
    they would add nothing to the post-grasp samples.
  * then past the still frames between grasp-complete and the arm starting to move, keeping
    --padding of them, so the policy does not learn to idle at the start pose.
  * the tail loses the still frames after both the arm and the gripper have stopped. The
    release stays in: it is part of the task ("... and put the sponge back"), and it happens
    with the arm already still, so arm motion alone would cut it off.

Unlike split_data.py this has to rewrite the h5 files (frames, timestamps, embedded videos and
freq.txt are all cut to the same range, using trim_bc_data_by_eef_motion.py's writer), so the
output is a new raw-data folder laid out like the input. Split it and convert it with the usual
scripts. Split by grasp position *before* trimming (split_data_by_position.py, then trim each
split): a trimmed episode starts closed, so grasp_position() on it would find the release.

Also writes, next to the episodes:
  * trim_report.json -- the kept range of every episode and why any were skipped
  * start_pose.json  -- a start configuration for deployment taken from the first kept frame of
    the most typical episode, plus how much the first frames vary across episodes
"""

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data_analysis"))
from split_data import is_valid_trajectory_dir  # noqa: E402
from split_data_by_position import grasp_index  # noqa: E402
from trim_bc_data_by_eef_motion import find_motion_run, write_trimmed_freq, write_trimmed_h5  # noqa: E402

MOTION_THRESHOLD = 0.001  # m/frame, same as trim_bc_data_by_eef_motion.py
MIN_MOTION_RUN = 3


def post_grasp_range(gripper, ee_xyz, settle_fraction, padding):
    """(start, end, grasp_complete) of the post-grasp part of one episode, or a reason string."""
    grasp = grasp_index(gripper)
    if grasp is None:
        return "no grasp: gripper never closes"

    # The closed segment runs from the grasp until the gripper is back within half way of open
    # (the release); the level it settles at within it is how tightly this demo held the object.
    travel = np.abs(gripper - gripper[0])
    closed = travel[grasp:] > 0.5 * (gripper.max() - gripper.min())
    closed_end = grasp + (len(closed) if closed.all() else int(np.argmin(closed)))
    segment = travel[grasp:closed_end]
    grasp_complete = grasp + int(np.argmax(segment >= settle_fraction * segment.max()))

    moving = np.linalg.norm(np.diff(ee_xyz, axis=0), axis=1) > MOTION_THRESHOLD
    first_run = find_motion_run(moving[grasp_complete:], MIN_MOTION_RUN)
    last_run = find_motion_run(moving, MIN_MOTION_RUN, reverse=True)
    if first_run is None or last_run is None or last_run[1] <= grasp_complete:
        return "arm never moves after the grasp"

    start = max(grasp_complete, grasp_complete + first_run[0] - padding)
    # last_run[1] indexes displacements; displacement i connects frame i -> i + 1. The release
    # usually happens with the arm already still, so the tail runs until the gripper stops too.
    last_gripper_change = int(np.flatnonzero(np.abs(np.diff(gripper)) > 1e-3)[-1]) + 1
    end = min(len(gripper), max(last_run[1] + 1, last_gripper_change + 1) + padding)
    return start, end, grasp_complete


def start_pose(kept):
    """Deployment start configuration from the most typical first kept frame."""
    arm = np.stack([k["first_joints"][:6] for k in kept])
    medoid = int(np.argmin(np.linalg.norm(arm - np.median(arm, axis=0), axis=1)))
    first_xyz = np.stack([k["first_ee_xyz"] for k in kept])
    return {
        "episode": kept[medoid]["name"],
        "arm_joints_rad": [float(x) for x in arm[medoid]],
        "arm_joints_deg": [round(float(x), 2) for x in np.degrees(arm[medoid])],
        # What to command the gripper to once the object is in it: the demos' own command at their
        # first kept frame, so the gripper state the policy sees matches what it was trained on.
        "gripper_command": float(np.median([k["first_gripper_command"] for k in kept])),
        "gripper_position_expected": float(np.median([k["first_gripper_position"] for k in kept])),
        "ee_xyz": [float(x) for x in kept[medoid]["first_ee_xyz"]],
        "spread_across_episodes": {
            "arm_joints_std_deg": [round(float(x), 2) for x in np.degrees(arm.std(axis=0))],
            "ee_xyz_std_cm": [round(float(x), 2) for x in first_xyz.std(axis=0) * 100],
            "ee_xy_distance_from_median_cm_p90": round(
                float(np.percentile(np.linalg.norm(first_xyz[:, :2] - np.median(first_xyz[:, :2], axis=0), axis=1), 90) * 100), 2
            ),
        },
    }


def trim_post_grasp(input_root, output_root, settle_fraction, padding, min_frames):
    if os.path.exists(output_root):
        raise SystemExit(f"{output_root} already exists; remove it first.")

    names = [p for p in sorted(os.listdir(input_root)) if is_valid_trajectory_dir(input_root, p)]
    print(f"trimming {len(names)} episodes from {input_root} to {output_root} (settle {settle_fraction}, padding {padding})")

    kept, skipped = [], []
    for name in names:
        h5_path = os.path.join(input_root, name, "trajectory.h5")
        with h5py.File(h5_path, "r") as f:
            frame_count = int(f.attrs.get("frame_count", len(f["frames/gripper_position"])))
            gripper = np.asarray(f["frames/gripper_position"])[:, 0]
            ee_xyz = np.asarray(f["frames/ee_pos_quat"])[:, :3]
            joints = np.asarray(f["frames/joint_positions"])
            control = np.asarray(f["frames/control"])

        result = post_grasp_range(gripper, ee_xyz, settle_fraction, padding)
        if isinstance(result, tuple) and result[1] - result[0] < min_frames:
            result = f"only {result[1] - result[0]} frames after the grasp"
        if isinstance(result, str):
            print(f"  skip {name}: {result}")
            skipped.append({"name": name, "reason": result})
            continue

        start, end, grasp_complete = result
        output_dir = os.path.join(output_root, name)
        os.makedirs(output_dir)
        output_h5 = os.path.join(output_dir, "trajectory.h5")
        write_trimmed_h5(Path(h5_path), Path(output_h5), start, end, frame_count)
        write_trimmed_freq(Path(input_root, name, "freq.txt"), Path(output_dir, "freq.txt"), start, end, frame_count)
        with h5py.File(output_h5, "a") as f:
            f.attrs["trim_source"] = os.path.realpath(h5_path)
            f.attrs["trim_range"] = np.array([start, end])

        kept.append({
            "name": name,
            "original_frames": frame_count,
            "grasp_complete": grasp_complete,
            "start": start,
            "end": end,
            "first_joints": joints[start],
            "first_ee_xyz": ee_xyz[start],
            "first_gripper_command": float(control[start, 6]),
            "first_gripper_position": float(gripper[start]),
        })
        print(f"\r  {len(kept) + len(skipped)}/{len(names)} {name}: keep [{start}:{end}] of {frame_count}", end="", flush=True)
    print()

    if not kept:
        raise SystemExit("no episode had a usable grasp")
    pose = start_pose(kept)
    with open(os.path.join(output_root, "start_pose.json"), "w") as f:
        json.dump(pose, f, indent=2)
    with open(os.path.join(output_root, "trim_report.json"), "w") as f:
        json.dump(
            {
                "input_root": os.path.realpath(input_root),
                "settle_fraction": settle_fraction,
                "padding": padding,
                "kept": [{k: v for k, v in item.items() if not k.startswith("first_")} for item in kept],
                "skipped": skipped,
            },
            f,
            indent=2,
        )

    frames = np.array([k["end"] - k["start"] for k in kept])
    removed_head = np.array([k["start"] for k in kept])
    print(f"kept {len(kept)} episodes, skipped {len(skipped)}")
    print(f"  frames per episode: median {int(np.median(frames))} (min {frames.min()}, max {frames.max()}); "
          f"head removed: median {int(np.median(removed_head))}")
    print(f"  start pose (from {pose['episode']}): arm {pose['arm_joints_deg']} deg, gripper command {pose['gripper_command']:.3f}")
    print(f"  first-frame spread: {pose['spread_across_episodes']}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arg.add_argument("--base_path", type=str, required=True)
    arg.add_argument("--output_path", type=str, required=True)
    arg.add_argument("--data_name", type=str, required=True)
    arg.add_argument("--suffix", type=str, default="_postgrasp", help="appended to data_name for the output folder")
    arg.add_argument("--settle_fraction", type=float, default=0.95, help="how closed the gripper must be to count as grasped")
    arg.add_argument("--padding", type=int, default=5, help="still frames to keep before the arm starts and after it stops")
    arg.add_argument("--min_frames", type=int, default=30, help="skip episodes with fewer post-grasp frames than this")
    args = arg.parse_args()

    trim_post_grasp(
        os.path.join(args.base_path, args.data_name),
        os.path.join(args.output_path, args.data_name + args.suffix),
        args.settle_fraction,
        args.padding,
        args.min_frames,
    )
