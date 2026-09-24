"""Splits trajectories into train/test by where the object was grasped.

split_data.py shuffles randomly, which cannot detect a policy that ignores vision: held-out
episodes are drawn from the same placement distribution, so a policy that memorises one approach
still lands within a couple of centimetres of every test episode and scores well.

This holds out episodes by grasp position instead, leaving a gap in the training coverage, so the
test set asks something the policy can actually fail: can it handle a placement it has not seen?
Comparing eval error on this test set against training episodes measures whether vision is being
used to locate the object, or whether one trajectory is being replayed.

The reported distance from each test episode to its nearest training episode is what tells you
whether a given split is worth running -- a few millimetres of separation proves nothing, because
replaying the neighbouring trajectory would pass.

Output layout matches split_data.py (symlinks into <output_path>/<data_name>_{train,test}), so the
downstream conversion and training scripts work unchanged.
"""

import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_data import is_valid_trajectory_dir  # noqa: E402


def grasp_position(h5_path):
    """End-effector xyz at the first large gripper transition, i.e. the grasp.

    Returns None when the episode has no usable gripper signal, in which case the caller keeps
    the episode for training rather than discarding perfectly good demonstration data.
    """
    with h5py.File(h5_path, "r") as f:
        if "frames/gripper_position" not in f or "frames/ee_pos_quat" not in f:
            return None
        gripper = np.asarray(f["frames/gripper_position"])[:, 0]
        ee_xyz = np.asarray(f["frames/ee_pos_quat"])[:, :3]

    span = gripper.max() - gripper.min()
    if span < 1e-6 or len(gripper) < 5:
        return None
    moved = np.abs(gripper - gripper[0]) > 0.5 * span
    if not moved.any():
        return None
    return ee_xyz[int(np.argmax(moved))]


def describe(label, positions, axis_index):
    if not positions:
        print(f"  {label}: (none)")
        return
    values = np.array([p[axis_index] for p in positions])
    print(
        f"  {label}: {len(values)} episodes, "
        f"{values.min():+.3f} .. {values.max():+.3f} m (span {values.max() - values.min():.3f})"
    )


def split_by_position(root, target_root, holdout, axis, mode):
    target_train_root = target_root + "_train"
    target_test_root = target_root + "_test"

    if os.path.exists(target_train_root) or os.path.exists(target_test_root):
        raise SystemExit(f"{target_train_root} or {target_test_root} already exists; remove them first.")

    episodes = [p for p in sorted(os.listdir(root)) if is_valid_trajectory_dir(root, p)]
    if len(episodes) < holdout + 2:
        raise SystemExit(f"found only {len(episodes)} valid episodes in {root}, need more than {holdout}")

    located, unlocated = [], []
    for name in episodes:
        position = grasp_position(os.path.join(root, name, "trajectory.h5"))
        (located if position is not None else unlocated).append(
            (name, position) if position is not None else name
        )

    if len(located) < holdout + 2:
        raise SystemExit(f"could only locate grasps in {len(located)} episodes, need more than {holdout}")

    axis_index = "xyz".index(axis)
    located.sort(key=lambda item: item[1][axis_index])

    if mode == "high":
        test, train = located[-holdout:], located[:-holdout]
    elif mode == "low":
        test, train = located[:holdout], located[holdout:]
    else:
        start = (len(located) - holdout) // 2
        test, train = located[start : start + holdout], located[:start] + located[start + holdout :]

    print(f"splitting {len(episodes)} episodes from {root} by grasp {axis} (mode={mode})")
    if unlocated:
        print(f"  ({len(unlocated)} episodes had no usable gripper signal; kept for training)")
    describe("train", [p for _, p in train], axis_index)
    describe("test ", [p for _, p in test], axis_index)

    # How far each test placement sits from the closest one the policy will have trained on.
    # This is what decides whether the split can detect a memorising policy at all: if the
    # nearest training episode is millimetres away, replaying its trajectory passes the test.
    train_values = np.array([p[axis_index] for _, p in train])
    gaps = [float(np.min(np.abs(train_values - p[axis_index]))) for _, p in test]
    print(f"  distance from each test episode to nearest training episode: median {np.median(gaps) * 100:.1f} cm, max {max(gaps) * 100:.1f} cm")

    os.makedirs(target_train_root)
    os.makedirs(target_test_root)
    for target, names in (
        (target_train_root, [name for name, _ in train] + unlocated),
        (target_test_root, [name for name, _ in test]),
    ):
        for name in names:
            os.symlink(os.path.abspath(os.path.join(root, name)), os.path.join(target, name))

    print(f"linked {target_train_root} and {target_test_root}")


if __name__ == "__main__":
    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--base_path", type=str, required=True)
    arg.add_argument("--output_path", type=str, required=True)
    arg.add_argument("--data_name", type=str, required=True)
    arg.add_argument("--holdout", type=int, default=15, help="episodes to reserve for test")
    arg.add_argument("--axis", type=str, default="x", choices=["x", "y", "z"])
    arg.add_argument(
        "--mode",
        type=str,
        default="high",
        choices=["high", "low", "middle"],
        help="which placements to hold out: the extremes (high/low) test extrapolation and give "
        "the clearest separation; middle tests interpolation but the gap is only as wide as the "
        "data is sparse there",
    )
    args = arg.parse_args()

    split_by_position(
        os.path.join(args.base_path, args.data_name),
        os.path.join(args.output_path, args.data_name),
        args.holdout,
        args.axis,
        args.mode,
    )
