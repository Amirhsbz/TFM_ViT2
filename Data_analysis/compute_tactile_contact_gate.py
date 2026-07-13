#!/usr/bin/env python3
"""Compute tactile contact gates for raw H5 BC data.

Usage examples:
  Dry-run one dataset and inspect threshold behavior:
    conda run -n robodiff python3 Data_analysis/compute_tactile_contact_gate.py \
      shared/data/bc_data/wipe_board --dry-run

  Write contact gates from existing run_env marker-motion traces:
    conda run -n robodiff python3 Data_analysis/compute_tactile_contact_gate.py \
      shared/data/bc_data/wipe_board --overwrite \
      --summary-json shared/data/bc_data/wipe_board/contact_gate_summary.json

  Try a different gate threshold without modifying data:
    conda run -n robodiff python3 Data_analysis/compute_tactile_contact_gate.py \
      shared/data/bc_data/wipe_board --dry-run \
      --contact-on-threshold 4.0 --contact-off-threshold 2.0

What it writes under frames/:
  Requires existing tactile_left_marker_motion and tactile_right_marker_motion.
  contact_marker_motion: max(left_marker_motion, right_marker_motion).
  contact_gate: 0 before contact, 1 after contact.
  Optional contact_gate_segment_id: non-contact segment ids, -1 during contact.

The default wipe_board gate first subtracts each tactile stream's early
pre-contact marker-motion baseline, then gates on the max per-sensor delta:
  contact_on_threshold=0.5 and gripper_position>=0.2 for 2 consecutive frames.
  After contact, marker motion can turn the gate off only when gripper stroke is
  no longer in the closed/holding range. This avoids flicker when tactile marker
  motion briefly drops during sustained grasp/contact.
  gripper_position<=0.08 for 2 consecutive frames also turns the gate off, but
  only after the gripper has once closed to >=0.2.

Run this before rebuilding DP memmap caches or converting LeRobot/pi0 data, so
training data and deployment use the same tactile masking rule.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from learning.tactile_contact_gate import (
    ContactGateConfig,
    contact_gate_from_motion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute run_env-compatible tactile marker motion and contact_gate for H5 trajectories."
    )
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Analyze without writing H5 files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing contact_marker_motion/contact_gate datasets.",
    )
    parser.add_argument(
        "--write-contact-segments",
        action="store_true",
        help="Also write frames/contact_gate_segment_id for multi-stage tactile baselines.",
    )
    parser.add_argument("--contact-on-threshold", type=float, default=0.5)
    parser.add_argument("--contact-off-threshold", type=float, default=0.2)
    parser.add_argument("--contact-on-consecutive-frames", type=int, default=2)
    parser.add_argument("--contact-off-consecutive-frames", type=int, default=8)
    parser.add_argument("--gripper-on-threshold", type=float, default=0.2)
    parser.add_argument("--gripper-open-threshold", type=float, default=0.08)
    parser.add_argument("--gripper-closed-threshold", type=float, default=0.2)
    parser.add_argument("--gripper-hold-threshold", type=float, default=0.2)
    parser.add_argument("--gripper-open-consecutive-frames", type=int, default=2)
    parser.add_argument(
        "--disable-gripper-hold",
        action="store_true",
        help="Allow low marker motion to turn contact_gate off even while the gripper remains closed.",
    )
    parser.add_argument(
        "--motion-baseline-mode",
        choices=["early_median", "none"],
        default="early_median",
        help=(
            "How to normalize marker motion before contact gating. "
            "early_median subtracts each tactile stream's initial baseline before taking max."
        ),
    )
    parser.add_argument(
        "--motion-baseline-num-frames",
        type=int,
        default=20,
        help="Number of earliest frames used for the marker-motion baseline.",
    )
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = _trajectory_paths(args.data_root)
    if not paths:
        raise SystemExit(f"No trajectory.h5 files found under {args.data_root}")

    gate_cfg = ContactGateConfig(
        contact_on_threshold=args.contact_on_threshold,
        contact_off_threshold=args.contact_off_threshold,
        contact_on_consecutive_frames=args.contact_on_consecutive_frames,
        contact_off_consecutive_frames=args.contact_off_consecutive_frames,
        gripper_on_threshold=args.gripper_on_threshold,
        gripper_open_threshold=args.gripper_open_threshold,
        gripper_closed_threshold=args.gripper_closed_threshold,
        gripper_hold_threshold=args.gripper_hold_threshold,
        gripper_open_consecutive_frames=args.gripper_open_consecutive_frames,
        hold_contact_while_gripper_closed=not args.disable_gripper_hold,
    )

    reports = []
    for index, path in enumerate(paths, start=1):
        print(f"[{index}/{len(paths)}] {path}")
        report = process_trajectory(
            path,
            gate_cfg,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
            motion_baseline_mode=args.motion_baseline_mode,
            motion_baseline_num_frames=args.motion_baseline_num_frames,
            write_contact_segments=args.write_contact_segments,
        )
        reports.append(report)
        print(
            "  frames={frames} contact={contact_frames} first_on={first_contact_frame} "
            "motion_p50={motion_p50:.3f} motion_p90={motion_p90:.3f} motion_max={motion_max:.3f}".format(**report)
        )

    summary = _summary(reports, args, dry_run=args.dry_run)
    print(json.dumps(summary, indent=2))
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def process_trajectory(
    path: Path,
    gate_cfg: ContactGateConfig,
    *,
    dry_run: bool,
    overwrite: bool,
    motion_baseline_mode: str,
    motion_baseline_num_frames: int,
    write_contact_segments: bool,
) -> dict:
    with h5py.File(path, "r") as f:
        frames = f.get("frames")
        if frames is None:
            raise RuntimeError(f"{path}: missing frames group")
        missing = [
            name
            for name in ("tactile_left_marker_motion", "tactile_right_marker_motion")
            if name not in frames
        ]
        if missing:
            raise RuntimeError(
                f"{path}: missing existing marker-motion dataset(s): "
                + ", ".join(f"frames/{name}" for name in missing)
            )
        left_motion = np.asarray(frames["tactile_left_marker_motion"][:], dtype=np.float32).reshape(-1)
        right_motion = np.asarray(frames["tactile_right_marker_motion"][:], dtype=np.float32).reshape(-1)
        gripper_position = _read_gripper_position(f)
        frame_count = int(f.attrs.get("frame_count", 0)) or max(len(left_motion), len(right_motion))

    left_motion = _pad_to_length(left_motion, frame_count)
    right_motion = _pad_to_length(right_motion, frame_count)
    motion, motion_baseline = _contact_motion(
        left_motion,
        right_motion,
        mode=motion_baseline_mode,
        baseline_num_frames=motion_baseline_num_frames,
    )
    gate = contact_gate_from_motion(
        motion,
        gate_config=gate_cfg,
        gripper_position=gripper_position,
    )
    segment_ids = _contact_gate_segment_ids(gate) if write_contact_segments else None

    if not dry_run:
        with h5py.File(path, "a") as f:
            frames = f.require_group("frames")
            datasets = [
                ("contact_marker_motion", motion),
                ("contact_gate", gate),
            ]
            if segment_ids is not None:
                datasets.append(("contact_gate_segment_id", segment_ids))
            for name, values in datasets:
                if name in frames:
                    if not overwrite:
                        raise RuntimeError(f"{path}: frames/{name} already exists; pass --overwrite")
                    del frames[name]
                dtype = np.int32 if name == "contact_gate_segment_id" else np.float32
                frames.create_dataset(name, data=values.astype(dtype), compression="gzip")
            f.attrs["contact_gate_config"] = json.dumps(
                {
                    "marker_motion_source": "existing_frames_tactile_left_right_marker_motion",
                    "motion_baseline_mode": motion_baseline_mode,
                    "motion_baseline_num_frames": int(motion_baseline_num_frames),
                    "left_motion_baseline": motion_baseline["left"],
                    "right_motion_baseline": motion_baseline["right"],
                    "contact_on_threshold": gate_cfg.contact_on_threshold,
                    "contact_off_threshold": gate_cfg.contact_off_threshold,
                    "contact_on_consecutive_frames": gate_cfg.contact_on_consecutive_frames,
                    "contact_off_consecutive_frames": gate_cfg.contact_off_consecutive_frames,
                    "gripper_on_threshold": gate_cfg.gripper_on_threshold,
                    "gripper_open_threshold": gate_cfg.gripper_open_threshold,
                    "gripper_closed_threshold": gate_cfg.gripper_closed_threshold,
                    "gripper_hold_threshold": gate_cfg.gripper_hold_threshold,
                    "gripper_open_consecutive_frames": gate_cfg.gripper_open_consecutive_frames,
                    "hold_contact_while_gripper_closed": gate_cfg.hold_contact_while_gripper_closed,
                    "write_contact_segments": bool(write_contact_segments),
                },
                sort_keys=True,
            )

    contact_indices = np.flatnonzero(gate > 0.5)
    noncontact_segments = _count_noncontact_segments(gate)
    return {
        "path": str(path),
        "frames": int(frame_count),
        "contact_frames": int(np.count_nonzero(gate > 0.5)),
        "first_contact_frame": int(contact_indices[0]) if len(contact_indices) else -1,
        "noncontact_segments": int(noncontact_segments),
        "motion_p50": float(np.percentile(motion, 50)) if len(motion) else 0.0,
        "motion_p90": float(np.percentile(motion, 90)) if len(motion) else 0.0,
        "motion_p95": float(np.percentile(motion, 95)) if len(motion) else 0.0,
        "motion_max": float(np.max(motion)) if len(motion) else 0.0,
        "left_motion_baseline": motion_baseline["left"],
        "right_motion_baseline": motion_baseline["right"],
    }


def _read_gripper_position(f: h5py.File) -> np.ndarray | None:
    frames = f.get("frames")
    if frames is None:
        return None
    if "gripper_position" in frames:
        return np.asarray(frames["gripper_position"][:], dtype=np.float32).reshape(-1)
    if "joint_positions" in frames:
        joints = np.asarray(frames["joint_positions"][:], dtype=np.float32)
        if joints.ndim == 2 and joints.shape[1] > 6:
            return joints[:, -1]
    return None


def _trajectory_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    if (root / "trajectory.h5").exists():
        return [root / "trajectory.h5"]
    return sorted(root.glob("**/trajectory.h5"))


def _pad_to_length(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) >= length:
        return values[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(values)] = values
    return out


def _contact_gate_segment_ids(gate: np.ndarray) -> np.ndarray:
    """Return ids for each gate=0 run and -1 while contact is active."""
    values = np.asarray(gate, dtype=np.float32).reshape(-1)
    out = np.full(len(values), -1, dtype=np.int32)
    segment_id = -1
    in_noncontact = False
    for i, value in enumerate(values):
        if value <= 0.5:
            if not in_noncontact:
                segment_id += 1
                in_noncontact = True
            out[i] = segment_id
        else:
            in_noncontact = False
    return out


def _count_noncontact_segments(gate: np.ndarray) -> int:
    segment_ids = _contact_gate_segment_ids(gate)
    valid = segment_ids[segment_ids >= 0]
    return int(valid.max() + 1) if len(valid) else 0


def _contact_motion(
    left_motion: np.ndarray,
    right_motion: np.ndarray,
    *,
    mode: str,
    baseline_num_frames: int,
) -> tuple[np.ndarray, dict[str, float]]:
    left_motion = np.asarray(left_motion, dtype=np.float32)
    right_motion = np.asarray(right_motion, dtype=np.float32)
    if mode == "none":
        return np.maximum(left_motion, right_motion).astype(np.float32), {
            "left": 0.0,
            "right": 0.0,
        }

    baseline_n = max(1, int(baseline_num_frames))
    left_baseline = _early_median(left_motion, baseline_n)
    right_baseline = _early_median(right_motion, baseline_n)
    left_delta = np.maximum(left_motion - left_baseline, 0.0)
    right_delta = np.maximum(right_motion - right_baseline, 0.0)
    return np.maximum(left_delta, right_delta).astype(np.float32), {
        "left": float(left_baseline),
        "right": float(right_baseline),
    }


def _early_median(values: np.ndarray, num_frames: int) -> float:
    if len(values) == 0:
        return 0.0
    samples = values[: min(len(values), num_frames)]
    return float(np.median(samples))


def _summary(reports: list[dict], args: argparse.Namespace, *, dry_run: bool) -> dict:
    first_frames = [r["first_contact_frame"] for r in reports if r["first_contact_frame"] >= 0]
    contact_ratios = [r["contact_frames"] / max(r["frames"], 1) for r in reports]
    return {
        "dry_run": dry_run,
        "trajectory_count": len(reports),
        "total_frames": int(sum(r["frames"] for r in reports)),
        "trajectories_with_contact": int(sum(r["first_contact_frame"] >= 0 for r in reports)),
        "mean_contact_ratio": float(np.mean(contact_ratios)) if contact_ratios else 0.0,
        "median_first_contact_frame": float(np.median(first_frames)) if first_frames else -1.0,
        "thresholds": {
            "contact_on_threshold": args.contact_on_threshold,
            "contact_off_threshold": args.contact_off_threshold,
            "contact_on_consecutive_frames": args.contact_on_consecutive_frames,
            "contact_off_consecutive_frames": args.contact_off_consecutive_frames,
            "gripper_on_threshold": args.gripper_on_threshold,
            "gripper_open_threshold": args.gripper_open_threshold,
            "gripper_closed_threshold": args.gripper_closed_threshold,
            "gripper_hold_threshold": args.gripper_hold_threshold,
            "gripper_open_consecutive_frames": args.gripper_open_consecutive_frames,
            "hold_contact_while_gripper_closed": not args.disable_gripper_hold,
            "write_contact_segments": args.write_contact_segments,
            "motion_baseline_mode": args.motion_baseline_mode,
            "motion_baseline_num_frames": args.motion_baseline_num_frames,
        },
        "episodes": reports,
    }


if __name__ == "__main__":
    main()
