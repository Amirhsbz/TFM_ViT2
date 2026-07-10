#!/usr/bin/env python3
"""Create a copy of an H5 BC dataset with pre-contact tactile images gated.

Examples:
  conda run -n robodiff python3 Data_analysis/make_contact_gated_tactile_dataset.py \
    shared/data/bc_data/wipe_board \
    shared/data/bc_data/wipe_board_gated_tactile

  conda run -n robodiff python3 Data_analysis/make_contact_gated_tactile_dataset.py \
    shared/data/bc_data/wipe_board \
    --mode black

The input H5 files must already contain frames/contact_gate. Frames with
contact_gate <= 0.5 are replaced in these embedded videos:
  videos/tactile_left_rgb
  videos/tactile_right_rgb

By default, each tactile stream uses that episode's first pre-contact frames as
a neutral baseline: median(contact_gate==0 first 20 frames). Pass --mode black to
write a *_gated_black inspection dataset where contact_gate=0 frames are black.
The older --mode zero spelling is kept as an alias for --mode black.

All other files, H5 groups, datasets, and attrs are copied unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np


TACTILE_VIDEO_KEYS = ("tactile_left_rgb", "tactile_right_rgb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an H5 BC dataset and replace tactile video frames before contact."
    )
    parser.add_argument("input_root", type=Path)
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Output dataset path. Defaults to *_gated_tactile for baseline mode "
            "and *_gated_black for black mode."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace output_root if it already exists.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for a JSON processing summary.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "black", "zero"],
        default="baseline",
        help=(
            "Replacement for contact_gate=0 frames. Default: per-episode tactile baseline. "
            "'black' writes all-black frames; 'zero' is a deprecated alias."
        ),
    )
    parser.add_argument(
        "--baseline-num-frames",
        type=int,
        default=20,
        help="Number of earliest pre-contact frames used to compute the baseline.",
    )
    parser.add_argument(
        "--baseline-stat",
        choices=["median", "mean", "first"],
        default="median",
        help="Statistic used for the per-episode baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root
    mode = _normalize_mode(args.mode)
    output_root = args.output_root or _default_output_root(input_root, mode)
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {output_root}. Pass --overwrite to replace it.")
        if output_root.is_dir():
            shutil.rmtree(output_root)
        else:
            output_root.unlink()

    print(f"Copying {input_root} -> {output_root}")
    if input_root.is_file():
        output_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_root, output_root)
        h5_paths = [output_root]
    else:
        shutil.copytree(input_root, output_root)
        h5_paths = sorted(output_root.glob("**/trajectory.h5"))

    if not h5_paths:
        raise SystemExit(f"No trajectory.h5 files found under copied output: {output_root}")

    reports = []
    for index, h5_path in enumerate(h5_paths, start=1):
        print(f"[{index}/{len(h5_paths)}] {h5_path}")
        report = replace_precontact_tactile_videos(
            h5_path,
            mode=mode,
            baseline_num_frames=args.baseline_num_frames,
            baseline_stat=args.baseline_stat,
        )
        reports.append(report)
        print(
            "  frames={frames} replaced={replaced_frames} contact={contact_frames} "
            "videos={videos_rewritten}".format(**report)
        )

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "trajectory_count": len(reports),
        "total_frames": int(sum(r["frames"] for r in reports)),
        "mode": mode,
        "baseline_num_frames": args.baseline_num_frames,
        "baseline_stat": args.baseline_stat,
        "total_replaced_frames": int(sum(r["replaced_frames"] for r in reports)),
        "total_contact_frames": int(sum(r["contact_frames"] for r in reports)),
        "episodes": reports,
    }
    summary_path = args.summary_json or _default_summary_path(output_root)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


def _default_summary_path(output_root: Path) -> Path:
    if output_root.suffix:
        return output_root.with_name(f"{output_root.stem}_summary.json")
    if output_root.name.endswith("_gated_black"):
        return output_root / "gated_black_summary.json"
    return output_root / "gated_tactile_summary.json"


def _normalize_mode(mode: str) -> str:
    return "black" if mode == "zero" else mode


def _default_output_root(input_root: Path, mode: str) -> Path:
    suffix = "_gated_black" if mode == "black" else "_gated_tactile"
    if input_root.suffix:
        return input_root.with_name(f"{input_root.stem}{suffix}{input_root.suffix}")
    return input_root.with_name(f"{input_root.name}{suffix}")


def replace_precontact_tactile_videos(
    h5_path: Path,
    *,
    mode: str,
    baseline_num_frames: int,
    baseline_stat: str,
) -> dict:
    with h5py.File(h5_path, "a") as f:
        if "frames" not in f or "contact_gate" not in f["frames"]:
            raise RuntimeError(f"{h5_path}: missing frames/contact_gate")
        gate = np.asarray(f["frames"]["contact_gate"][:], dtype=np.float32).reshape(-1)
        replace_mask = gate <= 0.5
        videos_rewritten = 0
        for key in TACTILE_VIDEO_KEYS:
            if "videos" not in f or key not in f["videos"]:
                continue
            ds = f["videos"][key]
            attrs = {name: ds.attrs[name] for name in ds.attrs}
            frames = _decode_h5_video_frames(ds)
            n = min(len(frames), len(replace_mask))
            if n > 0:
                if mode == "black":
                    replacement = np.zeros_like(frames[0])
                else:
                    replacement = _compute_tactile_baseline(
                        frames,
                        replace_mask,
                        num_frames=baseline_num_frames,
                        stat=baseline_stat,
                    )
                frames[:n][replace_mask[:n]] = replacement
            encoded = _encode_video_frames(frames, attrs)
            del f["videos"][key]
            out_ds = f["videos"].create_dataset(key, data=encoded, dtype=np.uint8)
            for name, value in attrs.items():
                out_ds.attrs[name] = value
            videos_rewritten += 1
        f.attrs["tactile_precontact_replacement"] = mode
        f.attrs["tactile_precontact_baseline_num_frames"] = int(baseline_num_frames)
        f.attrs["tactile_precontact_baseline_stat"] = baseline_stat

    return {
        "path": str(h5_path),
        "frames": int(len(gate)),
        "replaced_frames": int(np.count_nonzero(replace_mask)),
        "contact_frames": int(np.count_nonzero(~replace_mask)),
        "videos_rewritten": int(videos_rewritten),
    }


def _compute_tactile_baseline(
    frames: np.ndarray,
    replace_mask: np.ndarray,
    *,
    num_frames: int,
    stat: str,
) -> np.ndarray:
    precontact_indices = np.flatnonzero(replace_mask[: len(frames)])
    if len(precontact_indices) == 0:
        return np.zeros_like(frames[0])
    indices = precontact_indices[: max(1, int(num_frames))]
    samples = frames[indices].astype(np.float32)
    if stat == "first":
        baseline = samples[0]
    elif stat == "mean":
        baseline = samples.mean(axis=0)
    else:
        baseline = np.median(samples, axis=0)
    return np.clip(np.rint(baseline), 0, 255).astype(np.uint8)


def _decode_h5_video_frames(dataset) -> np.ndarray:
    video_bytes = np.asarray(dataset, dtype=np.uint8).tobytes()
    codec = str(dataset.attrs.get("codec", "mp4v"))
    suffix = ".avi" if codec == "MJPG" else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open embedded video stream '{dataset.name}'")
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    if not frames:
        raise RuntimeError(f"No frames decoded from embedded video stream '{dataset.name}'")
    return np.stack(frames, axis=0)


def _encode_video_frames(frames: np.ndarray, attrs: dict) -> np.ndarray:
    codec = str(attrs.get("codec", "mp4v"))
    fps = float(attrs.get("fps", 15.0))
    height = int(attrs.get("height", frames.shape[1]))
    width = int(attrs.get("width", frames.shape[2]))
    suffix = ".avi" if codec == "MJPG" else ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*codec)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height), isColor=True)
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open temporary video writer for codec {codec}")
        for frame in frames:
            rgb = np.asarray(frame)
            if rgb.shape[:2] != (height, width):
                rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
        return np.fromfile(tmp_path, dtype=np.uint8)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
