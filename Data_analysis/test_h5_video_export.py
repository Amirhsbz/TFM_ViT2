#!/usr/bin/env python3
"""Export and verify embedded videos from a TeleUR trajectory H5 file.
用--combined-only模式，只保存combined_2x2.mp4，临时的流文件会被删除。
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np

DEPTH_DISPLAY_MIN_MM = 200
DEPTH_DISPLAY_MAX_MM = 1500


def export_videos(h5_path: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    with h5py.File(h5_path, "r") as f:
        if "videos" not in f:
            raise ValueError(f"No 'videos' group found in {h5_path}")

        frame_rgb_keys = []
        if "frames" in f:
            for key in f["frames"].keys():
                if key.endswith("_rgb"):
                    frame_rgb_keys.append(key)

        if frame_rgb_keys:
            print("Warning: RGB data still present in /frames:")
            for key in frame_rgb_keys:
                ds = f["frames"][key]
                print(f"  - {key}: shape={ds.shape}, dtype={ds.dtype}")

        if len(f["videos"]) == 0:
            print("No embedded videos found under /videos")
            return 0

        print(f"Found {len(f['videos'])} embedded video stream(s)")
        for key, ds in sorted(f["videos"].items()):
            codec = str(ds.attrs.get("codec", "mp4v"))
            ext = ".avi" if codec == "MJPG" else ".mp4"
            output_path = output_dir / f"{key}{ext}"
            video_bytes = np.asarray(ds, dtype=np.uint8).tobytes()
            output_path.write_bytes(video_bytes)
            attrs = dict(ds.attrs)
            print(f"Exported {key} -> {output_path}")
            verify_video(output_path, key, attrs)
            exported.append(
                {
                    "key": key,
                    "path": output_path,
                    "attrs": attrs,
                }
            )

    return exported


def verify_video(video_path: Path, key: str, attrs: dict) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Exported video cannot be opened: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, _ = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Exported video opens but first frame cannot be read: {video_path}")

    print(
        f"  Verified {key}: "
        f"{width}x{height}, fps={fps:.2f}, frames={frame_count}, "
        f"stored_attrs={attrs}"
    )


def _fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    src_h, src_w = frame.shape[:2]
    if src_h == 0 or src_w == 0:
        return canvas

    scale = min(width / src_w, height / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (height - new_h) // 2
    x0 = (width - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas


def _resolve_stream_paths(output_dir: Path, preferred_order: list[str]) -> list[Path]:
    video_paths = []
    for name in preferred_order:
        stem = Path(name).stem
        # Try .avi first (MJPG codec from newer recordings), then .mp4 (legacy mp4v)
        for ext in (".avi", ".mp4"):
            path = output_dir / (stem + ext)
            if path.exists():
                video_paths.append(path)
                break
    return video_paths


def export_combined_video(
    output_dir: Path,
    fps: float,
    preferred_order: list[str],
    output_name: str,
) -> Path | None:
    video_paths = _resolve_stream_paths(output_dir, preferred_order)
    if len(video_paths) != 4:
        print(
            "Skipping combined video: expected 4 streams "
            f"({', '.join(preferred_order)}), found {len(video_paths)}"
        )
        return None

    captures = [cv2.VideoCapture(str(path)) for path in video_paths]
    try:
        for path, cap in zip(video_paths, captures):
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open exported video for stitching: {path}")

        widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) for cap in captures]
        heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) for cap in captures]
        frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in captures]

        tile_w = max(widths)
        tile_h = max(heights)
        out_size = (tile_w * 2, tile_h * 2)
        out_path = output_dir / output_name
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            out_size,
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open combined video writer: {out_path}")

        try:
            total_frames = min(frame_counts)
            for _ in range(total_frames):
                frames = []
                for cap in captures:
                    ok, frame = cap.read()
                    if not ok:
                        frames = []
                        break
                    frames.append(frame)
                if len(frames) != 4:
                    break

                tiles = [
                    _fit_frame(frame, tile_w, tile_h)
                    for frame in frames
                ]
                top = np.hstack([tiles[0], tiles[1]])
                bottom = np.hstack([tiles[2], tiles[3]])
                writer.write(np.vstack([top, bottom]))
        finally:
            writer.release()
    finally:
        for cap in captures:
            cap.release()

    print(f"Combined 4 streams -> {out_path}")
    verify_video(out_path, output_name, {"layout": "2x2", "inputs": [p.name for p in video_paths]})
    return out_path


def export_combined_only(
    h5_path: Path,
    output_dir: Path,
    fps: float,
    preferred_order: list[str],
    output_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / output_name

    with tempfile.TemporaryDirectory(
        prefix=f".{h5_path.stem}_combined_",
        dir=output_dir.parent,
    ) as tmp:
        tmp_output_dir = Path(tmp)
        export_videos(h5_path, tmp_output_dir)
        combined_path = export_combined_video(
            tmp_output_dir,
            fps,
            preferred_order,
            output_name,
        )
        if combined_path is None or not combined_path.is_file():
            raise FileNotFoundError(f"Combined video was not generated: {combined_path}")

        shutil.copy2(combined_path, final_path)

    print(f"Done. Exported combined video only to {final_path}")
    return final_path


def _decode_depth_frame(frame: np.ndarray, encoding: str, source_dtype: str) -> np.ndarray:
    if encoding == "depth_uint16_hi_lo":
        frame_u16 = frame.astype(np.uint16)
        return (frame_u16[..., 0] << 8) | frame_u16[..., 1]
    if encoding == "depth_uint8_gray":
        return frame[..., 0].astype(np.uint8)
    raise ValueError(f"Unsupported depth encoding: {encoding}, source_dtype={source_dtype}")


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth_mm = depth.astype(np.float32)
    valid = depth_mm > 0

    depth_clipped = np.clip(depth_mm, DEPTH_DISPLAY_MIN_MM, DEPTH_DISPLAY_MAX_MM)
    depth_scaled = (
        (depth_clipped - DEPTH_DISPLAY_MIN_MM)
        * 255.0
        / (DEPTH_DISPLAY_MAX_MM - DEPTH_DISPLAY_MIN_MM)
    )
    depth_uint8 = np.zeros(depth.shape, dtype=np.uint8)
    depth_uint8[valid] = depth_scaled[valid].astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)


def export_depth_visualizations(stream_infos: list[dict], output_dir: Path) -> list[dict]:
    depth_outputs = []
    for info in stream_infos:
        attrs = info["attrs"]
        if attrs.get("source_kind") != "depth":
            continue

        encoding = str(attrs.get("encoding", ""))
        source_dtype = str(attrs.get("source_dtype", ""))
        input_path = info["path"]
        output_path = output_dir / f"{info['key']}_viz.mp4"

        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open depth video for visualization: {input_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 10.0,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Failed to open depth visualization writer: {output_path}")

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                depth = _decode_depth_frame(frame, encoding, source_dtype)
                color = _colorize_depth(depth)
                writer.write(color)
        finally:
            writer.release()
            cap.release()

        print(f"Exported depth visualization {info['key']} -> {output_path}")
        verify_video(
            output_path,
            f"{info['key']}_viz",
            {"source": info["path"].name, "visualization": "depth_colormap"},
        )
        depth_outputs.append(
            {
                "key": f"{info['key']}_viz",
                "path": output_path,
            }
        )

    return depth_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and verify videos embedded in a TeleUR trajectory.h5 file."
    )
    parser.add_argument("h5_path", type=Path, help="Path to trajectory.h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write exported mp4 files into. Defaults to <h5_dir>/exported_videos",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only write combined_2x2.mp4 to the output directory; temporary stream files are removed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (
            h5_path.parent
            if args.combined_only
            else h5_path.parent / "exported_videos"
        )
    )

    with h5py.File(h5_path, "r") as f:
        fps = float(f.attrs.get("video_fps", 10.0))

    rgb_order = [
        "base_camera_rgb_0.mp4",
        "base_camera_rgb_1.mp4",
        "tactile_left_rgb.mp4",
        "tactile_right_rgb.mp4",
    ]
    if args.combined_only:
        export_combined_only(
            h5_path,
            output_dir,
            fps,
            rgb_order,
            "combined_2x2.mp4",
        )
        return

    exported = export_videos(h5_path, output_dir)
    export_combined_video(
        output_dir,
        fps,
        rgb_order,
        "combined_2x2.mp4",
    )

    depth_visualizations = export_depth_visualizations(exported, output_dir)
    if depth_visualizations:
        export_combined_video(
            output_dir,
            fps,
            [
                "base_camera_depth_0_viz.mp4",
                "base_camera_depth_1_viz.mp4",
                "tactile_left_depth_viz.mp4",
                "tactile_right_depth_viz.mp4",
            ],
            "combined_depth_2x2.mp4",
        )

    print(f"Done. Exported {len(exported)} embedded video(s) to {output_dir}")


if __name__ == "__main__":
    main()
