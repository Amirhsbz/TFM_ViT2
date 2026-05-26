#!/usr/bin/env python3
"""Select tactile crop configs from live camera frames for real-robot testing.

This writes configs that can be passed to run_env.py with:

  python run_env.py \
    --tactile-crop-config-dir sensor_configs/put_bottle_upright_test \
    --enable-marker-tracking \
    --use-marker-tracking-overlay-for-policy

Selection order:
  1. top-left
  2. top-right
  3. bottom-right
  4. bottom-left

Keys:
  u = undo last point
  r = restart current image
  Enter or s = save current image points
  q or Esc = quit
  
测试的时候重新选点：
python Data_analysis/select_tactile_crop_from_cameras.py \
  --config-dir sensor_configs/put_bottle_upright_test \
  --output-size 320x240 \
  --overwrite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cameras.opencv_camera import OpenCVCamera


TACTILE_CAM_PORTS = {
    "left": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:5.4:1.0-video-index0",
    "right": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.3.4:1.0-video-index0",
    "2": 2,
    "4": 4,
}


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected WIDTHxHEIGHT, e.g. 320x240") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select four-point tactile crop configs from current live camera frames."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory to write tactile_left_rgb.json and tactile_right_rgb.json.",
    )
    parser.add_argument("--left-camera-id", default="left")
    parser.add_argument("--right-camera-id", default="right")
    parser.add_argument("--width", type=int, default=640, help="Camera capture width.")
    parser.add_argument("--height", type=int, default=480, help="Camera capture height.")
    parser.add_argument(
        "--output-size",
        type=parse_size,
        default=(320, 240),
        help="Warp output size saved in config as WIDTHxHEIGHT. Default: 320x240.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=10,
        help="Frames to discard before capturing the selection frame.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_camera_id(camera_identifier: str | int) -> str | int:
    if isinstance(camera_identifier, int):
        return camera_identifier
    if camera_identifier in TACTILE_CAM_PORTS:
        return TACTILE_CAM_PORTS[camera_identifier]
    try:
        return int(camera_identifier)
    except (TypeError, ValueError):
        return camera_identifier


def select_points(image_rgb: np.ndarray, window_name: str) -> list[list[int]]:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    display = image_bgr.copy()
    points: list[list[int]] = []

    def redraw() -> None:
        nonlocal display
        display = image_bgr.copy()
        for idx, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(idx + 1),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if len(points) > 1:
            for a, b in zip(points, points[1:]):
                cv2.line(display, tuple(a), tuple(b), (255, 0, 0), 1)
        if len(points) == 4:
            cv2.line(display, tuple(points[-1]), tuple(points[0]), (255, 0, 0), 1)
        cv2.imshow(window_name, display)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([int(x), int(y)])
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    print(f"\n{window_name}: click top-left, top-right, bottom-right, bottom-left")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            raise KeyboardInterrupt("Selection cancelled")
        if key == ord("u") and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key in (13, 10, ord("s")):
            if len(points) == 4:
                cv2.destroyWindow(window_name)
                return points
            print(f"{window_name}: need 4 points before saving, got {len(points)}")


def capture_frame(camera_id: str | int, width: int, height: int, warmup_frames: int) -> np.ndarray:
    camera = OpenCVCamera(camera_id=camera_id, width=width, height=height)
    try:
        frame = None
        for _ in range(max(warmup_frames, 1)):
            frame, _ = camera.read()
        if frame is None:
            raise RuntimeError(f"Could not capture frame from camera {camera_id}")
        return frame
    finally:
        camera.release()


def write_config(
    config_path: Path,
    points: list[list[int]],
    output_size: tuple[int, int],
    source: dict[str, Any],
    overwrite: bool,
) -> None:
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"Config already exists: {config_path}. Use --overwrite.")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": points,
        "output_size": list(output_size),
        "point_order": "top_left, top_right, bottom_right, bottom_left",
        **source,
    }
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {config_path}")


def main() -> None:
    args = parse_args()
    cameras = {
        "tactile_left": resolve_camera_id(args.left_camera_id),
        "tactile_right": resolve_camera_id(args.right_camera_id),
    }

    for sensor_name, camera_id in cameras.items():
        print(f"Capturing {sensor_name} from {camera_id}")
        frame = capture_frame(camera_id, args.width, args.height, args.warmup_frames)
        points = select_points(frame, f"{sensor_name}_crop_select")
        write_config(
            args.config_dir / f"{sensor_name}_rgb.json",
            points,
            args.output_size,
            {
                "source": "live_camera",
                "camera_id": str(camera_id),
                "capture_size": [int(frame.shape[1]), int(frame.shape[0])],
            },
            args.overwrite,
        )


if __name__ == "__main__":
    main()
