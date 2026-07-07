from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def load_yaml(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML config loading requires PyYAML") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_json(path: str | Path, data: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def load_pickle(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_timestamp(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    try:
        return float(text)
    except ValueError:
        pass
    try:
        from datetime import datetime

        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return float(fallback)


def decode_h5_video(dataset, *, is_depth: bool = False) -> np.ndarray:
    import cv2

    video_bytes = np.asarray(dataset, dtype=np.uint8).tobytes()
    codec = str(dataset.attrs.get("codec", "mp4v"))
    suffix = ".avi" if codec == "MJPG" else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)
    frames = []
    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open embedded video stream {dataset.name}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[..., 0] if is_depth else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    finally:
        tmp_path.unlink(missing_ok=True)
    return np.stack(frames, axis=0) if frames else np.empty((0,))


def resize_rgb(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image shape HxWx3 or NxHxWx3, got {image.shape}")
    try:
        import cv2

        return cv2.resize(image.astype(np.uint8), size, interpolation=cv2.INTER_AREA)
    except ImportError:
        height, width = image.shape[:2]
        out_w, out_h = size
        y = np.clip((np.arange(out_h) * height / out_h).astype(int), 0, height - 1)
        x = np.clip((np.arange(out_w) * width / out_w).astype(int), 0, width - 1)
        return image[y][:, x].astype(np.uint8)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image = image.astype(np.uint8)
    try:
        import cv2

        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    except ImportError:
        # Minimal fallback for dependency-light tests. The file extension may be
        # .png, but the content is PPM; install opencv-python for production PNGs.
        with open(path, "wb") as f:
            f.write(f"P6\n{image.shape[1]} {image.shape[0]}\n255\n".encode("ascii"))
            f.write(image.tobytes())
