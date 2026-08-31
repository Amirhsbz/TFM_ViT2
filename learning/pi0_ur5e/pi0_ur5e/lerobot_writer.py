from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .io_utils import resize_rgb, write_json
from .schema import Episode


def write_lerobot_dataset(
    episodes: list[Episode],
    output_root: str | Path,
    *,
    task_name: str,
    image_size: tuple[int, int] = (224, 224),
    include_tactile: bool = False,
    camera_padding_strategy: str = "duplicate_base",
    repo_id: str = "local/pi0_ur5e_cup",
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write a real LeRobot v2 dataset readable by OpenPI's data loader."""
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "Writing an OpenPI-trainable LeRobot dataset requires LeRobot. "
            "Run this script with the OpenPI uv environment, e.g. "
            "`cd /home/rpl/yongqiang/openpi && uv run python "
            "/home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py ...`."
        ) from exc

    output = Path(output_root)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists. Pass --overwrite true to replace it.")
        shutil.rmtree(output)

    states_by_episode = []
    actions_by_episode = []
    lengths = []
    dt_values = []
    tactile_shape = None
    for episode in episodes:
        state = episode.robot_state.astype(np.float32)
        if include_tactile and episode.tactile is not None:
            state = np.concatenate([state, episode.tactile.astype(np.float32)], axis=1)
            tactile_shape = list(episode.tactile.shape)
        action = episode.action.astype(np.float32)
        states_by_episode.append(state)
        actions_by_episode.append(action)
        lengths.append(len(episode.timestamps))
        if len(episode.timestamps) > 1:
            dt_values.append(np.diff(episode.timestamps.astype(np.float64)))

    states = np.concatenate(states_by_episode, axis=0) if states_by_episode else np.empty((0, 0), dtype=np.float32)
    actions = np.concatenate(actions_by_episode, axis=0) if actions_by_episode else np.empty((0, 0), dtype=np.float32)
    dts = np.concatenate(dt_values, axis=0) if dt_values else np.empty((0,), dtype=np.float64)
    fps = _infer_fps(episodes, dts)
    image_shape = (image_size[1], image_size[0], 3)
    has_raw_tactile = include_tactile and any(
        episode.tactile_left_rgb is not None or episode.tactile_right_rgb is not None for episode in episodes
    )

    features = {
        "observation.images.base_rgb": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (int(states.shape[1]),),
            "names": ["state"],
        },
        "action": {
            "dtype": "float32",
            "shape": (int(actions.shape[1]),),
            "names": ["action"],
        },
    }
    if has_raw_tactile:
        features["observation.images.tactile_left_rgb"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }
        features["observation.images.tactile_right_rgb"] = {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type="ur5e",
        features=features,
        use_videos=False,
    )

    for episode, state in zip(episodes, states_by_episode, strict=True):
        for t in range(len(episode.timestamps)):
            frame = {
                "observation.images.base_rgb": _image_from_episode(
                    episode.base_rgb,
                    t,
                    image_size,
                ),
                "observation.images.wrist_rgb": _image_from_episode(
                    episode.wrist_rgb,
                    t,
                    image_size,
                ),
                "observation.state": state[t].astype(np.float32),
                "action": episode.action[t].astype(np.float32),
                "task": episode.language_instruction or task_name,
            }
            if has_raw_tactile:
                frame["observation.images.tactile_left_rgb"] = _image_from_episode(
                    episode.tactile_left_rgb,
                    t,
                    image_size,
                )
                frame["observation.images.tactile_right_rgb"] = _image_from_episode(
                    episode.tactile_right_rgb,
                    t,
                    image_size,
                )
            dataset.add_frame(frame)
        dataset.save_episode()
    if getattr(dataset, "image_writer", None) is not None:
        dataset.stop_image_writer()

    image_shape_report = {
        "observation.images.base_rgb": list(image_shape),
        "observation.images.wrist_rgb": list(image_shape),
    }
    if has_raw_tactile:
        image_shape_report["observation.images.tactile_left_rgb"] = list(image_shape)
        image_shape_report["observation.images.tactile_right_rgb"] = list(image_shape)

    report = {
        "episode_count": len(episodes),
        "total_frames": int(sum(lengths)),
        "trajectory_lengths": lengths,
        "action": _array_stats(actions),
        "state": _array_stats(states),
        "has_nan_or_inf": bool((not np.isfinite(actions).all()) or (not np.isfinite(states).all())),
        "image_shape": image_shape_report,
        "tactile_shape": tactile_shape,
        "timestamp_dt": _array_stats(dts),
        "camera_padding_strategy": camera_padding_strategy,
        "action_format": episodes[0].metadata.get("action_type") if episodes else None,
        "repo_id": repo_id,
        "fps": fps,
        "format": "lerobot_v2",
        "openpi_image_mapping": {
            "cam_high/base": "observation.images.base_rgb",
            "wrist": "observation.images.wrist_rgb",
            "missing_third_camera": camera_padding_strategy,
        },
    }
    write_json(output / "conversion_report.json", report)
    return report


def _image_from_episode(source: Any, index: int, size: tuple[int, int]) -> np.ndarray:
    if source is None:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if isinstance(source, list) and source and isinstance(source[index], str):
        try:
            import cv2

            image = cv2.cvtColor(cv2.imread(source[index]), cv2.COLOR_BGR2RGB)
        except Exception as exc:
            raise RuntimeError(f"Failed to read image path {source[index]}") from exc
    else:
        image = np.asarray(source[index])
    return resize_rgb(image, size)


def _array_stats(arr: np.ndarray) -> dict[str, Any]:
    if arr.size == 0:
        return {}
    return {
        "min": np.nanmin(arr, axis=0).astype(float).tolist() if arr.ndim > 1 else float(np.nanmin(arr)),
        "max": np.nanmax(arr, axis=0).astype(float).tolist() if arr.ndim > 1 else float(np.nanmax(arr)),
        "mean": np.nanmean(arr, axis=0).astype(float).tolist() if arr.ndim > 1 else float(np.nanmean(arr)),
        "std": np.nanstd(arr, axis=0).astype(float).tolist() if arr.ndim > 1 else float(np.nanstd(arr)),
        "has_nan": bool(np.isnan(arr).any()),
        "has_inf": bool(np.isinf(arr).any()),
    }


def _infer_fps(episodes: list[Episode], dt: np.ndarray) -> int:
    for episode in episodes:
        hz = episode.metadata.get("hz")
        if hz:
            return max(int(round(float(hz))), 1)
    if dt.size and np.any(dt > 0):
        return max(int(round(1.0 / float(np.mean(dt[dt > 0])))), 1)
    return 10
