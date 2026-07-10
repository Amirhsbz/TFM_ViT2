from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from marker_tracking.utils import find_marker, find_marker_centers


@dataclass
class MarkerMotionConfig:
    marker_flow_win_size: tuple[int, int] = (15, 15)
    marker_flow_max_level: int = 2
    marker_flow_fb_max_error: float = 1.5
    marker_mask_range: tuple[int, int] = (145, 255)
    marker_value_threshold: int = 90
    marker_morph_open_size: int = 5
    marker_morph_open_iter: int = 1
    marker_morph_close_size: int = 5
    marker_morph_close_iter: int = 1
    marker_dilate_size: int = 3
    marker_dilate_iter: int = 0
    marker_motion_deadband: float = 0.2
    marker_motion_smoothing: float = 0.2
    marker_motion_release_smoothing: float = 0.8
    marker_motion_min_valid_points: int = 8
    marker_motion_compensate_global_drift: bool = True
    marker_tracking_reset_on_loss: bool = True
    left_marker_broken_region: tuple[int, int, int, int] = (0, 0, 0, 0)
    right_marker_missing_right_cols: int = 0
    marker_detection_margin: int = 0


@dataclass
class ContactGateConfig:
    contact_on_threshold: float = 0.5
    contact_off_threshold: float = 0.2
    contact_on_consecutive_frames: int = 2
    contact_off_consecutive_frames: int = 8
    gripper_on_threshold: float = 0.2
    gripper_open_threshold: float = 0.08
    gripper_closed_threshold: float = 0.2
    gripper_hold_threshold: float = 0.2
    gripper_open_consecutive_frames: int = 2
    hold_contact_while_gripper_closed: bool = True


@dataclass
class _MarkerTrackingState:
    ref_gray: np.ndarray | None = None
    ref_points: np.ndarray | None = None
    motion_ema: float = 0.0


def marker_motion_from_tactile_frames(
    tactile_left_rgb: np.ndarray | None,
    tactile_right_rgb: np.ndarray | None,
    *,
    motion_config: MarkerMotionConfig | None = None,
) -> dict[str, np.ndarray]:
    """Compute run_env-compatible marker motion for tactile frame sequences."""
    cfg = motion_config or MarkerMotionConfig()
    left_motion = _motion_sequence("tactile_left", tactile_left_rgb, cfg)
    right_motion = _motion_sequence("tactile_right", tactile_right_rgb, cfg)
    length = max(len(left_motion), len(right_motion))
    left_motion = _pad_to_length(left_motion, length)
    right_motion = _pad_to_length(right_motion, length)
    sum_motion = np.maximum(left_motion, right_motion).astype(np.float32)
    return {
        "tactile_left_marker_motion": left_motion.astype(np.float32),
        "tactile_right_marker_motion": right_motion.astype(np.float32),
        "contact_marker_motion": sum_motion,
    }


def contact_gate_from_motion(
    motion: np.ndarray,
    *,
    gate_config: ContactGateConfig | None = None,
    gripper_position: np.ndarray | None = None,
) -> np.ndarray:
    cfg = gate_config or ContactGateConfig()
    on_n = max(1, int(cfg.contact_on_consecutive_frames))
    off_n = max(1, int(cfg.contact_off_consecutive_frames))
    gripper_open_n = max(1, int(cfg.gripper_open_consecutive_frames))
    gripper = _pad_optional_series(gripper_position, len(motion))
    gate = np.zeros(len(motion), dtype=np.float32)
    contact = False
    on_count = 0
    off_count = 0
    gripper_open_count = 0
    gripper_was_closed = False
    for i, value in enumerate(np.asarray(motion, dtype=np.float32)):
        if contact:
            if gripper is not None and gripper[i] >= cfg.gripper_closed_threshold:
                gripper_was_closed = True
            gripper_holding = (
                gripper is not None
                and cfg.hold_contact_while_gripper_closed
                and gripper[i] >= cfg.gripper_hold_threshold
            )
            if value < cfg.contact_off_threshold and not gripper_holding:
                off_count += 1
            else:
                off_count = 0
            if (
                gripper is not None
                and gripper_was_closed
                and gripper[i] <= cfg.gripper_open_threshold
            ):
                gripper_open_count += 1
            else:
                gripper_open_count = 0
            if off_count >= off_n or gripper_open_count >= gripper_open_n:
                contact = False
                off_count = 0
                gripper_open_count = 0
                gripper_was_closed = False
            on_count = 0
        else:
            gripper_ready = (
                gripper is None or gripper[i] >= cfg.gripper_on_threshold
            )
            if value > cfg.contact_on_threshold and gripper_ready:
                on_count += 1
            else:
                on_count = 0
            if on_count >= on_n:
                contact = True
                off_count = 0
                gripper_open_count = 0
                gripper_was_closed = (
                    gripper is not None and gripper[i] >= cfg.gripper_closed_threshold
                )
        gate[i] = 1.0 if contact else 0.0
    return gate


def apply_contact_gate_to_tactile_obs(
    obs: dict[str, Any],
    *,
    neutral_value: int = 0,
    neutral_images: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = obs.get("contact_gate")
    if gate is None:
        return obs
    gate_value = float(np.asarray(gate).reshape(-1)[-1])
    if gate_value > 0.5:
        return obs
    for key in ("tactile_left_rgb", "tactile_right_rgb"):
        if key in obs and obs[key] is not None:
            if neutral_images is not None and neutral_images.get(key) is not None:
                obs[key] = np.asarray(neutral_images[key], dtype=obs[key].dtype).copy()
            else:
                obs[key] = np.full_like(obs[key], neutral_value)
    return obs


def contact_gate_value(gate: Any) -> float:
    if gate is None:
        return 1.0
    return float(np.asarray(gate, dtype=np.float32).reshape(-1)[-1])


def _motion_sequence(sensor_name: str, frames: np.ndarray | None, cfg: MarkerMotionConfig) -> np.ndarray:
    if frames is None:
        return np.zeros(0, dtype=np.float32)
    frames = np.asarray(frames)
    if frames.ndim < 4:
        return np.zeros(0, dtype=np.float32)

    marker_params = _marker_tracking_params(cfg)
    lk_params = {
        "winSize": tuple(cfg.marker_flow_win_size),
        "maxLevel": int(cfg.marker_flow_max_level),
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }
    find_marker_kwargs = _sensor_find_marker_kwargs(sensor_name, cfg)
    state: _MarkerTrackingState | None = None
    motion = np.zeros(len(frames), dtype=np.float32)
    for i, frame in enumerate(frames):
        state, value = _update_marker_motion(
            sensor_name,
            _as_rgb_uint8(frame),
            state,
            marker_params,
            lk_params,
            cfg,
            find_marker_kwargs,
        )
        motion[i] = 0.0 if value is None else float(value)
    return motion


def _update_marker_motion(
    sensor_name: str,
    frame: np.ndarray,
    state: _MarkerTrackingState | None,
    marker_params: dict[str, Any],
    lk_params: dict[str, Any],
    cfg: MarkerMotionConfig,
    find_marker_kwargs: dict[str, Any],
) -> tuple[_MarkerTrackingState | None, float | None]:
    if state is None or state.ref_points is None or state.ref_gray is None:
        state = _init_marker_tracking_state(frame, marker_params, find_marker_kwargs)
        if state is None or state.ref_points is None or state.ref_gray is None:
            return state, None

    track_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(state.ref_gray, track_gray, state.ref_points, None, **lk_params)
    if next_points is None or status is None:
        return (_init_marker_tracking_state(frame, marker_params, find_marker_kwargs), None) if cfg.marker_tracking_reset_on_loss else (state, None)

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(track_gray, state.ref_gray, next_points, None, **lk_params)
    if back_points is None or back_status is None:
        return (_init_marker_tracking_state(frame, marker_params, find_marker_kwargs), None) if cfg.marker_tracking_reset_on_loss else (state, None)

    ref_points_all = state.ref_points.reshape(-1, 2)
    tracked_points_all = next_points.reshape(-1, 2)
    back_points_all = back_points.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(back_points_all - ref_points_all, axis=1)
    valid = status & back_status & (fb_error <= cfg.marker_flow_fb_max_error)

    if np.count_nonzero(valid) < cfg.marker_motion_min_valid_points:
        if cfg.marker_tracking_reset_on_loss:
            return _init_marker_tracking_state(frame, marker_params, find_marker_kwargs), None
        return state, 0.0

    deltas = tracked_points_all[valid] - ref_points_all[valid]
    if cfg.marker_motion_compensate_global_drift and len(deltas) > 0:
        deltas = deltas - np.median(deltas, axis=0, keepdims=True)

    delta_norms = np.linalg.norm(deltas, axis=1)
    delta_norms = np.clip(delta_norms - cfg.marker_motion_deadband, 0.0, None)
    sum_motion = float(delta_norms.sum()) if len(delta_norms) > 0 else 0.0
    alpha = cfg.marker_motion_release_smoothing if sum_motion < state.motion_ema else cfg.marker_motion_smoothing
    alpha = float(np.clip(alpha, 0.0, 1.0))
    state.motion_ema = (1.0 - alpha) * state.motion_ema + alpha * sum_motion
    return state, state.motion_ema


def _init_marker_tracking_state(
    frame: np.ndarray,
    marker_params: dict[str, Any],
    find_marker_kwargs: dict[str, Any],
) -> _MarkerTrackingState | None:
    marker_mask = find_marker(frame, **marker_params)
    centers = find_marker_centers(marker_mask, **find_marker_kwargs)
    if not centers:
        return _MarkerTrackingState()
    return _MarkerTrackingState(
        ref_gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
        ref_points=np.asarray(centers, dtype=np.float32).reshape(-1, 1, 2),
    )


def _marker_tracking_params(cfg: MarkerMotionConfig) -> dict[str, Any]:
    return {
        "morphop_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.marker_morph_open_size, cfg.marker_morph_open_size)),
        "morphclose_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.marker_morph_close_size, cfg.marker_morph_close_size)),
        "dilate_kernel": cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.marker_dilate_size, cfg.marker_dilate_size)),
        "mask_range": tuple(cfg.marker_mask_range),
        "min_value": cfg.marker_value_threshold,
        "morphop_iter": cfg.marker_morph_open_iter,
        "morphclose_iter": cfg.marker_morph_close_iter,
        "dilate_iter": cfg.marker_dilate_iter,
    }


def _sensor_find_marker_kwargs(sensor_name: str, cfg: MarkerMotionConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if sensor_name == "tactile_left":
        rs, re, cs, ce = cfg.left_marker_broken_region
        if rs > 0 and re >= rs and cs > 0 and ce >= cs:
            kwargs["broken_cells"] = {(r, c) for r in range(rs - 1, re) for c in range(cs - 1, ce)}
    elif sensor_name == "tactile_right" and cfg.right_marker_missing_right_cols > 0:
        kwargs["n_missing_right_cols"] = cfg.right_marker_missing_right_cols
    if cfg.marker_detection_margin > 0:
        kwargs["detection_margin"] = cfg.marker_detection_margin
    return kwargs


def _pad_to_length(values: np.ndarray, length: int) -> np.ndarray:
    if len(values) >= length:
        return values[:length]
    out = np.zeros(length, dtype=np.float32)
    out[: len(values)] = values
    return out


def _pad_optional_series(values: np.ndarray | None, length: int) -> np.ndarray | None:
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) == 0:
        return None
    if len(values) >= length:
        return values[:length]
    out = np.full(length, values[-1], dtype=np.float32)
    out[: len(values)] = values
    return out


def _as_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.ndim == 3 and frame.shape[0] in (1, 3) and frame.shape[-1] not in (1, 3):
        frame = np.moveaxis(frame, 0, -1)
    if frame.dtype != np.uint8:
        if np.issubdtype(frame.dtype, np.floating):
            frame = np.clip(frame, 0.0, 1.0) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    return np.ascontiguousarray(frame[..., :3])
