import datetime
import json
import os
import pickle
import shutil
import socket
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import termcolor
import tyro

# foot pedal
from pynput import keyboard

from agents.agent import BimanualAgent, SafetyWrapper
from camera_node import ZMQClientCamera
from cameras.opencv_camera import OpenCVCamera
from cameras.realsense_camera import RealSenseCamera
from env import RobotEnv
from marker_tracking.utils import (
    find_marker,
    find_marker_centers,
    plot_marker_delta,
)
from robot_node import ZMQClientRobot
from udp_haptics_sender import clamp01, send_packet

trigger_state = {"r": False, "l": False, "s": False, "d": False}

# Mapping for tactile camera names to v4l2 by-path ports (persistent device paths)
# Change these paths to match your actual v4l/by-path devices
TACTILE_CAM_PORTS = {
    # "left": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.3.4:1.0-video-index0",
    "left": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:1.4:1.0-video-index0",
    # "right": "/dev/v4l/by-path/pci-0000:80:14.0-usbv2-0:5.4:1.0-video-index0",  # Update with your actual by-path
    "right": "/dev/v4l/by-path/pci-0000:80:14.0-usbv2-0:2:1.0-video-index0",  # Example for second UR tactile cam
    # Fallback to int IDs if needed
    "2": 2,
    "4": 4,
}


def _resolve_tactile_warp_config(config_path: str, sensor_name: str):
    """Resolve tactile warp config path with simple project-local defaults."""
    sensor_suffix = sensor_name.replace("tactile_", "")
    candidates = [
        f"robo_test/sensor_config_{sensor_suffix}.json",
        f"robo_test/{sensor_name}_sensor_config.json",
        f"robo_test/{sensor_suffix}_sensor_config.json",
        f"robo_test/sensor_config_{sensor_name}.json",
        "robo_test/sensor_config.json",
    ]

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"No tactile warp config found for {sensor_name}. "
        "Expected one of: "
        + ", ".join(candidates)
    )


def _load_tactile_crop_configs(config_dir: str):
    """Load task-specific tactile camera crop config paths."""
    if not config_dir:
        return {}

    config_dir = Path(config_dir).expanduser()
    configs = {}
    for sensor_name in ("tactile_left", "tactile_right"):
        suffix = sensor_name.replace("tactile_", "")
        candidates = [
            config_dir / f"{sensor_name}_rgb.json",
            config_dir / f"{sensor_name}.json",
            config_dir / f"sensor_config_{suffix}.json",
            config_dir / f"sensor_config_{sensor_name}.json",
        ]
        config_path = next((path for path in candidates if path.exists()), None)
        if config_path is None:
            raise FileNotFoundError(
                f"No tactile crop config found for {sensor_name} in {config_dir}. "
                f"Expected one of: {', '.join(str(path) for path in candidates)}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        entry = config.get(sensor_name) or config.get(sensor_name.replace("tactile_", "")) or config
        points = np.array(entry["points"], dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError(f"{config_path} points must have shape (4, 2), got {points.shape}")
        out_width, out_height = entry.get("output_size", entry.get("size", (320, 240)))
        configs[sensor_name] = {
            "path": config_path,
            "size": (int(out_width), int(out_height)),
        }
    return configs


def _apply_tactile_crop_configs(
    obs: Dict[str, np.ndarray],
    crop_configs: dict,
    input_size: tuple[int, int],
) -> None:
    # Crop configs are applied inside OpenCVCamera as the tactile camera's only
    # perspective transform. Keeping this hook as a no-op preserves the call
    # sites while preventing accidental second crops.
    return


def _resolve_camera_id(camera_identifier):
    """Resolve camera identifier to either int device ID or v4l/by-path string.
    
    Args:
        camera_identifier: Either an int, or a string that could be:
            - A v4l/by-path path: "/dev/v4l/by-path/pci-..."
            - An int string: "2", "4", "22", etc.
            - A preset name: "left", "right"
    
    Returns:
        Either an int (for device ID) or str (for v4l/by-path path)
    """
    if isinstance(camera_identifier, int):
        return camera_identifier
    
    if isinstance(camera_identifier, str):
        # Check if it's a preset name
        if camera_identifier in TACTILE_CAM_PORTS:
            return TACTILE_CAM_PORTS[camera_identifier]
        
        # Check if it's a v4l/by-path path
        if camera_identifier.startswith("/dev/v4l/by-path/"):
            return camera_identifier
        
        # Check if it's a numeric string (convert to int)
        if camera_identifier.isdigit():
            return int(camera_identifier)
    
    raise ValueError(f"Invalid camera identifier: {camera_identifier}")

def listen_key(key):
    global trigger_state
    try:
        trigger_state[key.char] = True
    except:
        pass


def reset_key(key):
    global trigger_state
    try:
        trigger_state[key.char] = False
    except:
        pass


listener = keyboard.Listener(on_press=listen_key)
listener2 = keyboard.Listener(on_release=reset_key)
listener.start()
listener2.start()

###


def count_folders(path):
    """Counts the number of folders under the given path."""
    folder_count = 0
    for root, dirs, files in os.walk(path):
        folder_count += len(dirs)  # Count directories only at current level
        break  # Prevents descending into subdirectories
    return folder_count


def print_color(*args, color=None, attrs=(), **kwargs):
    if len(args) > 0:
        args = tuple(termcolor.colored(arg, color=color, attrs=attrs) for arg in args)
    print(*args, **kwargs)


def _policy_obs_with_raw_tactile(obs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Feed raw tactile frames to the policy while preserving env/display obs."""
    policy_obs = dict(obs)
    for sensor_name in ("tactile_left", "tactile_right"):
        raw_key = f"{sensor_name}_raw_rgb"
        rgb_key = f"{sensor_name}_rgb"
        if raw_key in obs:
            policy_obs[rgb_key] = obs[raw_key]
    return policy_obs


@dataclass
class MarkerTrackingState:
    name: str
    ref_gray: np.ndarray | None = None
    ref_points: np.ndarray | None = None
    display_window: str | None = None
    motion_ema: float = 0.0


@dataclass
class HapticsConfig:
    host: str
    port: int
    min_motion: float
    max_motion: float
    discrete_levels: bool
    soft_threshold: float
    mild_threshold: float
    firm_threshold: float
    soft_force: float
    mild_force: float
    firm_force: float
    smoothing: float
    release_smoothing: float
    active_only: bool


class HeadsetHapticsSender:
    def __init__(self, config: HapticsConfig):
        self.config = config
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.left_force = 0.0
        self.right_force = 0.0
        self._send_failed = False
        self._send_error = None

    def _send_forces(self, left_force: float, right_force: float) -> bool:
        if self._send_failed:
            return False
        try:
            send_packet(
                self.sock,
                self.config.host,
                self.config.port,
                left_force,
                right_force,
            )
            return True
        except OSError as exc:
            self._send_failed = True
            self._send_error = exc
            print(
                "[haptics] disabled after send failure: "
                f"host={self.config.host!r}, port={self.config.port}, error={exc}"
            )
            return False

    def _motion_to_force(self, motion: float) -> float:
        if motion <= self.config.min_motion:
            return 0.0
        motion_span = max(self.config.max_motion - self.config.min_motion, 1e-6)
        normalized = clamp01((motion - self.config.min_motion) / motion_span)
        if not self.config.discrete_levels:
            return normalized

        soft_threshold = clamp01(self.config.soft_threshold)
        mild_threshold = max(soft_threshold, clamp01(self.config.mild_threshold))
        firm_threshold = max(mild_threshold, clamp01(self.config.firm_threshold))
        if normalized <= soft_threshold:
            return 0.0
        if normalized <= mild_threshold:
            return clamp01(self.config.soft_force)
        if normalized <= firm_threshold:
            return clamp01(self.config.mild_force)
        return clamp01(self.config.firm_force)

    def _smooth_force(self, current_force: float, target_force: float) -> float:
        if target_force <= 0.0:
            return 0.0
        if target_force < current_force:
            alpha = clamp01(self.config.release_smoothing)
        else:
            alpha = clamp01(self.config.smoothing)
        return (1.0 - alpha) * current_force + alpha * target_force

    def _force_to_state(self, force: float) -> str:
        if force <= 1e-6:
            return "off"
        if self.config.discrete_levels:
            if force >= clamp01(self.config.firm_force) - 1e-6:
                return "firm"
            if force >= clamp01(self.config.mild_force) - 1e-6:
                return "mild"
            if force >= clamp01(self.config.soft_force) - 1e-6:
                return "soft"
        return "active"

    def current_state(self) -> str:
        return self._force_to_state(max(self.left_force, self.right_force))

    def update(self, left_motion: float | None, right_motion: float | None, enabled: bool = True):
        target_left = self._motion_to_force(left_motion or 0.0) if enabled else 0.0
        target_right = self._motion_to_force(right_motion or 0.0) if enabled else 0.0
        if self.config.discrete_levels:
            self.left_force = target_left
            self.right_force = target_right
        else:
            self.left_force = self._smooth_force(self.left_force, target_left)
            self.right_force = self._smooth_force(self.right_force, target_right)
        self._send_forces(self.left_force, self.right_force)

    def stop(self):
        try:
            self._send_forces(0.0, 0.0)
        finally:
            self.sock.close()


def _build_marker_tracking_params(args: "Args") -> dict:
    return {
        "morphop_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.marker_morph_open_size, args.marker_morph_open_size)
        ),
        "morphclose_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.marker_morph_close_size, args.marker_morph_close_size),
        ),
        "dilate_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (args.marker_dilate_size, args.marker_dilate_size)
        ),
        "mask_range": tuple(args.marker_mask_range),
        "min_value": args.marker_value_threshold,
        "morphop_iter": args.marker_morph_open_iter,
        "morphclose_iter": args.marker_morph_close_iter,
        "dilate_iter": args.marker_dilate_iter,
    }


def _init_marker_tracking_state(
    sensor_name: str,
    frame: np.ndarray | None,
    marker_tracking_params: dict,
    create_window: bool,
) -> MarkerTrackingState | None:
    if frame is None:
        return None

    marker_mask = find_marker(frame, **marker_tracking_params)
    centers = find_marker_centers(marker_mask)
    if not centers:
        print_color(
            f"[marker_tracking] {sensor_name}: no markers found in initial frame",
            color="yellow",
        )
        return MarkerTrackingState(name=sensor_name)

    state = MarkerTrackingState(
        name=sensor_name,
        ref_gray=cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
        ref_points=np.array(centers, dtype=np.float32).reshape(-1, 1, 2),
        display_window=f"{sensor_name}_marker_tracking",
    )
    if create_window:
        cv2.namedWindow(state.display_window, cv2.WINDOW_NORMAL)
    print(
        f"[marker_tracking] {sensor_name}: initialized with {len(centers)} markers"
    )
    return state


def _mask_contains_points(mask: np.ndarray, points: np.ndarray, radius: int = 3) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    if radius > 0:
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    h, w = mask.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < w)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < h)
    )
    on_mask = np.zeros(len(points), dtype=bool)
    if np.any(in_bounds):
        valid_points = rounded[in_bounds]
        on_mask[in_bounds] = mask[valid_points[:, 1], valid_points[:, 0]] > 0
    return on_mask


def _update_marker_tracking(
    sensor_name: str,
    frame: np.ndarray | None,
    state: MarkerTrackingState | None,
    marker_tracking_params: dict,
    lk_params: dict,
    reset_on_loss: bool,
    arrow_scale: float,
    fb_max_error: float,
    motion_deadband: float,
    motion_smoothing: float,
    motion_release_smoothing: float,
    min_valid_points: int,
    compensate_global_drift: bool,
) -> tuple[MarkerTrackingState | None, np.ndarray | None, float | None]:
    if frame is None:
        return state, None, None

    if state is None or state.ref_points is None or state.ref_gray is None:
        state = _init_marker_tracking_state(
            sensor_name=sensor_name,
            frame=frame,
            marker_tracking_params=marker_tracking_params,
            create_window=False,
        )
        if state is None or state.ref_points is None or state.ref_gray is None:
            return state, None, None

    track_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        state.ref_gray,
        track_gray,
        state.ref_points,
        None,
        **lk_params,
    )
    if next_points is None or status is None:
        if reset_on_loss:
            return _init_marker_tracking_state(
                state.name, frame, marker_tracking_params, create_window=False
            ), None, None
        return state, None, None

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        track_gray,
        state.ref_gray,
        next_points,
        None,
        **lk_params,
    )
    if back_points is None or back_status is None:
        if reset_on_loss:
            return _init_marker_tracking_state(
                state.name, frame, marker_tracking_params, create_window=False
            ), None, None
        return state, None, None

    current_marker_mask = find_marker(frame, **marker_tracking_params)

    ref_points_all = state.ref_points.reshape(-1, 2)
    tracked_points_all = next_points.reshape(-1, 2)
    back_points_all = back_points.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(back_points_all - ref_points_all, axis=1)
    on_marker_mask = _mask_contains_points(current_marker_mask, tracked_points_all)
    valid = status & back_status & (fb_error <= fb_max_error) & on_marker_mask

    if np.count_nonzero(valid) < min_valid_points:
        if reset_on_loss:
            return _init_marker_tracking_state(
                state.name, frame, marker_tracking_params, create_window=False
            ), None, None
        return state, None, 0.0

    ref_points = ref_points_all[valid]
    tracked_points = tracked_points_all[valid]
    deltas = tracked_points - ref_points

    if compensate_global_drift and len(deltas) > 0:
        deltas = deltas - np.median(deltas, axis=0, keepdims=True)

    delta_norms = np.linalg.norm(deltas, axis=1)
    delta_norms = np.clip(delta_norms - motion_deadband, 0.0, None)
    sum_motion = float(delta_norms.sum()) if len(delta_norms) > 0 else 0.0
    if sum_motion < state.motion_ema:
        alpha = float(np.clip(motion_release_smoothing, 0.0, 1.0))
    else:
        alpha = float(np.clip(motion_smoothing, 0.0, 1.0))
    state.motion_ema = (1.0 - alpha) * state.motion_ema + alpha * sum_motion
    overlay = plot_marker_delta(
        frame,
        tracked_points,
        deltas,
        scale=arrow_scale,
        arrow_color=(255, 0, 0),
    )
    return state, overlay, state.motion_ema


def _is_right_b_pressed(agent) -> bool:
    oculus_reader = getattr(agent, "oculus_reader", None)
    if oculus_reader is None:
        return False
    _, button_data = oculus_reader.get_transformations_and_buttons()
    return bool(button_data.get("B", False))


def _update_obs_marker_tracking(
    obs: Dict[str, np.ndarray],
    args: "Args",
    marker_tracking_states: dict[str, MarkerTrackingState | None],
    marker_tracking_params: dict,
    lk_params: dict,
    marker_motion: dict[str, float],
) -> None:
    if not (args.enable_marker_tracking and args.use_tactile):
        return

    for sensor_name in ("tactile_left", "tactile_right"):
        obs_key = f"{sensor_name}_rgb"
        tracking_key = f"{sensor_name}_marker_tracking"
        state, overlay, motion = _update_marker_tracking(
            sensor_name=sensor_name,
            frame=obs.get(obs_key),
            state=marker_tracking_states.get(sensor_name),
            marker_tracking_params=marker_tracking_params,
            lk_params=lk_params,
            reset_on_loss=args.marker_tracking_reset_on_loss,
            arrow_scale=args.marker_arrow_scale,
            fb_max_error=args.marker_flow_fb_max_error,
            motion_deadband=args.marker_motion_deadband,
            motion_smoothing=args.marker_motion_smoothing,
            motion_release_smoothing=args.marker_motion_release_smoothing,
            min_valid_points=args.marker_motion_min_valid_points,
            compensate_global_drift=args.marker_motion_compensate_global_drift,
        )
        marker_tracking_states[sensor_name] = state
        marker_motion[sensor_name] = motion or 0.0
        obs[f"{sensor_name}_marker_motion"] = np.array(
            marker_motion[sensor_name], dtype=np.float32
        )
        if overlay is not None:
            obs[tracking_key] = overlay
            if args.use_marker_tracking_overlay_for_policy:
                obs[obs_key] = overlay
            if (
                args.marker_tracking_show_view
                and state is not None
                and state.display_window is not None
            ):
                cv2.imshow(
                    state.display_window,
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR),
                )
                cv2.waitKey(1)


def _is_rocker_pressed(agent) -> bool:
    oculus_reader = getattr(agent, "oculus_reader", None)
    if oculus_reader is None:
        return bool(trigger_state.get("r", False))
    _, button_data = oculus_reader.get_transformations_and_buttons()
    return bool(button_data.get("RJ", False))


def _reset_agent_temporal_state(agent) -> None:
    reset_fn = getattr(agent, "reset_temporal_state", None)
    if callable(reset_fn):
        reset_fn()


def _prepare_obs_to_save(
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    activated=True,
    save_raw_images_only=False,
    save_base_size: tuple[int, int] | None = None,
    save_tactile_size: tuple[int, int] | None = None,
) -> Dict[str, np.ndarray]:
    obs_to_save = dict(obs)
    obs_to_save["activated"] = activated
    obs_to_save["control"] = action  # add action to obs

    if save_raw_images_only:
        filtered_obs = {}
        for key, value in obs_to_save.items():
            if key.endswith("_marker_tracking"):
                continue
            if key.endswith("_raw_rgb"):
                continue
            filtered_obs[key] = value

        # Prefer raw tactile frames when available, while keeping the saved field
        # names consistent for downstream loaders.
        for key in list(filtered_obs.keys()):
            if key.endswith("_rgb"):
                raw_key = key.replace("_rgb", "_raw_rgb")
                if raw_key in obs_to_save:
                    filtered_obs[key] = obs_to_save[raw_key]

        obs_to_save = filtered_obs

    if save_base_size is not None and "base_camera_rgb" in obs_to_save:
        save_width, save_height = save_base_size
        image = obs_to_save["base_camera_rgb"]
        if isinstance(image, np.ndarray):
            if image.ndim == 3:
                if image.shape[1] != save_width or image.shape[0] != save_height:
                    obs_to_save["base_camera_rgb"] = cv2.resize(
                        image,
                        (save_width, save_height),
                        interpolation=cv2.INTER_AREA,
                    )
            elif image.ndim == 4:
                if image.shape[2] != save_width or image.shape[1] != save_height:
                    obs_to_save["base_camera_rgb"] = np.stack(
                        [
                            cv2.resize(
                                frame,
                                (save_width, save_height),
                                interpolation=cv2.INTER_AREA,
                            )
                            for frame in image
                        ],
                        axis=0,
                    )

    if save_tactile_size is not None:
        save_width, save_height = save_tactile_size
        for key in ("tactile_left_rgb", "tactile_right_rgb"):
            if key not in obs_to_save:
                continue
            image = obs_to_save[key]
            if not isinstance(image, np.ndarray) or image.ndim != 3:
                continue
            if image.shape[1] == save_width and image.shape[0] == save_height:
                continue
            obs_to_save[key] = cv2.resize(
                image,
                (save_width, save_height),
                interpolation=cv2.INTER_AREA,
            )

    return obs_to_save


def _save_pngs(
    recorded_file: Path,
    obs_to_save: Dict[str, np.ndarray],
    save_png=False,
    save_tactile_png=False,
    use_tactile=True,
) -> None:
    # save rgb image as png
    if save_png:
        if "base_camera_rgb" in obs_to_save:
            rgb = obs_to_save["base_camera_rgb"]
            # Handle different dimensions from RealSense (can have multiple cameras)
            if rgb.ndim == 4:  # (num_cameras, H, W, 3)
                for i in range(rgb.shape[0]):
                    rgbi = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
                    fn = str(recorded_file)[:-4] + f"-base_{i}.png"
                    cv2.imwrite(fn, rgbi)
            elif rgb.ndim == 3:  # (H, W, 3) - single camera
                rgbi = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                fn = str(recorded_file)[:-4] + f"-base.png"
                cv2.imwrite(fn, rgbi)

    # save tactile images as png (only if use_tactile is True)
    if save_tactile_png and use_tactile:
        # Save left tactile sensor image
        if "tactile_left_rgb" in obs_to_save:
            tactile_left = obs_to_save["tactile_left_rgb"]
            if tactile_left.ndim == 4 and tactile_left.shape[0] == 1:
                tactile_left = tactile_left[0]
            if tactile_left.ndim == 3:
                tactile_left_bgr = cv2.cvtColor(tactile_left, cv2.COLOR_RGB2BGR)
                fn_left = str(recorded_file)[:-4] + f"-tactile_left.png"
                cv2.imwrite(fn_left, tactile_left_bgr)

        # Save right tactile sensor image
        if "tactile_right_rgb" in obs_to_save:
            tactile_right = obs_to_save["tactile_right_rgb"]
            if tactile_right.ndim == 4 and tactile_right.shape[0] == 1:
                tactile_right = tactile_right[0]
            if tactile_right.ndim == 3:
                tactile_right_bgr = cv2.cvtColor(tactile_right, cv2.COLOR_RGB2BGR)
                fn_right = str(recorded_file)[:-4] + f"-tactile_right.png"
                cv2.imwrite(fn_right, tactile_right_bgr)


def save_frame(
    folder: Path,
    timestamp: datetime.datetime,
    obs: Dict[str, np.ndarray],
    action: np.ndarray,
    activated=True,
    save_png=False,
    save_tactile_png=False,
    use_tactile=True,
    save_raw_images_only=False,
    save_base_size: tuple[int, int] | None = None,
    save_tactile_size: tuple[int, int] | None = None,
) -> None:
    obs_to_save = _prepare_obs_to_save(
        obs,
        action,
        activated=activated,
        save_raw_images_only=save_raw_images_only,
        save_base_size=save_base_size,
        save_tactile_size=save_tactile_size,
    )

    recorded_file = folder / (
        timestamp.isoformat().replace(":", "-").replace(".", "-") + ".pkl"
    )
    with open(recorded_file, "wb") as f:
        pickle.dump(obs_to_save, f)

    _save_pngs(
        recorded_file,
        obs_to_save,
        save_png=save_png,
        save_tactile_png=save_tactile_png,
        use_tactile=use_tactile,
    )


class H5TrajectoryWriter:
    DEPTH_DISPLAY_MIN_MM = 200
    DEPTH_DISPLAY_MAX_MM = 1500

    def __init__(
        self,
        file_path: Path,
        video_fps: float,
        compression: str = "gzip",
        compression_level: int = 4,
    ):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Saving to H5 requires the 'h5py' package in the active Python environment."
            ) from exc

        self._h5py = h5py
        self.file_path = file_path
        self.file = h5py.File(file_path, "w")
        self.frames_group = self.file.create_group("frames")
        self.datasets = {}
        self.frame_count = 0
        self.compression = compression
        self.compression_level = compression_level
        self.video_fps = float(video_fps)
        self.timestamps = self.file.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self.videos_group = self.file.create_group("videos")
        self.video_tempfiles: dict[str, tempfile.NamedTemporaryFile] = {}
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.video_source_kinds: dict[str, str] = {}
        self.file.attrs["video_fps"] = self.video_fps

    @staticmethod
    def _pack_depth_frame(array: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
        if array.dtype == np.uint16:
            depth_mm = array.astype(np.float32)
        elif array.dtype == np.uint8:
            depth_mm = array.astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth dtype for video export: {array.dtype}")

        valid = depth_mm > 0
        depth_clipped = np.clip(
            depth_mm,
            H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM,
            H5TrajectoryWriter.DEPTH_DISPLAY_MAX_MM,
        )
        depth_scaled = (
            (depth_clipped - H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM)
            * 255.0
            / (
                H5TrajectoryWriter.DEPTH_DISPLAY_MAX_MM
                - H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM
            )
        )
        depth_uint8 = np.zeros(array.shape, dtype=np.uint8)
        depth_uint8[valid] = depth_scaled[valid].astype(np.uint8)
        depth_rgb = np.repeat(depth_uint8[..., None], 3, axis=-1)
        return (
            depth_rgb,
            {
                "source_kind": "depth",
                "source_dtype": str(array.dtype),
                "encoding": "depth_uint8_gray",
                "depth_min_mm": H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM,
                "depth_max_mm": H5TrajectoryWriter.DEPTH_DISPLAY_MAX_MM,
            },
        )

    @staticmethod
    def _iter_video_streams(key: str, array: np.ndarray):
        if key.endswith("_rgb") and array.dtype == np.uint8:
            if array.ndim == 3:
                yield key, array, {"source_kind": "rgb", "source_dtype": "uint8"}
                return
            if array.ndim == 4 and array.shape[-1] == 3:
                for idx, frame in enumerate(array):
                    yield (
                        f"{key}_{idx}",
                        frame,
                        {"source_kind": "rgb", "source_dtype": "uint8", "source_index": idx},
                    )
            return

        if key.endswith("_depth"):
            if array.ndim == 2:
                packed, attrs = H5TrajectoryWriter._pack_depth_frame(array)
                yield key, packed, attrs
                return
            if array.ndim == 3:
                for idx, frame in enumerate(array):
                    packed, attrs = H5TrajectoryWriter._pack_depth_frame(frame)
                    attrs = dict(attrs)
                    attrs["source_index"] = idx
                    yield f"{key}_{idx}", packed, attrs

    def _append_video_frame(self, key: str, array: np.ndarray, extra_attrs: dict | None = None) -> None:
        writer = self.video_writers.get(key)
        if writer is None:
            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            temp_file.close()
            self.video_tempfiles[key] = temp_file
            height, width = array.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(temp_file.name, fourcc, self.video_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open MP4 writer for key '{key}'")
            self.video_writers[key] = writer
            video_ds = self.videos_group.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8,
            )
            video_ds.attrs["fps"] = self.video_fps
            video_ds.attrs["height"] = height
            video_ds.attrs["width"] = width
            video_ds.attrs["channels"] = array.shape[2]
            video_ds.attrs["codec"] = "mp4v"
            if extra_attrs is not None:
                for attr_key, attr_value in extra_attrs.items():
                    video_ds.attrs[attr_key] = attr_value
            self.video_source_kinds[key] = (
                str(extra_attrs.get("source_kind"))
                if extra_attrs is not None and "source_kind" in extra_attrs
                else "rgb"
            )

        if self.video_source_kinds.get(key) == "depth":
            writer.write(array)
        else:
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def _create_dataset(self, key: str, array: np.ndarray):
        if array.dtype.kind in {"U", "S", "O"} and array.ndim == 0:
            kwargs = {
                "shape": (0,),
                "maxshape": (None,),
                "dtype": self._h5py.string_dtype(encoding="utf-8"),
                "chunks": (256,),
            }
            self.datasets[key] = self.frames_group.create_dataset(key, **kwargs)
            return

        dataset_shape = (0,) + array.shape
        maxshape = (None,) + array.shape
        chunks = (1,) + array.shape if array.ndim > 0 else (256,)
        kwargs = {
            "shape": dataset_shape,
            "maxshape": maxshape,
            "dtype": array.dtype,
            "chunks": chunks,
        }
        if array.dtype.kind not in {"U", "S", "O"}:
            kwargs["compression"] = self.compression
            kwargs["compression_opts"] = self.compression_level
        self.datasets[key] = self.frames_group.create_dataset(key, **kwargs)

    def append(
        self,
        timestamp: datetime.datetime,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
        activated=True,
        save_raw_images_only=False,
        save_base_size: tuple[int, int] | None = None,
        save_tactile_size: tuple[int, int] | None = None,
    ) -> Dict[str, np.ndarray]:
        obs_to_save = _prepare_obs_to_save(
            obs,
            action,
            activated=activated,
            save_raw_images_only=save_raw_images_only,
            save_base_size=save_base_size,
            save_tactile_size=save_tactile_size,
        )

        for key, value in obs_to_save.items():
            array = np.asarray(value)
            video_streams = list(self._iter_video_streams(key, array))
            if video_streams:
                for video_key, video_array, video_attrs in video_streams:
                    self._append_video_frame(video_key, video_array, video_attrs)
                continue
            if key not in self.datasets:
                self._create_dataset(key, array)
            dataset = self.datasets[key]
            if dataset.dtype.metadata is not None and dataset.dtype.metadata.get("vlen") is str:
                if array.ndim != 0:
                    raise ValueError(
                        f"Inconsistent shape/dtype for key '{key}': "
                        f"expected scalar string; got shape {array.shape}, dtype {array.dtype}"
                    )
                dataset.resize(self.frame_count + 1, axis=0)
                item = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                dataset[self.frame_count] = item
                continue
            if dataset.shape[1:] != array.shape or dataset.dtype != array.dtype:
                raise ValueError(
                    f"Inconsistent shape/dtype for key '{key}': "
                    f"expected shape {dataset.shape[1:]}, dtype {dataset.dtype}; "
                    f"got shape {array.shape}, dtype {array.dtype}"
                )
            dataset.resize(self.frame_count + 1, axis=0)
            dataset[self.frame_count] = array

        self.timestamps.resize(self.frame_count + 1, axis=0)
        self.timestamps[self.frame_count] = timestamp.isoformat()
        self.frame_count += 1
        self.file.attrs["frame_count"] = self.frame_count
        return obs_to_save

    def close(self):
        if getattr(self, "file", None) is not None:
            for key, writer in self.video_writers.items():
                writer.release()
                temp_file = self.video_tempfiles[key]
                with open(temp_file.name, "rb") as f:
                    video_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                dataset = self.videos_group[key]
                dataset.resize((video_bytes.shape[0],))
                dataset[...] = video_bytes
                os.unlink(temp_file.name)
            self.file.flush()
            self.file.close()
            self.file = None


@dataclass
class Args:
    robot_port: int = 6000
    wrist_camera_port: int = 5001
    base_camera_port: int = 5000
    tactile_left_camera_id: str = "left"  # v4l/by-path or int ID (supports: "left", "2", "/dev/v4l/by-path/...")
    tactile_right_camera_id: str = "right"  # v4l/by-path or int ID (supports: "right", "4", "/dev/v4l/by-path/...")
    hostname: str = "127.0.0.1"
    hz: int = 15
    show_camera_view: bool = True
    agent: str = "quest"
    robot_type: str = "ur5"
    save_data: bool = False
    save_depth: bool = True
    save_png: bool = False
    use_tactile: bool = True  # whether to use tactile sensors
    save_tactile_png: bool = False  # save tactile images as PNG
    save_raw_images_only: bool = True  # keep robot state/action, but only raw RGB images
    save_format: str = "h5"  # "h5" for one file per trajectory, "pkl" for one file per frame
    realsense_width: int = 640  # RealSense camera resolution width
    realsense_height: int = 480  # RealSense camera resolution height
    realsense_fps: int = 30  # RealSense camera FPS
    save_base_width: int = 320  # Saved base image width
    save_base_height: int = 240  # Saved base image height
    tactile_width: int = 640  # Tactile camera resolution width
    tactile_height: int = 480  # Tactile camera resolution height
    save_tactile_width: int = 320  # Saved tactile image width
    save_tactile_height: int = 240  # Saved tactile image height
    tactile_crop_config_dir: str = ""
    tactile_crop_input_width: int = 320
    tactile_crop_input_height: int = 240
    enable_marker_tracking: bool = True
    use_marker_tracking_overlay_for_policy: bool = False
    marker_tracking_show_view: bool = True
    marker_tracking_reset_on_loss: bool = True
    marker_flow_win_size: tuple[int, int] = (15, 15)
    marker_flow_max_level: int = 2
    marker_flow_fb_max_error: float = 1.5
    marker_arrow_scale: float = 6.0
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
    headset_haptics_host: str = "192.168.1.126"
    headset_haptics_port: int = 9000
    haptics_min_motion: float = 0.0
    haptics_max_motion: float = 20.0
    haptics_discrete_levels: bool = True
    haptics_soft_threshold: float = 0.1 #0.02 0.25
    haptics_mild_threshold: float = 0.3 #0.05  0.5
    haptics_firm_threshold: float = 0.4#0.09 0.75
    haptics_soft_force: float = 0.1
    haptics_mild_force: float = 0.4
    haptics_firm_force: float = 1.0
    haptics_smoothing: float = 0.35
    haptics_release_smoothing: float = 1.0
    haptics_active_only: bool = True
    data_dir: str = "./shared/data/bc_data"
    verbose: bool = False
    safe: bool = False
    use_vel_ik: bool = False

    use_camera_node: bool = True # use camera node

    num_diffusion_iters_compile: int = 15  # used for compilation only for now
    jit_compile: bool = False  # send the compilation signal to the server (only need to do this once per inference server run).
    use_jit_agent: bool = False  # use the inference server to get actions. The inference_agent_port and the inference_agent_host need to be set to the proper values.
    inference_agent_port: str = (
        "1234"  # port must be the same as the inference server port
    )
    inference_agent_host = "127.0.0.2"  # ip of the inference server (localhost if running locally; currently defaults to bt) inference server needs to use the same checkpoint folder when launching the inference node (args need to match)

    dp_ckpt_path: str = "./shared/ckpts/best.ckpt"
    act_ckpt_path: str = ""
    pi0_policy_host: str = ""
    pi0_policy_port: int = 8000
    pi0_prompt: str = "pick up the paper cup and place it on the target"
    pi0_state_dim: int = 7
    pi0_base_camera_index: int = 1
    pi0_wrist_camera_index: int = 0
    pi0_action_chunk_size: int = 6
    pi0_joint_ema_alpha: float = 0.35
    pi0_gripper_ema_alpha: float = 0.35
    pi0_gripper_close_threshold: float = 0.18
    pi0_gripper_open_threshold: float = 0.08
    pi0_gripper_min_hold_steps: int = 30
    pi0_debug_actions: bool = False
    pi0_async_prefetch: bool = True
    pi0_prefetch_threshold: int = 2
    pi0_include_tactile: bool = False
    pi0_tactile_feature_mode: str = "none"
    pi0_tactile_embedding_dim: int = 128
    pi0_eef_translation_scale: float = 1.0
    pi0_eef_rotation_scale: float = 1.0
    safe_max_joint_delta: float = 0.03
    safe_max_hand_delta: float = 0.03

    temporal_ensemble_mode: str = "avg"
    temporal_ensemble_act_tau: float = 0.5


def main(args):
    marker_tracking_params = _build_marker_tracking_params(args)
    if args.agent in ("pi0", "pi0_eef") and args.use_marker_tracking_overlay_for_policy:
        if not args.pi0_include_tactile or args.pi0_tactile_feature_mode != "image_embedding":
            print_color(
                "[pi0] warning: --use-marker-tracking-overlay-for-policy is enabled, "
                "but pi0 tactile embedding is not enabled. Add "
                "--pi0-include-tactile --pi0-tactile-feature-mode image_embedding "
                "for a tactile pi0 checkpoint.",
                color="yellow",
                attrs=("bold",),
            )
    tactile_crop_configs = _load_tactile_crop_configs(args.tactile_crop_config_dir)
    tactile_crop_input_size = (
        int(args.tactile_crop_input_width),
        int(args.tactile_crop_input_height),
    )
    for sensor_name, config in tactile_crop_configs.items():
        print(
            f"Tactile camera crop config - {sensor_name}: {config['path']} "
            f"output={config['size'][0]}x{config['size'][1]}"
        )
    lk_params = {
        "winSize": tuple(args.marker_flow_win_size),
        "maxLevel": args.marker_flow_max_level,
        "criteria": (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        ),
    }

    # Initialize cameras
    if args.use_camera_node:
        print("Using camera node...")
        camera_clients = {
            "base_camera": ZMQClientCamera(port=args.base_camera_port, host=args.hostname),
        }
    else:
        print("Initializing RealSense base camera...")
        camera_clients = {
            "base_camera": RealSenseCamera(
                height=args.realsense_height,
                width=args.realsense_width,
                fps=args.realsense_fps,
                img_size=(args.realsense_width, args.realsense_height),
            ),
        }
    
    # Add tactile cameras if enabled (using OpenCV webcams)
    if args.use_tactile:
        print("Initializing tactile sensors with OpenCV...")
        # Resolve camera IDs/paths
        left_cam_id = _resolve_camera_id(args.tactile_left_camera_id)
        right_cam_id = _resolve_camera_id(args.tactile_right_camera_id)
        left_warp_config = (
            tactile_crop_configs["tactile_left"]["path"]
            if tactile_crop_configs
            else _resolve_tactile_warp_config("", "tactile_left")
        )
        right_warp_config = (
            tactile_crop_configs["tactile_right"]["path"]
            if tactile_crop_configs
            else _resolve_tactile_warp_config("", "tactile_right")
        )
        
        camera_clients["tactile_left"] = OpenCVCamera(
            camera_id=left_cam_id,
            width=args.tactile_width,
            height=args.tactile_height,
            perspective_config_path=left_warp_config,
            perspective_key="tactile_left",
        )
        camera_clients["tactile_right"] = OpenCVCamera(
            camera_id=right_cam_id,
            width=args.tactile_width,
            height=args.tactile_height,
            perspective_config_path=right_warp_config,
            perspective_key="tactile_right",
        )
        print(f"Tactile sensors enabled - Left: {left_cam_id}, Right: {right_cam_id}")
        print(
            "Tactile warp config - "
            f"Left: {left_warp_config or 'disabled'}, "
            f"Right: {right_warp_config or 'disabled'}"
        )
    else:
        print("Tactile sensors disabled")
    
    robot_client = ZMQClientRobot(port=args.robot_port, host=args.hostname)
    env = RobotEnv(
        robot_client,
        control_rate_hz=args.hz,
        camera_dict=camera_clients,
        show_camera_view=args.show_camera_view,
        save_depth=args.save_depth,
    )

    if args.agent == "quest":
        # Use the Quest agent variant with A-button gripper control.
        from agents.quest_agent_Akey import SingleArmQuestAgent

        agent = SingleArmQuestAgent(robot_type=args.robot_type, which_hand="r")
        print("Quest agent created")
    elif args.agent in ["dp", "dp_eef"]:
        if args.use_jit_agent:
            from agents.dp_agent_zmq import BimanualDPAgent

            agent = BimanualDPAgent(
                ckpt_path=args.dp_ckpt_path,
                port=args.inference_agent_port,
                host=args.inference_agent_host,
                temporal_ensemble_act_tau=args.temporal_ensemble_act_tau,
                temporal_ensemble_mode=args.temporal_ensemble_mode,
            )
        else:
            from agents.dp_agent import BimanualDPAgent

            agent = BimanualDPAgent(ckpt_path=args.dp_ckpt_path)
    elif args.agent in ["act", "act_eef"]:
        if args.use_jit_agent:
            raise ValueError("ACT deployment currently supports local inference only; set --no-use-jit-agent.")
        from agents.act_agent import BimanualACTAgent

        ckpt_path = args.act_ckpt_path or args.dp_ckpt_path
        agent = BimanualACTAgent(ckpt_path=ckpt_path)
    elif args.agent in ["pi0", "pi0_eef"]:
        from agents.pi0_agent import Pi0Agent

        pi0_host = args.pi0_policy_host or args.inference_agent_host
        pi0_port = args.pi0_policy_port or int(args.inference_agent_port)
        agent = Pi0Agent(
            host=pi0_host,
            port=pi0_port,
            predict_eef_delta=args.agent.endswith("_eef"),
            state_dim=args.pi0_state_dim,
            base_camera_index=args.pi0_base_camera_index,
            wrist_camera_index=args.pi0_wrist_camera_index,
            prompt=args.pi0_prompt,
            action_chunk_size=args.pi0_action_chunk_size,
            joint_ema_alpha=args.pi0_joint_ema_alpha,
            gripper_ema_alpha=args.pi0_gripper_ema_alpha,
            gripper_close_threshold=args.pi0_gripper_close_threshold,
            gripper_open_threshold=args.pi0_gripper_open_threshold,
            gripper_min_hold_steps=args.pi0_gripper_min_hold_steps,
            debug_actions=args.pi0_debug_actions,
            async_prefetch=args.pi0_async_prefetch,
            prefetch_threshold=args.pi0_prefetch_threshold,
            include_tactile=args.pi0_include_tactile,
            tactile_feature_mode=args.pi0_tactile_feature_mode,
            tactile_embedding_dim=args.pi0_tactile_embedding_dim,
            eef_translation_scale=args.pi0_eef_translation_scale,
            eef_rotation_scale=args.pi0_eef_rotation_scale,
        )
        print(f"pi0 agent created: ws://{pi0_host}:{pi0_port}")
    else:
        raise ValueError(f"Invalid agent name : {args.agent}")

    if args.agent == "quest":
        # using grippers
        #To-do   90
        # reset_joints = np.deg2rad([-82, -102, -70, -98, 86, 90, 0])
        # reset_joints = np.deg2rad([-87, -88, -112, -67, 90, 0, 0])
        reset_joints = np.deg2rad([0, -88, -112, -67, 90, 0, 0]) 
    else:
        from agents.dp_agent import get_reset_joints

        reset_joints = get_reset_joints(ur_eef=False)
    curr_joints = env.get_obs()["joint_positions"]
    print("Current joints:", curr_joints)
    print("Reset joints:", reset_joints)
    max_delta = (np.abs(curr_joints - reset_joints)).max()
    steps = min(int(max_delta / 0.01), 20)
    for jnt in np.linspace(curr_joints, reset_joints, steps):
        env.step(jnt)

    obs = env.get_obs()
    _apply_tactile_crop_configs(obs, tactile_crop_configs, tactile_crop_input_size)
    marker_tracking_states: dict[str, MarkerTrackingState | None] = {}
    marker_motion = {"tactile_left": 0.0, "tactile_right": 0.0}
    if args.enable_marker_tracking and args.use_tactile:
        for sensor_name in ("tactile_left", "tactile_right"):
            obs_key = f"{sensor_name}_rgb"
            marker_tracking_states[sensor_name] = _init_marker_tracking_state(
                sensor_name=sensor_name,
                frame=obs.get(obs_key),
                marker_tracking_params=marker_tracking_params,
                create_window=args.marker_tracking_show_view,
            )
    else:
        marker_tracking_states = {}

    haptics_sender = None
    prev_haptics_state = None
    if args.headset_haptics_host:
        haptics_sender = HeadsetHapticsSender(
            HapticsConfig(
                host=args.headset_haptics_host,
                port=args.headset_haptics_port,
                min_motion=args.haptics_min_motion,
                max_motion=args.haptics_max_motion,
                discrete_levels=args.haptics_discrete_levels,
                soft_threshold=args.haptics_soft_threshold,
                mild_threshold=args.haptics_mild_threshold,
                firm_threshold=args.haptics_firm_threshold,
                soft_force=args.haptics_soft_force,
                mild_force=args.haptics_mild_force,
                firm_force=args.haptics_firm_force,
                smoothing=args.haptics_smoothing,
                release_smoothing=args.haptics_release_smoothing,
                active_only=args.haptics_active_only,
            )
        )
        print(
            "Headset haptics enabled - "
            f"host: {args.headset_haptics_host}, port: {args.headset_haptics_port}"
        )
    else:
        print("Headset haptics disabled")

    _update_obs_marker_tracking(
        obs,
        args,
        marker_tracking_states,
        marker_tracking_params,
        lk_params,
        marker_motion,
    )
    policy_obs = (
        obs if args.use_marker_tracking_overlay_for_policy else _policy_obs_with_raw_tactile(obs)
    )
    if args.jit_compile and args.agent.startswith(("dp", "act")):
        agent.compile_inference(
            policy_obs, num_diffusion_iters=args.num_diffusion_iters_compile
        )
        _reset_agent_temporal_state(agent)
    # going to start position
    print("Going to start position")
    start_pos = agent.act(policy_obs)  # in mujoco
    _reset_agent_temporal_state(agent)
    obs = env.get_obs()
    _apply_tactile_crop_configs(obs, tactile_crop_configs, tactile_crop_input_size)
    joints = obs["joint_positions"]

    ur_idx = [i for i in range(min(6, len(joints)))]
    hand_idx = [len(joints) - 1] if len(joints) == 7 else None

    if args.safe:
        safety_wrapper = SafetyWrapper(
            ur_idx,
            hand_idx,
            agent,
            delta=args.safe_max_joint_delta,
            hand_delta=args.safe_max_hand_delta,
        )

    print(f"Start pos: {len(start_pos)}", f"Joints: {len(joints)}")
    assert len(start_pos) == len(
        joints
    ), f"agent output dim = {len(start_pos)}, but env dim = {len(joints)}"

    traj_idx = count_folders(args.data_dir)

    try:
        while True:  # outer loop: one iteration per trajectory
            traj_idx += 1
            print(f"\nTrajectory #{traj_idx}")
            has_oculus = getattr(agent, "oculus_reader", None) is not None
            start_hint = (
                ">>> Press rocker [RJ] ONCE to move to initial position and start recording"
                if has_oculus
                else ">>> Press and release [r] ONCE to move to initial position and start recording"
            )
            print_color(start_hint, color="cyan", attrs=("bold",))

            # Ensure rocker is released before listening for the next click
            # (prevents the 2nd click of a double-click from spuriously triggering a new trajectory)
            while _is_rocker_pressed(agent):
                time.sleep(0.05)
            time.sleep(0.1)  # debounce

            # Wait for single rocker click (rising edge)
            prev_rocker_wait = False
            while True:
                rocker_wait = _is_rocker_pressed(agent)
                if rocker_wait and not prev_rocker_wait:
                    break
                prev_rocker_wait = rocker_wait
                time.sleep(0.05)

            # Move to initial position
            print_color("\nMoving to initial position...", color="cyan")
            curr_joints = env.get_obs()["joint_positions"]
            max_delta = (np.abs(curr_joints - reset_joints)).max()
            steps = min(int(max_delta / 0.01), 20)
            for jnt in np.linspace(curr_joints, reset_joints, steps):
                env.step(jnt)

            obs = env.get_obs()
            _reset_agent_temporal_state(agent)
            stop_hint = (
                "Recording started! Press rocker [RJ] TWICE to stop."
                if has_oculus
                else "Recording started! Press [s] to stop+save, or [d] to stop+delete."
            )
            print_color(stop_hint, color="green", attrs=("bold",))

            start_time = time.time()

            # Setup data saving for this trajectory
            trajectory_writer = None
            save_path = None
            if args.save_data:
                time_str = datetime.datetime.now().strftime("%m%d_%H%M%S")
                save_path = Path(args.data_dir).expanduser() / time_str
                save_path.mkdir(parents=True, exist_ok=True)
                print(f"Saving to {save_path}")
                if args.save_format == "h5":
                    trajectory_writer = H5TrajectoryWriter(
                        save_path / "trajectory.h5",
                        video_fps=args.hz,
                    )
                    print(f"Trajectory file: {save_path / 'trajectory.h5'}")
                elif args.save_format != "pkl":
                    raise ValueError(
                        f"Invalid save_format: {args.save_format}. Expected 'h5' or 'pkl'."
                    )

            is_first_frame = True
            frame_freq = []
            prev_b_pressed = False
            stop_type = None  # "double" → save, "triple" → delete

            # Click detection state
            prev_rocker = False
            rocker_click_count = 0
            last_rocker_click_time = 0.0
            DOUBLE_CLICK_WINDOW = 0.5

            try:
                while True:
                    new_start_time = time.time()
                    num = new_start_time - start_time
                    message = f"\rTime passed: {round(num, 2)}          "
                    print_color(
                        message,
                        color="white",
                        attrs=("bold",),
                        end="",
                        flush=True,
                    )
                    b_pressed = _is_right_b_pressed(agent)
                    if (
                        args.enable_marker_tracking
                        and args.use_tactile
                        and b_pressed
                        and not prev_b_pressed
                    ):
                        for sensor_name in ("tactile_left", "tactile_right"):
                            obs_key = f"{sensor_name}_rgb"
                            marker_tracking_states[sensor_name] = _init_marker_tracking_state(
                                sensor_name=sensor_name,
                                frame=obs.get(obs_key),
                                marker_tracking_params=marker_tracking_params,
                                create_window=False,
                            )
                            marker_motion[sensor_name] = 0.0
                        print_color(
                            "\n[marker_tracking] reset tactile reference frame from right controller B",
                            color="cyan",
                            attrs=("bold",),
                        )
                    prev_b_pressed = b_pressed

                    _update_obs_marker_tracking(
                        obs,
                        args,
                        marker_tracking_states,
                        marker_tracking_params,
                        lk_params,
                        marker_motion,
                    )

                    policy_obs = (
                        obs
                        if args.use_marker_tracking_overlay_for_policy
                        else _policy_obs_with_raw_tactile(obs)
                    )
                    if args.safe:
                        action = safety_wrapper.act_safe(
                            agent,
                            policy_obs,
                            eef=(args.agent.endswith("_eef")),
                        )
                    else:
                        action = agent.act(policy_obs)
                    dt = datetime.datetime.now()

                    if haptics_sender is not None:
                        sum_motion = max(
                            marker_motion["tactile_left"],
                            marker_motion["tactile_right"],
                        )
                        motion_span = max(
                            haptics_sender.config.max_motion - haptics_sender.config.min_motion,
                            1e-6,
                        )
                        normalized_sum_motion = clamp01(
                            (sum_motion - haptics_sender.config.min_motion) / motion_span
                        )
                        haptics_sender.update(
                            left_motion=0.0,
                            right_motion=sum_motion,
                            enabled=(not haptics_sender.config.active_only)
                            or getattr(agent, "control_active", True),
                        )
                        current_haptics_state = haptics_sender.current_state()
                        obs["haptics_state"] = np.bytes_(current_haptics_state)
                        obs["haptics_sum_marker_motion"] = np.array(sum_motion, dtype=np.float32)
                        obs["haptics_normalized_sum_marker_motion"] = np.array(
                            normalized_sum_motion, dtype=np.float32
                        )
                        if current_haptics_state != prev_haptics_state:
                            print_color(
                                "\n"
                                f"[haptics] state: {current_haptics_state}, "
                                f"sum_marker_motion: {sum_motion:.4f}, "
                                f"normalized_sum: {normalized_sum_motion:.4f}",
                                color="magenta",
                                attrs=("bold",),
                            )
                            prev_haptics_state = current_haptics_state
                    else:
                        obs["haptics_state"] = np.bytes_("disabled")
                        obs["haptics_sum_marker_motion"] = np.array(0.0, dtype=np.float32)
                        obs["haptics_normalized_sum_marker_motion"] = np.array(0.0, dtype=np.float32)

                    if args.save_data:
                        if is_first_frame:
                            is_first_frame = False
                        else:
                            if args.save_format == "h5":
                                obs_to_save = trajectory_writer.append(
                                    dt,
                                    obs,
                                    action,
                                    save_raw_images_only=args.save_raw_images_only,
                                    save_base_size=(
                                        args.save_base_width,
                                        args.save_base_height,
                                    ),
                                    save_tactile_size=(
                                        args.save_tactile_width,
                                        args.save_tactile_height,
                                    ),
                                )
                                _save_pngs(
                                    save_path / f"{dt.isoformat().replace(':', '-').replace('.', '-')}.h5",
                                    obs_to_save,
                                    save_png=args.save_png,
                                    save_tactile_png=args.save_tactile_png,
                                    use_tactile=args.use_tactile,
                                )
                            else:
                                save_frame(
                                    save_path,
                                    dt,
                                    obs,
                                    action,
                                    save_png=args.save_png,
                                    save_tactile_png=args.save_tactile_png,
                                    use_tactile=args.use_tactile,
                                    save_raw_images_only=args.save_raw_images_only,
                                    save_base_size=(
                                        args.save_base_width,
                                        args.save_base_height,
                                    ),
                                    save_tactile_size=(
                                        args.save_tactile_width,
                                        args.save_tactile_height,
                                    ),
                                )

                    if args.agent.endswith("_eef"):
                        obs = env.step_eef(action)
                    else:
                        obs = env.step(action)
                    _apply_tactile_crop_configs(
                        obs, tactile_crop_configs, tactile_crop_input_size
                    )

                    ff = 1 / (time.time() - new_start_time)
                    frame_freq.append(ff)

                    # Stop detection
                    if not has_oculus:
                        # Keyboard mode: s=stop+save, d=stop+delete
                        if trigger_state.get("s", False):
                            print_color(
                                "\n[s] pressed, stopping and saving trajectory.",
                                color="yellow",
                                attrs=("bold",),
                            )
                            stop_type = "double"
                            break
                        elif trigger_state.get("d", False):
                            print_color(
                                "\n[d] pressed, stopping and deleting trajectory.",
                                color="red",
                                attrs=("bold",),
                            )
                            stop_type = "triple"
                            break
                    else:
                        # Quest mode: double-click to stop+save, triple-click to stop+delete
                        rocker_pressed = _is_rocker_pressed(agent)
                        if rocker_pressed and not prev_rocker:
                            now = time.time()
                            if now - last_rocker_click_time < DOUBLE_CLICK_WINDOW:
                                rocker_click_count += 1
                            else:
                                rocker_click_count = 1
                            last_rocker_click_time = now
                        prev_rocker = rocker_pressed

                        if rocker_click_count >= 3:
                            print_color(
                                "\nTriple-click detected, stopping and deleting trajectory.",
                                color="red",
                                attrs=("bold",),
                            )
                            stop_type = "triple"
                            break
                        elif rocker_click_count == 2 and (time.time() - last_rocker_click_time) > DOUBLE_CLICK_WINDOW:
                            print_color(
                                "\nDouble-click detected, stopping recording.",
                                color="yellow",
                                attrs=("bold",),
                            )
                            stop_type = "double"
                            break

            finally:
                if trajectory_writer is not None:
                    trajectory_writer.close()

                if stop_type == "triple" and save_path is not None and save_path.exists():
                    shutil.rmtree(save_path)
                    traj_idx -= 1
                    print_color(
                        f"Trajectory deleted: {save_path}",
                        color="red",
                        attrs=("bold",),
                    )
                else:
                    if trajectory_writer is not None:
                        print("Trajectory file saved.")
                    if args.save_data and save_path is not None and len(frame_freq) > 1:
                        with open(save_path / "freq.txt", "w") as f:
                            f.write(
                                f"Average FPS: {np.mean(frame_freq[1:])}\n"
                                f"Max FPS: {np.max(frame_freq[1:])}\n"
                                f"Min FPS: {np.min(frame_freq[1:])}\n"
                                f"Std FPS: {np.std(frame_freq[1:])}\n\n"
                            )
                            for step, freq in enumerate(frame_freq):
                                f.write(f"{step}: {freq}\n")
                        print(f"Saved {len(frame_freq)} frames to {save_path}")

    except KeyboardInterrupt:
        print_color("\nInterrupted!", color="red", attrs=("bold",))
    except Exception:
        print_color(
            "\nrun_env crashed with an exception; traceback follows.",
            color="red",
            attrs=("bold",),
        )
        traceback.print_exc()
    finally:
        print("Done")

        if haptics_sender is not None:
            haptics_sender.stop()

        # Release camera resources
        print("Releasing camera resources...")
        for camera_name, camera in camera_clients.items():
            if hasattr(camera, 'release'):
                camera.release()
                print(f"Released {camera_name}")

        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
