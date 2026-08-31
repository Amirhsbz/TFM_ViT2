from __future__ import annotations

import collections
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np

from agents.dp_agent import get_eef_pose, get_reset_joints
from learning.pi0_ur5e.pi0_ur5e.policy_client import (
    WebsocketPolicyClient,
    build_policy_observation,
)
from learning.pi0_ur5e.pi0_ur5e.tactile_features import TactileImageEncoder
from learning.pi0_ur5e.pi0_ur5e.transforms import clip_pi0_action
from learning.pi0_ur5e.pi0_ur5e.schema import Pi0Ur5eConfig


class Pi0Agent:
    """Single-arm pi0 deployment agent for run_env.py.

    Set predict_eef_delta=False for DP-style joint-space checkpoints that output
    [joint_0, ..., joint_5, gripper].
    """

    def __init__(
        self,
        host: str,
        port: int = 8000,
        *,
        predict_eef_delta: bool = False,
        state_dim: int = 7,
        image_size: tuple[int, int] = (224, 224),
        include_tactile: bool = False,
        tactile_feature_mode: str = "none",
        tactile_embedding_dim: int = 128,
        base_camera_index: int = 1,
        wrist_camera_index: int = 0,
        prompt: str | None = None,
        gripper_min: float = 0.0,
        gripper_max: float = 1.0,
        reject_translation_delta: float = 0.25,
        action_chunk_size: int = 6,
        joint_ema_alpha: float = 0.35,
        gripper_ema_alpha: float = 0.35,
        gripper_close_threshold: float = 0.18,
        gripper_open_threshold: float = 0.08,
        gripper_min_hold_steps: int = 30,
        debug_actions: bool = False,
        async_prefetch: bool = True,
        prefetch_threshold: int = 2,
        eef_translation_scale: float = 1.0,
        eef_rotation_scale: float = 1.0,
    ):
        self.client = WebsocketPolicyClient(host, port)
        self.config = Pi0Ur5eConfig()
        self.predict_eef_delta = predict_eef_delta
        self.include_tactile = bool(include_tactile)
        self.tactile_feature_mode = tactile_feature_mode
        self.tactile_embedding_dim = int(tactile_embedding_dim)
        if self.include_tactile and self.tactile_feature_mode == "image_embedding" and state_dim <= 7:
            state_dim = 7 + self.tactile_embedding_dim
        self.state_dim = state_dim
        self.image_size = image_size
        self.tactile_encoder = (
            TactileImageEncoder(embedding_dim=self.tactile_embedding_dim)
            if self.include_tactile and self.tactile_feature_mode == "image_embedding"
            else None
        )
        self.base_camera_index = base_camera_index
        self.wrist_camera_index = wrist_camera_index
        self.prompt = prompt or self.config.default_prompt
        self.config.action_limits["min_gripper"] = gripper_min
        self.config.action_limits["max_gripper"] = gripper_max
        self.reject_translation_delta = reject_translation_delta
        self.action_chunk_size = max(int(action_chunk_size), 1)
        self.joint_ema_alpha = float(np.clip(joint_ema_alpha, 0.0, 1.0))
        self.gripper_ema_alpha = float(np.clip(gripper_ema_alpha, 0.0, 1.0))
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.gripper_min_hold_steps = max(int(gripper_min_hold_steps), 0)
        self.debug_actions = debug_actions
        self.async_prefetch = bool(async_prefetch)
        self.prefetch_threshold = max(int(prefetch_threshold), 0)
        self.eef_translation_scale = float(eef_translation_scale)
        self.eef_rotation_scale = float(eef_rotation_scale)
        self.action_queue: collections.deque[np.ndarray] = collections.deque()
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: np.ndarray | None = None
        self._prefetch_error: BaseException | None = None
        self._prefetch_lock = threading.Lock()
        self.last_joint_action: np.ndarray | None = None
        self.last_gripper_action: float | None = None
        self.gripper_closed = False
        self.gripper_hold_steps = 0
        self.control = get_reset_joints(ur_eef=predict_eef_delta)
        self.trigger_state = True
        self.control_active = True

        metadata_action_format = self.client.metadata.get("action_format")
        expected_action_format = "ee_delta_6d_gripper" if self.predict_eef_delta else "joint_position_gripper"
        if metadata_action_format and metadata_action_format != expected_action_format:
            print(
                "[pi0] warning: server metadata action_format="
                f"{metadata_action_format!r}; this agent expects {expected_action_format!r}."
            )

    def reset_temporal_state(self) -> None:
        self._finish_prefetch_blocking()
        self.control = get_reset_joints(ur_eef=self.predict_eef_delta)
        self.action_queue.clear()
        self.last_joint_action = None
        self.last_gripper_action = None
        self.gripper_closed = False
        self.gripper_hold_steps = 0
        reset = getattr(self.client, "reset", None)
        if callable(reset):
            reset()

    def _select_image(self, obs: Dict[str, Any], canonical: str) -> np.ndarray:
        direct_keys = {
            "base": ("base_rgb", "third_view_camera_rgb", "base_camera_rgb_1"),
            "wrist": ("wrist_rgb", "wrist_camera_rgb", "camera_wrist_rgb", "base_camera_rgb_0"),
        }[canonical]
        for key in direct_keys:
            if key in obs and obs[key] is not None:
                return self._resize_image(obs[key])

        if "base_camera_rgb" not in obs or obs["base_camera_rgb"] is None:
            raise KeyError(f"Missing image for {canonical}; expected one of {direct_keys} or base_camera_rgb")

        images = np.asarray(obs["base_camera_rgb"])
        if images.ndim == 3:
            return self._resize_image(images)
        if images.ndim != 4:
            raise ValueError(f"Expected base_camera_rgb shape [N,H,W,C] or [H,W,C], got {images.shape}")

        index = self.base_camera_index if canonical == "base" else self.wrist_camera_index
        if index >= images.shape[0]:
            index = min(images.shape[0] - 1, 0)
        return self._resize_image(images[index])

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
            image = np.moveaxis(image, 0, -1)
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(np.uint8)
        if image.shape[:2] != self.image_size:
            image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_AREA)
        return image

    def _base_state(self, obs: Dict[str, Any]) -> np.ndarray:
        if self.predict_eef_delta:
            if "ee_pos_quat" in obs and obs["ee_pos_quat"] is not None:
                state = np.asarray(obs["ee_pos_quat"], dtype=np.float32).reshape(-1)[:6]
            elif "joint_positions" in obs and obs["joint_positions"] is not None:
                state = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1)[:6]
            else:
                raise KeyError("pi0_eef requires obs['ee_pos_quat'] or obs['joint_positions']")
        elif "joint_positions" in obs and obs["joint_positions"] is not None:
            state = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1)[:6]
        elif "ee_pos_quat" in obs and obs["ee_pos_quat"] is not None:
            state = np.asarray(obs["ee_pos_quat"], dtype=np.float32).reshape(-1)[:6]
        else:
            raise KeyError("pi0 requires obs['ee_pos_quat'] or obs['joint_positions']")

        gripper = 0.0
        if "gripper_position" in obs and obs["gripper_position"] is not None:
            gripper = float(np.asarray(obs["gripper_position"]).reshape(-1)[0])
        elif "joint_positions" in obs and obs["joint_positions"] is not None:
            joints = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1)
            if len(joints) > 6:
                gripper = float(joints[-1])
        return np.concatenate([state, np.asarray([gripper], dtype=np.float32)]).astype(np.float32)

    def _state(self, obs: Dict[str, Any]) -> np.ndarray:
        state = self._base_state(obs)
        if self.tactile_encoder is not None:
            state = np.concatenate([state, self._tactile_embedding(obs)], axis=0)
        if state.shape[0] < self.state_dim:
            state = np.pad(state, (0, self.state_dim - state.shape[0]))
        return state[: self.state_dim].astype(np.float32)

    def _tactile_embedding(self, obs: Dict[str, Any]) -> np.ndarray:
        left = self._first_obs_value(obs, ("tactile_left_rgb", "left_tactile_rgb"))
        right = self._first_obs_value(obs, ("tactile_right_rgb", "right_tactile_rgb"))
        if left is None and right is None:
            print("[pi0] warning: tactile image embedding enabled but tactile_left_rgb/tactile_right_rgb are missing.")
        return self.tactile_encoder.encode_pair(left, right)

    def _tactile_raw_images(self, obs: Dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray | None]:
        left = self._first_obs_value(obs, ("tactile_left_rgb", "left_tactile_rgb"))
        right = self._first_obs_value(obs, ("tactile_right_rgb", "right_tactile_rgb"))
        if left is None and right is None:
            print("[pi0] warning: tactile_feature_mode=raw_image but tactile_left_rgb/tactile_right_rgb are missing.")
        return (
            self._resize_image(left) if left is not None else None,
            self._resize_image(right) if right is not None else None,
        )

    @staticmethod
    def _first_obs_value(obs: Dict[str, Any], keys: tuple[str, ...]) -> Any | None:
        for key in keys:
            if key in obs and obs[key] is not None:
                return obs[key]
        return None

    def _policy_observation(self, obs: Dict[str, Any]) -> dict[str, Any]:
        tactile_left_rgb = tactile_right_rgb = None
        if self.include_tactile and self.tactile_feature_mode == "raw_image":
            tactile_left_rgb, tactile_right_rgb = self._tactile_raw_images(obs)
        return build_policy_observation(
            base_rgb=self._select_image(obs, "base"),
            wrist_rgb=self._select_image(obs, "wrist"),
            state=self._state(obs),
            prompt=self.prompt,
            tactile_left_rgb=tactile_left_rgb,
            tactile_right_rgb=tactile_right_rgb,
        )

    def _request_action_chunk(self, policy_obs: dict[str, Any], *, source: str) -> np.ndarray:
        start = time.perf_counter()
        chunk = np.asarray(self.client.action_chunk(policy_obs), dtype=np.float32)
        latency = time.perf_counter() - start
        print(f"[pi0] action_chunk latency ({source}): {latency:.3f}s shape={tuple(chunk.shape)}")
        return chunk

    def _enqueue_action_chunk(self, chunk: np.ndarray) -> None:
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        selected = chunk[: self.action_chunk_size]
        if len(selected) == 0:
            raise ValueError(f"Policy returned an empty action chunk with shape {chunk.shape}")
        for action in selected:
            self.action_queue.append(np.asarray(action, dtype=np.float32).reshape(-1))

    def _consume_prefetch_if_ready(self) -> None:
        thread = self._prefetch_thread
        if thread is not None and thread.is_alive():
            return
        if thread is not None:
            thread.join()
            self._prefetch_thread = None

        with self._prefetch_lock:
            result = self._prefetch_result
            error = self._prefetch_error
            self._prefetch_result = None
            self._prefetch_error = None

        if error is not None:
            raise RuntimeError("Asynchronous pi0 action prefetch failed") from error
        if result is not None:
            self._enqueue_action_chunk(result)

    def _finish_prefetch_blocking(self) -> None:
        thread = self._prefetch_thread
        if thread is not None:
            thread.join()
            self._prefetch_thread = None
        with self._prefetch_lock:
            self._prefetch_result = None
            self._prefetch_error = None

    def _start_prefetch_if_needed(self, obs: Dict[str, Any]) -> None:
        if not self.async_prefetch or len(self.action_queue) > self.prefetch_threshold:
            return
        if self._prefetch_thread is not None:
            return
        with self._prefetch_lock:
            if self._prefetch_result is not None or self._prefetch_error is not None:
                return

        policy_obs = self._policy_observation(obs)

        def worker() -> None:
            try:
                result = self._request_action_chunk(policy_obs, source="prefetch")
                with self._prefetch_lock:
                    self._prefetch_result = result
                    self._prefetch_error = None
            except BaseException as exc:
                with self._prefetch_lock:
                    self._prefetch_result = None
                    self._prefetch_error = exc

        self._prefetch_thread = threading.Thread(target=worker, name="pi0-action-prefetch", daemon=True)
        self._prefetch_thread.start()

    def _next_raw_action(self, obs: Dict[str, Any]) -> np.ndarray:
        self._consume_prefetch_if_ready()
        if not self.action_queue:
            if self._prefetch_thread is not None:
                print("[pi0] waiting for prefetched action_chunk")
                self._prefetch_thread.join()
                self._consume_prefetch_if_ready()
            if not self.action_queue:
                chunk = self._request_action_chunk(self._policy_observation(obs), source="sync")
                self._enqueue_action_chunk(chunk)
        action = self.action_queue.popleft()
        self._start_prefetch_if_needed(obs)
        return action

    def _stabilize_gripper(self, raw_gripper: float) -> float:
        raw_gripper = float(np.clip(raw_gripper, self.config.action_limits["min_gripper"], self.config.action_limits["max_gripper"]))
        previous = raw_gripper if self.last_gripper_action is None else self.last_gripper_action

        if self.gripper_closed:
            self.gripper_hold_steps += 1
            can_release = self.gripper_hold_steps >= self.gripper_min_hold_steps
            if raw_gripper <= self.gripper_open_threshold and can_release:
                self.gripper_closed = False
                self.gripper_hold_steps = 0
                target = raw_gripper
            else:
                target = max(previous, raw_gripper, self.gripper_close_threshold)
        elif raw_gripper >= self.gripper_close_threshold:
            self.gripper_closed = True
            self.gripper_hold_steps = 0
            target = raw_gripper
        elif raw_gripper <= self.gripper_open_threshold:
            target = raw_gripper
        else:
            target = previous

        smoothed = self.gripper_ema_alpha * target + (1.0 - self.gripper_ema_alpha) * previous
        smoothed = float(np.clip(smoothed, self.config.action_limits["min_gripper"], self.config.action_limits["max_gripper"]))
        self.last_gripper_action = smoothed
        return smoothed

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        raw_action = self._next_raw_action(obs)
        if self.predict_eef_delta and raw_action.shape[0] >= 3:
            max_xyz = float(np.max(np.abs(raw_action[:3])))
            if max_xyz > self.reject_translation_delta:
                raise RuntimeError(
                    "pi0 policy produced an unsafe EEF delta before clipping: "
                    f"xyz={raw_action[:3].tolist()}. This usually means the checkpoint was trained "
                    "with absolute/joint actions instead of ee_delta_6d_gripper. Stop robot rollout, "
                    "reconvert the dataset, and retrain."
                )
        if self.predict_eef_delta:
            scaled_action = raw_action.copy()
            if scaled_action.shape[0] >= 3:
                scaled_action[:3] *= self.eef_translation_scale
            if scaled_action.shape[0] >= 6:
                scaled_action[3:6] *= self.eef_rotation_scale
            action = clip_pi0_action(scaled_action, self.config.action_limits)
            if action.shape[0] >= 7:
                action[6] = self._stabilize_gripper(float(action[6]))
            if self.debug_actions:
                print(
                    "[pi0] action",
                    f"raw_xyz={raw_action[:3].tolist() if raw_action.shape[0] >= 3 else []}",
                    f"scaled_xyz={scaled_action[:3].tolist() if scaled_action.shape[0] >= 3 else []}",
                    f"cmd_xyz={action[:3].tolist() if action.shape[0] >= 3 else []}",
                    f"raw_rot={raw_action[3:6].tolist() if raw_action.shape[0] >= 6 else []}",
                    f"scaled_rot={scaled_action[3:6].tolist() if scaled_action.shape[0] >= 6 else []}",
                    f"cmd_rot={action[3:6].tolist() if action.shape[0] >= 6 else []}",
                    f"gripper_raw={float(raw_action[6]):.3f}" if raw_action.shape[0] >= 7 else "gripper_raw=NA",
                    f"gripper_cmd={float(action[6]):.3f}" if action.shape[0] >= 7 else "gripper_cmd=NA",
                    f"closed={self.gripper_closed}",
                    f"hold={self.gripper_hold_steps}",
                )
            curr_eef_pose = np.asarray(obs["ee_pos_quat"], dtype=np.float32).reshape(-1)[:6]
            return get_eef_pose(curr_eef_pose, action)

        action = raw_action
        if len(action) != len(np.asarray(obs["joint_positions"]).reshape(-1)):
            raise ValueError(
                "pi0 without _eef expects a joint-space policy output matching joint_positions. "
                f"Got action shape {action.shape}; use --agent pi0_eef for the current ee_delta_6d_gripper policy."
            )
        if self.last_joint_action is not None and self.last_joint_action.shape == action.shape:
            action = self.joint_ema_alpha * action + (1.0 - self.joint_ema_alpha) * self.last_joint_action
        action[-1] = np.clip(action[-1], self.config.action_limits["min_gripper"], self.config.action_limits["max_gripper"])
        if self.debug_actions:
            joints = np.asarray(obs["joint_positions"], dtype=np.float32).reshape(-1)
            diff = action - joints[: action.shape[0]]
            print(
                "[pi0] joint action",
                f"state={self._state(obs).tolist()}",
                f"raw={raw_action.tolist()}",
                f"cmd={action.tolist()}",
                f"current={joints[: action.shape[0]].tolist()}",
                f"diff={diff.tolist()}",
            )
        self.last_joint_action = action.copy()
        return action.astype(np.float32)
