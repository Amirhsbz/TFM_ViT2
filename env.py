import glob
import os
import pickle
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
from natsort import natsorted

from cameras.camera import CameraDriver
from robots.robot import Robot


class Rate:
    def __init__(self, rate: float):
        self.last = time.time()
        self.rate = rate

    def sleep(self) -> None:
        while self.last + 1.0 / self.rate > time.time():
            time.sleep(0.0001)
        self.last = time.time()


class EvalRobotEnv:
    def __init__(
        self,
        robot: Robot,
        traj_path: str,
        control_rate_hz: float,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self.traj_path = traj_path

        self.pkls = natsorted(
            glob.glob(os.path.join(self.traj_path, "*.pkl"), recursive=True)
        )
        print("Finished reading dir", self.traj_path)
        print("No. of files:", len(self.pkls))
        self.traj_len = len(self.pkls)
        self.count = 0

    def robot(self) -> Robot:
        """Get the robot object.

        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        # Return positive integer for batched envs.
        return self.traj_len

    def step_eef(self, eef_pose: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            eef_pose: end effector pose command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(eef_pose) == self._robot.num_dofs(), f"input:{len(eef_pose)}"
        self._robot.command_eef_pose(eef_pose)
        self._rate.sleep()
        return self.get_obs()

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.

        Args:
            joints: joint angles command to step the environment with.

        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_real_obs(self) -> Dict[str, Any]:
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            observations[f"{name}_rgb"] = image
            observations[f"{name}_depth"] = depth

        robot_obs = self._robot.get_observations()
        for k, v in robot_obs.items():
            observations[k] = v
        return observations

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.

        Returns:
            obs: observation from the environment.
        """
        if self.count >= self.traj_len:
            return None
        pkl = self.pkls[self.count]
        with open(pkl, "rb") as f:
            observations = pickle.load(f)
        self.count += 1
        return observations


class RobotEnv:
    DEPTH_DISPLAY_MIN_MM = 200
    DEPTH_DISPLAY_MAX_MM = 1500

    def __init__(
        self,
        robot: Robot,
        control_rate_hz: float = 100.0,
        camera_dict: Optional[Dict[str, CameraDriver]] = None,
        show_camera_view: bool = True,
        save_depth: bool = True,
    ) -> None:
        self._robot = robot
        self._rate = Rate(control_rate_hz)
        print("RobotEnv: control_rate_hz", control_rate_hz)
        self._camera_dict = {} if camera_dict is None else camera_dict
        self._show_camera_view = show_camera_view
        if self._show_camera_view:
            for name in list(self._camera_dict.keys()):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        self._save_depth = save_depth
        self.last_cv2_key: int = -1  # last non-(-1) key seen by cv2.waitKey in get_obs

    def robot(self) -> Robot:
        """Get the robot object.
        Returns:
            robot: the robot object.
        """
        return self._robot

    def __len__(self):
        # Return positive integer for batched envs.
        return 0

    @staticmethod
    def _depth_to_colormap(depth: np.ndarray) -> np.ndarray:
        depth_mm = depth.astype(np.float32)
        valid = depth_mm > 0

        depth_clipped = np.clip(
            depth_mm, RobotEnv.DEPTH_DISPLAY_MIN_MM, RobotEnv.DEPTH_DISPLAY_MAX_MM
        )
        depth_scaled = (
            (depth_clipped - RobotEnv.DEPTH_DISPLAY_MIN_MM)
            * 255.0
            / (RobotEnv.DEPTH_DISPLAY_MAX_MM - RobotEnv.DEPTH_DISPLAY_MIN_MM)
        )
        depth_uint8 = np.zeros(depth.shape, dtype=np.uint8)
        depth_uint8[valid] = depth_scaled[valid].astype(np.uint8)
        return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)

    @staticmethod
    def _compose_camera_view(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            image_batch = image[None, ...]
        elif image.ndim == 4:
            image_batch = image
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        if depth.ndim == 2:
            depth_batch = depth[None, ...]
        elif depth.ndim == 3:
            depth_batch = depth
        else:
            raise ValueError(f"Unsupported depth shape: {depth.shape}")

        if image_batch.shape[0] != depth_batch.shape[0]:
            raise ValueError(
                f"Mismatched camera count: image.shape={image.shape}, depth.shape={depth.shape}"
            )

        panels = []
        for camera_idx in range(image_batch.shape[0]):
            rgb = image_batch[camera_idx][:, :, ::-1]
            depth_colormap = RobotEnv._depth_to_colormap(depth_batch[camera_idx])

            if rgb.shape[:2] != depth_colormap.shape[:2] or rgb.dtype != depth_colormap.dtype:
                raise ValueError(
                    f"image.shape: {rgb.shape}, depth.shape: {depth_colormap.shape}, "
                    f"image.dtype: {rgb.dtype}, depth.dtype: {depth_colormap.dtype}"
                )

            panel = cv2.hconcat([rgb, depth_colormap])
            if image_batch.shape[0] > 1:
                cv2.putText(
                    panel,
                    f"cam {camera_idx}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            panels.append(panel)

        if len(panels) == 1:
            return panels[0]
        return cv2.vconcat(panels)

    def step_eef(self, eef_pose: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.
        Args:
            eef_pose: end effector pose command to step the environment with.
        Returns:
            obs: observation from the environment.
        """
        assert len(eef_pose) == self._robot.num_dofs(), f"input:{len(eef_pose)}"
        self._robot.command_eef_pose(eef_pose)
        self._rate.sleep()
        return self.get_obs()

    def step(self, joints: np.ndarray) -> Dict[str, Any]:
        """Step the environment forward.
        Args:
            joints: joint angles command to step the environment with.
        Returns:
            obs: observation from the environment.
        """
        assert len(joints) == (
            self._robot.num_dofs()
        ), f"input:{len(joints)}, robot:{self._robot.num_dofs()}"
        assert self._robot.num_dofs() == len(joints)
        self._robot.command_joint_state(joints)
        self._rate.sleep()
        return self.get_obs()

    def get_obs(self) -> Dict[str, Any]:
        """Get observation from the environment.
        Returns:
            obs: observation from the environment.
        """
        observations = {}
        for name, camera in self._camera_dict.items():
            image, depth = camera.read()
            # print("here")
            observations[f"{name}_rgb"] = image
            if hasattr(camera, "get_last_raw_frame_rgb"):
                raw_image = camera.get_last_raw_frame_rgb()
                if raw_image is not None:
                    observations[f"{name}_raw_rgb"] = raw_image
            if self._save_depth and not name.startswith("tactile_"):
                observations[f"{name}_depth"] = depth

            if self._show_camera_view:
                display_image = image
                display_depth = depth
                if hasattr(camera, "get_last_raw_frame_rgb") and name.startswith("tactile_"):
                    raw = camera.get_last_raw_frame_rgb()
                    if raw is not None:
                        display_image = raw
                        display_depth = np.zeros((raw.shape[0], raw.shape[1]), dtype=np.uint16)
                image_depth = self._compose_camera_view(display_image, display_depth)
                cv2.imshow(name, image_depth)
                _k = cv2.waitKey(1) & 0xFF
                if _k != 255:
                    self.last_cv2_key = _k

        robot_obs = self._robot.get_observations()
        for k, v in robot_obs.items():
            observations[k] = v
        return observations
