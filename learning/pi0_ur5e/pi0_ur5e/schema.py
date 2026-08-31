from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

ActionMode = Literal[
    "ee_delta_6d_gripper",
    "ee_absolute_6d_gripper",
    "joint_position_gripper",
    "joint_delta_gripper",
]


@dataclass
class Episode:
    episode_id: str
    timestamps: np.ndarray
    base_rgb: Any
    wrist_rgb: Any
    robot_state: np.ndarray
    action: np.ndarray
    gripper_state: np.ndarray | None = None
    tactile: np.ndarray | None = None
    tactile_left_rgb: np.ndarray | None = None
    tactile_right_rgb: np.ndarray | None = None
    language_instruction: str = "pick up the paper cup and place it on the target"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        length = len(self.timestamps)
        for name in ("robot_state", "action"):
            value = getattr(self, name)
            if value is None or len(value) != length:
                raise ValueError(f"{self.episode_id}: {name} length does not match timestamps")
        if self.gripper_state is not None and len(self.gripper_state) != length:
            raise ValueError(f"{self.episode_id}: gripper_state length does not match timestamps")
        if self.tactile is not None and len(self.tactile) != length:
            raise ValueError(f"{self.episode_id}: tactile length does not match timestamps")
        if self.tactile_left_rgb is not None and len(self.tactile_left_rgb) != length:
            raise ValueError(f"{self.episode_id}: tactile_left_rgb length does not match timestamps")
        if self.tactile_right_rgb is not None and len(self.tactile_right_rgb) != length:
            raise ValueError(f"{self.episode_id}: tactile_right_rgb length does not match timestamps")


@dataclass
class Pi0Ur5eConfig:
    action_mode: ActionMode = "joint_position_gripper"
    image_size: tuple[int, int] = (224, 224)
    camera_padding_strategy: str = "duplicate_base"
    include_tactile: bool = False
    tactile_feature_mode: str = "none"
    tactile_embedding_dim: int = 128
    default_prompt: str = "pick up the paper cup and place it on the target"
    hz: float | None = None
    field_map: dict[str, Any] = field(default_factory=dict)
    action_limits: dict[str, float] = field(default_factory=lambda: {
        "max_translation_delta": 0.05,
        "max_rotation_delta": 0.35,
        "min_gripper": 0.0,
        "max_gripper": 1.0,
    })
    workspace_bounds: dict[str, list[float]] = field(default_factory=lambda: {
        "x": [-0.8, 0.8],
        "y": [-0.8, 0.8],
        "z": [0.0, 0.8],
    })


def pathify(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)
