# BEGIN TELE_GSY_PI0_UR5E_CUP
# Paste this block into openpi/src/openpi/training/config.py immediately before:
#   if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
#
# It is written as an append-only patch so it does not require modifying
# OpenPI's existing config list entries.

import os as _tele_gsy_os
import numpy as _tele_gsy_np
import flax.traverse_util as _tele_gsy_traverse_util
import openpi.shared.nnx_utils as _tele_gsy_nnx_utils


def _tele_gsy_env_bool(name: str, default: bool) -> bool:
    value = _tele_gsy_os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _tele_gsy_env_int(name: str, default: int | None = None) -> int | None:
    value = _tele_gsy_os.environ.get(name)
    if value in {None, ""}:
        return default
    return int(value)


def _tele_gsy_parse_image(image) -> _tele_gsy_np.ndarray:
    image = _tele_gsy_np.asarray(image)
    if _tele_gsy_np.issubdtype(image.dtype, _tele_gsy_np.floating):
        image = _tele_gsy_np.clip(image, 0.0, 1.0) * 255.0
        image = image.astype(_tele_gsy_np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = _tele_gsy_np.moveaxis(image, 0, -1)
    return image.astype(_tele_gsy_np.uint8)


@dataclasses.dataclass(frozen=True)
class TeleGsyShapeTolerantCheckpointWeightLoader(weight_loaders.WeightLoader):
    params_path: str

    def load(self, params):
        loaded_params = _model.restore_params(
            weight_loaders.download.maybe_download(self.params_path),
            restore_type=_tele_gsy_np.ndarray,
        )
        flat_ref = _tele_gsy_traverse_util.flatten_dict(params, sep="/")
        flat_loaded = _tele_gsy_traverse_util.flatten_dict(loaded_params, sep="/")
        result = {}
        for key, value in flat_loaded.items():
            if key not in flat_ref:
                continue
            if value.shape != flat_ref[key].shape:
                continue
            result[key] = value.astype(flat_ref[key].dtype) if value.dtype != flat_ref[key].dtype else value
        for key, value in flat_ref.items():
            if key not in result:
                result[key] = value
        return _tele_gsy_traverse_util.unflatten_dict(result, sep="/")


def _tele_gsy_freeze_filter(model_config):
    freeze_mode = _tele_gsy_os.environ.get("PI0_UR5E_FREEZE_MODE", "default")
    if freeze_mode == "vision_action_head":
        # Train only the SigLIP image tower and robot-specific projection heads.
        # There are no SigLIP LoRA parameters in this OpenPI checkout; this mode
        # fully trains PaliGemma.img while freezing the language/action-expert LLMs.
        trainable = _tele_gsy_nnx_utils.PathRegex(".*(PaliGemma/img|state_proj|action_in_proj|action_out_proj).*")
        return nnx.All(nnx.Param, nnx.Not(trainable))
    if freeze_mode == "action_head":
        # Train only robot-specific state/action projection heads.
        trainable = _tele_gsy_nnx_utils.PathRegex(".*(state_proj|action_in_proj|action_out_proj).*")
        return nnx.All(nnx.Param, nnx.Not(trainable))

    base_filter = model_config.get_freeze_filter() if _TELE_GSY_PI0_UR5E_LORA else nnx.Nothing()
    train_resized_projection = nnx.Not(_tele_gsy_nnx_utils.PathRegex(".*(state_proj|action_in_proj|action_out_proj).*"))
    return nnx.All(base_filter, train_resized_projection)


@dataclasses.dataclass(frozen=True)
class TeleGsyUR5eInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0
    camera_padding_strategy: str = "zeros"

    def __call__(self, data: dict) -> dict:
        base_image = _tele_gsy_parse_image(data["base_rgb"])
        wrist_image = _tele_gsy_parse_image(data["wrist_rgb"])
        if self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            if self.camera_padding_strategy == "duplicate_wrist":
                images = (base_image, wrist_image, wrist_image)
            elif self.camera_padding_strategy == "duplicate_base":
                images = (base_image, base_image, wrist_image)
            else:
                images = (base_image, _tele_gsy_np.zeros_like(base_image), wrist_image)
            # FAST attends over all slots, including the padded one.
            image_masks = (_tele_gsy_np.True_, _tele_gsy_np.True_, _tele_gsy_np.True_)
        else:
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            if self.camera_padding_strategy == "duplicate_wrist":
                right_wrist_image = wrist_image
                right_wrist_mask = _tele_gsy_np.True_
            elif self.camera_padding_strategy == "duplicate_base":
                right_wrist_image = base_image
                right_wrist_mask = _tele_gsy_np.True_
            else:
                right_wrist_image = _tele_gsy_np.zeros_like(base_image)
                right_wrist_mask = _tele_gsy_np.False_
            images = (base_image, wrist_image, right_wrist_image)
            image_masks = (_tele_gsy_np.True_, _tele_gsy_np.True_, right_wrist_mask)

        inputs = {
            "state": _tele_gsy_np.asarray(data["state"], dtype=_tele_gsy_np.float32),
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            inputs["actions"] = _tele_gsy_np.asarray(data["actions"], dtype=_tele_gsy_np.float32)
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        # Raw single-frame tactile images for the FTP1-style tactile-expert model (see
        # openpi_patches_pytorch/haptile_tactile_pytorch.py). Placed under their own top-level
        # keys (not inside "image"/"image_mask") so they never route into the SigLIP vision
        # tower -- only present when TeleGsyLeRobotUR5eDataConfig.include_tactile_images=True
        # repacks them in, so this is a no-op for the existing non-tactile pi0_ur5e_cup config.
        if "tactile_left_rgb" in data:
            inputs["tactile_left_image"] = _tele_gsy_parse_image(data["tactile_left_rgb"])
        if "tactile_right_rgb" in data:
            inputs["tactile_right_image"] = _tele_gsy_parse_image(data["tactile_right_rgb"])
        return inputs


@dataclasses.dataclass(frozen=True)
class TeleGsyUR5eOutputs(_transforms.DataTransformFn):
    action_dim: int = 7
    action_format: str = "joint_position_gripper"

    def __call__(self, data: dict) -> dict:
        return {"actions": _tele_gsy_np.asarray(data["actions"][:, : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class TeleGsyLeRobotUR5eDataConfig(DataConfigFactory):
    expected_state_dim: int | None = None
    action_dim: int = 7
    action_format: str = "joint_position_gripper"
    camera_padding_strategy: str = "zeros"
    use_delta_actions: bool = True
    # Repacks observation.images.tactile_left_rgb/tactile_right_rgb (written by the pi0_ur5e
    # data pipeline's raw_image tactile mode) into TeleGsyUR5eInputs. Defaults to False: the
    # LeRobot RepackTransform raises KeyError on a missing source column, so this must stay off
    # for the existing non-tactile pi0_ur5e_cup dataset/config and only be turned on for a
    # dataset actually converted with --tactile-feature-mode raw_image (e.g. the tactile
    # TrainConfig in haptile_train_config_patch.py).
    include_tactile_images: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_structure = {
            "base_rgb": "observation.images.base_rgb",
            "wrist_rgb": "observation.images.wrist_rgb",
            "state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }
        if self.include_tactile_images:
            repack_structure["tactile_left_rgb"] = "observation.images.tactile_left_rgb"
            repack_structure["tactile_right_rgb"] = "observation.images.tactile_right_rgb"
        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_structure)])
        data_transforms = _transforms.Group(
            inputs=[
                TeleGsyUR5eInputs(
                    model_type=model_config.model_type,
                    camera_padding_strategy=self.camera_padding_strategy,
                )
            ],
            outputs=[TeleGsyUR5eOutputs(action_dim=self.action_dim, action_format=self.action_format)],
        )
        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory(
            default_prompt=_tele_gsy_os.environ.get(
                "PI0_UR5E_DEFAULT_PROMPT",
                "pick up the paper cup and place it on the target",
            )
        )(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


_TELE_GSY_PI0_UR5E_LORA = _tele_gsy_env_bool("PI0_UR5E_LORA", True)
_TELE_GSY_PI0_UR5E_REPO_ID = _tele_gsy_os.environ.get("PI0_UR5E_LEROBOT_REPO_ID", "local/pi0_ur5e_cup")
_TELE_GSY_PI0_UR5E_ACTION_FORMAT = _tele_gsy_os.environ.get("PI0_UR5E_ACTION_FORMAT", "joint_position_gripper")
_TELE_GSY_PI0_UR5E_USE_DELTA_ACTIONS = _tele_gsy_env_bool("PI0_UR5E_USE_DELTA_ACTIONS", True)
_TELE_GSY_PI0_UR5E_ACTION_ORDER = (
    ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    if _TELE_GSY_PI0_UR5E_ACTION_FORMAT == "ee_delta_6d_gripper"
    else ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
)
_TELE_GSY_PI0_UR5E_PI05 = _tele_gsy_env_bool("PI0_UR5E_PI05", True)
_TELE_GSY_PI0_UR5E_MODEL_FAMILY = _tele_gsy_os.environ.get(
    "PI0_UR5E_MODEL_FAMILY",
    "pi05" if _TELE_GSY_PI0_UR5E_PI05 else "pi0",
)
_TELE_GSY_PI0_UR5E_MODEL_ACTION_DIM = _tele_gsy_env_int("PI0_UR5E_STATE_DIM", 32)
_TELE_GSY_PI0_UR5E_MAX_TOKEN_LEN = _tele_gsy_env_int("PI0_UR5E_MAX_TOKEN_LEN")
_TELE_GSY_PI0_UR5E_MODEL = (
    pi0_fast.Pi0FASTConfig(
        paligemma_variant="gemma_2b_lora" if _TELE_GSY_PI0_UR5E_LORA else "gemma_2b",
        action_dim=_TELE_GSY_PI0_UR5E_MODEL_ACTION_DIM,
        action_horizon=_tele_gsy_env_int("PI0_UR5E_ACTION_HORIZON", 10),
        max_token_len=_TELE_GSY_PI0_UR5E_MAX_TOKEN_LEN or 180,
    )
    if _TELE_GSY_PI0_UR5E_MODEL_FAMILY == "pi0_fast"
    else
    pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        pi05=_TELE_GSY_PI0_UR5E_PI05,
        action_dim=_TELE_GSY_PI0_UR5E_MODEL_ACTION_DIM,
        action_horizon=_tele_gsy_env_int("PI0_UR5E_ACTION_HORIZON", 50),
        max_token_len=_TELE_GSY_PI0_UR5E_MAX_TOKEN_LEN,
    )
    if _TELE_GSY_PI0_UR5E_LORA
    else pi0_config.Pi0Config(
        pi05=_TELE_GSY_PI0_UR5E_PI05,
        action_dim=_TELE_GSY_PI0_UR5E_MODEL_ACTION_DIM,
        action_horizon=_tele_gsy_env_int("PI0_UR5E_ACTION_HORIZON", 50),
        max_token_len=_TELE_GSY_PI0_UR5E_MAX_TOKEN_LEN,
    )
)
_TELE_GSY_PI0_UR5E_CHECKPOINT = (
    "pi0_fast_base"
    if _TELE_GSY_PI0_UR5E_MODEL_FAMILY == "pi0_fast"
    else "pi05_base"
    if _TELE_GSY_PI0_UR5E_PI05
    else "pi0_base"
)

_CONFIGS.append(
    TrainConfig(
        name="pi0_ur5e_cup",
        # Keep the base pi0/pi0.5 action dimension for checkpoint compatibility.
        # The dataset/policy action dimension is 7 and is sliced in TeleGsyUR5eOutputs.
        model=_TELE_GSY_PI0_UR5E_MODEL,
        data=TeleGsyLeRobotUR5eDataConfig(
            repo_id=_TELE_GSY_PI0_UR5E_REPO_ID,
            assets=AssetsConfig(asset_id=_tele_gsy_os.environ.get("PI0_UR5E_ASSET_ID", _TELE_GSY_PI0_UR5E_REPO_ID)),
            base_config=DataConfig(prompt_from_task=False),
            expected_state_dim=_tele_gsy_env_int("PI0_UR5E_STATE_DIM"),
            action_dim=7,
            action_format=_TELE_GSY_PI0_UR5E_ACTION_FORMAT,
            camera_padding_strategy=_tele_gsy_os.environ.get("PI0_UR5E_CAMERA_PADDING", "zeros"),
            use_delta_actions=_TELE_GSY_PI0_UR5E_USE_DELTA_ACTIONS,
        ),
        weight_loader=TeleGsyShapeTolerantCheckpointWeightLoader(
            f"gs://openpi-assets/checkpoints/{_TELE_GSY_PI0_UR5E_CHECKPOINT}/params"
        ),
        num_train_steps=_tele_gsy_env_int("PI0_UR5E_TRAIN_STEPS", 3000),
        batch_size=_tele_gsy_env_int("PI0_UR5E_BATCH_SIZE", 16),
        assets_base_dir=_tele_gsy_os.environ.get("PI0_UR5E_ASSETS_BASE_DIR", "./assets"),
        checkpoint_base_dir=_tele_gsy_os.environ.get("PI0_UR5E_CHECKPOINT_BASE_DIR", "./checkpoints"),
        keep_period=_tele_gsy_env_int("PI0_UR5E_KEEP_PERIOD", 1000),
        freeze_filter=_tele_gsy_freeze_filter(_TELE_GSY_PI0_UR5E_MODEL),
        ema_decay=None if (_TELE_GSY_PI0_UR5E_LORA or _TELE_GSY_PI0_UR5E_MODEL_FAMILY == "pi0_fast") else 0.99,
        policy_metadata={
            "robot_type": "ur5e",
            "model_family": _TELE_GSY_PI0_UR5E_MODEL_FAMILY,
            "action_dim": 7,
            "action_format": _TELE_GSY_PI0_UR5E_ACTION_FORMAT,
            "action_order": _TELE_GSY_PI0_UR5E_ACTION_ORDER,
            "joint_action_is_delta": _TELE_GSY_PI0_UR5E_USE_DELTA_ACTIONS,
            "gripper_is_delta": False,
            "camera_padding_strategy": _tele_gsy_os.environ.get("PI0_UR5E_CAMERA_PADDING", "zeros"),
            "freeze_mode": _tele_gsy_os.environ.get("PI0_UR5E_FREEZE_MODE", "default"),
        },
    )
)
# END TELE_GSY_PI0_UR5E_CUP
