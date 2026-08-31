# BEGIN TELE_GSY_PI0_UR5E_CUP_TACTILE
# Paste this block into openpi/src/openpi/training/config.py immediately before:
#   if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
#
# It is written as an append-only patch, same convention as pi0_ur5e_cup_config.py's
# TELE_GSY_PI0_UR5E_CUP block (which must already be installed -- this block reuses that one's
# TeleGsyUR5eInputs/TeleGsyUR5eOutputs/TeleGsyLeRobotUR5eDataConfig/
# TeleGsyShapeTolerantCheckpointWeightLoader classes rather than redefining them).
#
# Adds a second TrainConfig, "pi0_ur5e_cup_tactile", using the new HaptileTactileConfig/
# HaptileTactilePI0Pytorch model (openpi_patches_pytorch/haptile_tactile_{config,pytorch}.py,
# installed alongside this block by scripts/install_openpi_pytorch_patch.py) instead of the
# plain pi0_config.Pi0Config/PI0Pytorch the first TrainConfig uses. PyTorch-only: unlike the
# first TrainConfig, this one has no JAX/pi0_fast branch, since HaptileTactilePI0Pytorch has no
# JAX counterpart (HaptileTactileConfig.create() raises NotImplementedError by design).

from openpi.models_pytorch.haptile_tactile_config import HaptileTactileConfig


def _tele_gsy_tactile_env_bool(name: str, default: bool) -> bool:
    return _tele_gsy_env_bool(name, default)


def _tele_gsy_tactile_env_int(name: str, default: int | None = None) -> int | None:
    return _tele_gsy_env_int(name, default)


_TELE_GSY_PI0_UR5E_TACTILE_REPO_ID = _tele_gsy_os.environ.get(
    "PI0_UR5E_TACTILE_LEROBOT_REPO_ID", "local/pi0_ur5e_cup_tactile"
)
_TELE_GSY_PI0_UR5E_TACTILE_ACTION_FORMAT = _tele_gsy_os.environ.get(
    "PI0_UR5E_TACTILE_ACTION_FORMAT", "joint_position_gripper"
)
_TELE_GSY_PI0_UR5E_TACTILE_USE_DELTA_ACTIONS = _tele_gsy_tactile_env_bool("PI0_UR5E_TACTILE_USE_DELTA_ACTIONS", True)
_TELE_GSY_PI0_UR5E_TACTILE_ACTION_ORDER = (
    ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
    if _TELE_GSY_PI0_UR5E_TACTILE_ACTION_FORMAT == "ee_delta_6d_gripper"
    else ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
)
_TELE_GSY_PI0_UR5E_TACTILE_USE_TACTILE_INPUT = _tele_gsy_tactile_env_bool("PI0_UR5E_TACTILE_USE_TACTILE_INPUT", True)
_TELE_GSY_PI0_UR5E_TACTILE_MODEL = HaptileTactileConfig(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    # head_dim must match paligemma's across every expert (FTP1PaliGemmaWithExpertModel applies
    # one shared rotary embedding sized for paligemma's head_dim to all branches) -- gemma_2b(_lora)
    # and gemma_300m(_lora) both have head_dim=256, so this holds; do not swap in "dummy" here.
    tactile_expert_variant=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_EXPERT_VARIANT", "gemma_300m"),
    use_tactile_input=_TELE_GSY_PI0_UR5E_TACTILE_USE_TACTILE_INPUT,
    action_dim=7,
    action_horizon=_tele_gsy_tactile_env_int("PI0_UR5E_TACTILE_ACTION_HORIZON", 50),
    max_token_len=_tele_gsy_tactile_env_int("PI0_UR5E_TACTILE_MAX_TOKEN_LEN"),
    dtype=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_DTYPE", "bfloat16"),
    pytorch_compile_mode=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_PYTORCH_COMPILE_MODE") or None,
)

_CONFIGS.append(
    TrainConfig(
        name="pi0_ur5e_cup_tactile",
        model=_TELE_GSY_PI0_UR5E_TACTILE_MODEL,
        data=TeleGsyLeRobotUR5eDataConfig(
            repo_id=_TELE_GSY_PI0_UR5E_TACTILE_REPO_ID,
            assets=AssetsConfig(
                asset_id=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_ASSET_ID", _TELE_GSY_PI0_UR5E_TACTILE_REPO_ID)
            ),
            base_config=DataConfig(prompt_from_task=False),
            expected_state_dim=_tele_gsy_tactile_env_int("PI0_UR5E_TACTILE_STATE_DIM", 7),
            action_dim=7,
            action_format=_TELE_GSY_PI0_UR5E_TACTILE_ACTION_FORMAT,
            camera_padding_strategy=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_CAMERA_PADDING", "zeros"),
            use_delta_actions=_TELE_GSY_PI0_UR5E_TACTILE_USE_DELTA_ACTIONS,
            include_tactile_images=True,
        ),
        # No weight_loader: TeleGsyShapeTolerantCheckpointWeightLoader (used by the first
        # TrainConfig) shape-tolerantly restores a JAX params pytree, which doesn't correspond
        # to this PyTorch model's state_dict naming at all. To seed from a pretrained PyTorch
        # pi0.5 checkpoint, pass --pytorch_weight_path to train_haptile_tactile_pytorch.py
        # instead (TrainConfig.pytorch_weight_path); left unset (None) here since there's no
        # tactile-expert-shaped base checkpoint to default to.
        num_train_steps=_tele_gsy_tactile_env_int("PI0_UR5E_TACTILE_TRAIN_STEPS", 3000),
        batch_size=_tele_gsy_tactile_env_int("PI0_UR5E_TACTILE_BATCH_SIZE", 16),
        assets_base_dir=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_ASSETS_BASE_DIR", "./assets"),
        checkpoint_base_dir=_tele_gsy_os.environ.get("PI0_UR5E_TACTILE_CHECKPOINT_BASE_DIR", "./checkpoints"),
        keep_period=_tele_gsy_tactile_env_int("PI0_UR5E_TACTILE_KEEP_PERIOD", 1000),
        ema_decay=None,
        policy_metadata={
            "robot_type": "ur5e",
            "model_family": "pi0_ur5e_cup_tactile",
            "action_dim": 7,
            "action_format": _TELE_GSY_PI0_UR5E_TACTILE_ACTION_FORMAT,
            "action_order": _TELE_GSY_PI0_UR5E_TACTILE_ACTION_ORDER,
            "joint_action_is_delta": _TELE_GSY_PI0_UR5E_TACTILE_USE_DELTA_ACTIONS,
            "gripper_is_delta": False,
            "camera_padding_strategy": _tele_gsy_os.environ.get("PI0_UR5E_TACTILE_CAMERA_PADDING", "zeros"),
            "use_tactile_input": _TELE_GSY_PI0_UR5E_TACTILE_USE_TACTILE_INPUT,
        },
    )
)
# END TELE_GSY_PI0_UR5E_CUP_TACTILE
