"""Model config for HaptileTactilePI0Pytorch.

Mirrors openpi.models.pi0_config.Pi0Config's real field set (not FTP1's ftp1_model_config.py,
which also carries state_input_mode/tactile_tokenizer_config/etc. for the full heterogeneous
design this port explicitly skips), plus two tactile-specific fields.

Authored in the tele-amir repo; installed into $OPENPI_ROOT/src/openpi/models_pytorch/ by
scripts/install_openpi_pytorch_patch.py.
"""

from __future__ import annotations

import dataclasses

import openpi.models.gemma as _gemma
import openpi.models.model as _model


@dataclasses.dataclass(frozen=True)
class HaptileTactileConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b_lora"
    action_expert_variant: _gemma.Variant = "gemma_300m_lora"
    # "gemma_small" (used by FTP1's own ftp1_model_config.py) doesn't exist in plain openpi's
    # gemma.py -- only dummy/gemma_300m(_lora)/gemma_2b(_lora) are defined there. gemma_300m is
    # the closest real variant: a new module with nothing to reuse a LoRA adapter against, so it
    # trains full-rank rather than _lora.
    tactile_expert_variant: _gemma.Variant = "gemma_300m"
    use_tactile_input: bool = True

    # Whether the tactile ViT encoder should initialize from a pretrained T3 checkpoint
    # (downloaded from HuggingFace, see haptile_tactile_encoder.py's
    # T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL) rather than random weights, then
    # fine-tune from there. t3_sensor_name selects which sensor-specific checkpoint to load --
    # must match the physical tactile sensor, not just the "GelSight" brand: "gs_tag" is the
    # marker/dot-pattern gel variant (tracks shear/slip via marker displacement), "gs_black" is
    # the plain/markerless black-gel variant (photometric-stereo surface reconstruction, no
    # markers) -- confirmed against the actual hardware (visible dots/markers on the gel pad) to
    # be "gs_tag". Defaults True: verified end-to-end (download + strict load + forward pass) --
    # see haptile_tactile_encoder.py's T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL comment
    # for the size-class bug (t3_large, not t3_medium) that had to be fixed first.
    load_t3_tactile_checkpoint: bool = True
    t3_sensor_name: str = "gs_tag"
    t3_checkpoint_cache_dir: str | None = None

    # Selects the pi0 vs pi0.5 transform/embedding convention, mirroring Pi0Config.pi05.
    # Defaults to False (plain pi0): every other task's TrainConfig in this repo (fold_Tshirt
    # included) passes --pi05 false to train_pi0_base.sh -- pi0.5 was never this project's actual
    # convention, only the unset-env-var default of the *unrelated* plain pi0_ur5e_cup config
    # (which this field used to be copied from without checking real usage). Set True to use
    # pi0.5's discrete-state-in-prompt convention instead.
    pi05: bool = False

    # Set the model specific defaults (mirrors Pi0Config).
    action_dim: int = 7
    action_horizon: int = 50
    max_token_len: int | None = None
    pytorch_compile_mode: str | None = None
    # Not used directly by the model -- read by ModelTransformFactory (see model_type below).
    # Resolved from `pi05` in __post_init__ if left unset, exactly like Pi0Config does. Only
    # meaningful when pi05=True: HaptileTactilePI0Pytorch.embed_suffix (like PI0Pytorch's own
    # pi05 branch) never embeds `state` as a continuous suffix token when pi05=True -- state only
    # reaches the model via TokenizePrompt discretizing it into the tokenized prompt text in that
    # case. When pi05=False, embed_suffix embeds `state` directly as a continuous suffix token
    # instead (PI0Pytorch's non-pi05 branch), so discrete_state_input must be False there too.
    discrete_state_input: bool | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    def model_type(self) -> _model.ModelType:
        # PI05 when pi05=True: 3-camera pi0.5-style prefix, discretized state folded into the
        # tokenized prompt rather than a continuous suffix token. PI0 when pi05=False (the
        # default, matching this project's actual convention -- see the `pi05` field comment):
        # plain pi0-style prefix, state embedded as its own continuous suffix token instead.
        # Either way, tactile is an additive branch on top, not a new transform family.
        return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0

    # This config only supports PyTorch training/inference (like FTP1ModelConfig) -- the JAX
    # abstract methods on BaseModelConfig are stubbed out rather than implemented.
    def create(self, rng):
        raise NotImplementedError("HaptileTactileConfig only supports the PyTorch backend; use load_pytorch.")

    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        raise NotImplementedError("HaptileTactileConfig only supports the PyTorch backend; use load_pytorch.")

    def get_freeze_filter(self):
        raise NotImplementedError("HaptileTactileConfig only supports the PyTorch backend; use load_pytorch.")

    def load_pytorch(self, train_config, weight_path: str):
        import safetensors.torch

        from openpi.models_pytorch.haptile_tactile_pytorch import HaptileTactilePI0Pytorch

        model = HaptileTactilePI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model
