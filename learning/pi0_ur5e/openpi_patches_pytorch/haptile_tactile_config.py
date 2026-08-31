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

    # Set the model specific defaults (mirrors Pi0Config).
    action_dim: int = 7
    action_horizon: int = 50
    max_token_len: int | None = None
    pytorch_compile_mode: str | None = None

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200)

    @property
    def model_type(self) -> _model.ModelType:
        # Reuses PI05: same transform branch (3-camera pi0.5-style prefix, no explicit state
        # token in the suffix) as the existing pi0_ur5e_cup config; tactile is an additive
        # branch on top, not a new transform family.
        return _model.ModelType.PI05

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
