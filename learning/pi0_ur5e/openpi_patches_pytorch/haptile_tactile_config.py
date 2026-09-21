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

    # Vision tower (SigLIP) training strategy -- "full" (default, matches Pi0Config.get_freeze_filter's
    # JAX-side precedent: never frozen or LoRA'd), "lora" (freeze base, train rank-`vision_lora_rank`
    # adapters), or "frozen" (no adaptation at all). "full" is the established default for this
    # project's larger/more diverse datasets; "lora"/"frozen" exist for small-dataset regimes (a
    # handful of demos per task) where full fine-tuning of a ~400M-param pretrained vision tower
    # risks catastrophic forgetting / overfitting -- see the "Vision tower training strategy"
    # section of ftp1_tactile_expert_port.md for the full reasoning. "lora"/"frozen" only make
    # sense adapting a *pretrained* vision tower (bundled in the same checkpoint as the VLM), so
    # train_haptile_tactile_pytorch.py requires --pytorch_weight_path to be set for either.
    vision_tower_mode: str = "full"
    vision_lora_rank: int = 16
    vision_lora_alpha: float = 16.0

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
        if self.vision_tower_mode not in ("full", "lora", "frozen"):
            raise ValueError(f"vision_tower_mode must be one of 'full'/'lora'/'frozen', got {self.vision_tower_mode!r}")

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
        """Rebuilds the architecture the checkpoint was saved with, then loads it.

        A checkpoint trained with LoRA carries peft-injected parameter names
        ("...q_proj.base_layer.weight", "...q_proj.lora_A.default.weight") rather than the plain
        "...q_proj.weight" a freshly-constructed model has, so loading it into a bare model fails
        on every renamed and every extra key. train_haptile_tactile_pytorch.py applies LoRA itself
        (after loading pretrained weights); this is the serving-side counterpart of that sequence.

        Whether to apply LoRA is decided from the checkpoint's own keys rather than from this
        config: the tactile TrainConfig always names the "_lora" gemma variants, but training only
        actually applies LoRA when --pytorch_weight_path was given, so the config says what *could*
        have been adapted, not what this particular run did.
        """
        import dataclasses as _dc
        import logging
        import pathlib

        import safetensors
        import safetensors.torch

        from openpi.models_pytorch.haptile_tactile_pytorch import HaptileTactilePI0Pytorch

        with safetensors.safe_open(weight_path, framework="pt") as f:
            lora_keys = [key for key in f.keys() if "lora_" in key]  # noqa: SIM118
        has_vision_lora = any("vision_tower" in key for key in lora_keys)
        has_backbone_lora = any("vision_tower" not in key for key in lora_keys)

        model_config = self
        if has_vision_lora:
            # configure_vision_tower_training() reads vision_tower_mode off the config, which
            # defaults to "full" (a no-op) unless the serving machine happens to set
            # PI0_UR5E_TACTILE_VISION_TOWER_MODE -- so force it here instead of depending on the
            # serve-time environment matching the training environment.
            overrides = {"vision_tower_mode": "lora"}
            overrides.update(_read_trained_vision_lora_params(pathlib.Path(weight_path).parent))
            model_config = _dc.replace(self, **overrides)

        model = HaptileTactilePI0Pytorch(config=model_config)
        if has_backbone_lora:
            model.apply_lora_to_backbone()
        if has_vision_lora:
            model.configure_vision_tower_training()
        if lora_keys:
            logging.info(
                "Rebuilt LoRA adapters before loading (backbone=%s, vision_tower=%s, rank=%s, alpha=%s)",
                has_backbone_lora,
                has_vision_lora,
                model_config.vision_lora_rank,
                model_config.vision_lora_alpha,
            )

        safetensors.torch.load_model(model, weight_path)
        return model


def _read_trained_vision_lora_params(checkpoint_dir) -> dict:
    """Reads the vision-tower LoRA rank/alpha the checkpoint was actually trained with.

    Both must match training, and neither is fully recoverable from the weights: a wrong rank
    changes the adapter shapes and fails the load loudly, but a wrong alpha only rescales them,
    so it loads cleanly and silently changes what the policy outputs. metadata.pt (written
    alongside model.safetensors by train_haptile_tactile_pytorch.py's save_checkpoint) records
    the training config, which is a more reliable source than serve-time environment variables.

    Falls back to {} -- i.e. whatever the serving config already holds -- if metadata.pt is
    absent (a partially copied checkpoint) or unreadable (it is a pickle, so it can fail to load
    against a different openpi revision than the one that wrote it).
    """
    import logging

    metadata_path = checkpoint_dir / "metadata.pt"
    if not metadata_path.exists():
        logging.warning("No metadata.pt next to the checkpoint; using the config's vision LoRA rank/alpha.")
        return {}

    import torch

    try:
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
        trained_model_config = metadata["config"]["model"]
    except Exception:
        logging.exception("Could not read metadata.pt; using the config's vision LoRA rank/alpha.")
        return {}

    return {
        field: trained_model_config[field]
        for field in ("vision_lora_rank", "vision_lora_alpha")
        if field in trained_model_config
    }
