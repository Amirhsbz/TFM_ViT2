"""HaptileTactilePI0Pytorch: PI0Pytorch + a dedicated tactile-expert branch.

Phase-1 port of FTP1's tactile-fusion architecture onto Haptile's existing PI0Pytorch-based
training path: a learned ViT encoder (HaptileTactileEncoder) produces one CLS-pooled token per
tactile camera, cross-attended by the action expert via a third Gemma "tactile expert" branch
(FTP1PaliGemmaWithExpertModel, copied verbatim from ftp1-policy), with FTP1's KV-cache-reuse
inference strategy (tactile tokens computed once per action-chunk prediction, not once per
flow-matching denoising step).

This class intentionally mirrors openpi.models_pytorch.pi0_pytorch.PI0Pytorch almost line for
line -- embed_prefix, action_in_proj/action_out_proj, sample_noise/sample_time, gradient
checkpointing helpers, and the overall forward/sample_actions/denoise_step structure are the
same. The only real additions are: (a) an always-pi05 embed_suffix (this model always uses
ModelType.PI05, so the non-pi05 branch of PI0Pytorch.embed_suffix is simply dropped rather than
carried over unused), (b) embed_tactile, and (c) routing everything through
FTP1PaliGemmaWithExpertModel's 3-branch attention instead of the 2-branch model, via
ftp1_attention_masks's block-structured layout builders.

Authored in the tele-amir repo; installed into $OPENPI_ROOT/src/openpi/models_pytorch/ by
scripts/install_openpi_pytorch_patch.py.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.cache_utils import DynamicCache

import openpi.models.gemma as _gemma
from openpi.models_pytorch.ftp1_attention_masks import build_action_denoise_layout
from openpi.models_pytorch.ftp1_attention_masks import build_expert_attention_layout
from openpi.models_pytorch.ftp1_attention_masks import build_prefix_attention_layout
from openpi.models_pytorch.ftp1_attention_masks import build_tactile_attention_layout
from openpi.models_pytorch.ftp1_gemma_pytorch import FTP1PaliGemmaWithExpertModel
from openpi.models_pytorch.haptile_tactile_encoder import HaptileTactileEncoder
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import sample_beta
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


class HaptileTactilePI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # This model always targets ModelType.PI05 (see HaptileTactileConfig.model_type):
        # unlike PI0Pytorch, there is no non-pi05 code path here at all.

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        tactile_expert_config = (
            _gemma.get_config(config.tactile_expert_variant) if config.use_tactile_input else None
        )

        self.paligemma_with_expert = FTP1PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            tactile_expert_config,
            use_tactile_input=config.use_tactile_input,
            use_adarms=[False, True],
            precision=config.dtype,
        )

        self.tactile_encoder = (
            HaptileTactileEncoder(token_dim=tactile_expert_config.width) if config.use_tactile_input else None
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        if config.pytorch_compile_mode is not None:
            self.sample_actions = torch.compile(self.sample_actions, mode=config.pytorch_compile_mode)

        self.gradient_checkpointing_enabled = False

        # Same sentinel check as PI0Pytorch -- FTP1PaliGemmaWithExpertModel also hard-depends on
        # the adaRMS transformers_replace patch (modeling_gemma._gated_residual, GemmaRMSNorm's
        # cond kwarg).
        msg = (
            "transformers_replace is not installed correctly. Please install it with "
            "`uv pip install transformers==4.53.2` and "
            "`cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        )
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        if self.paligemma_with_expert.gemma_tactile_expert is not None:
            self.paligemma_with_expert.gemma_tactile_expert.model.gradient_checkpointing = True
        logging.info("Enabled gradient checkpointing for HaptileTactilePI0Pytorch model")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        if self.paligemma_with_expert.gemma_tactile_expert is not None:
            self.paligemma_with_expert.gemma_tactile_expert.model.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for HaptileTactilePI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        processed = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(processed.images.values()),
            list(processed.image_masks.values()),
            processed.tokenized_prompt,
            processed.tokenized_prompt_mask,
            processed.state,
            getattr(processed, "tactile_left_image", None),
            getattr(processed, "tactile_right_image", None),
        )

    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Identical to PI0Pytorch.embed_prefix -- unchanged by the tactile addition."""
        embs = []
        pad_masks = []
        att_masks = []

        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)
            bsize, num_img_embs = img_emb.shape[:2]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            att_masks += [0] * num_img_embs

        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_tactile(self, tactile_left, tactile_right) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Runs the tactile ViT encoder once per camera; None if tactile is disabled/missing."""
        if not self.config.use_tactile_input or self.tactile_encoder is None:
            return None, None
        if tactile_left is None or tactile_right is None:
            return None, None

        def tactile_embed_func(tactile_left, tactile_right):
            return self.tactile_encoder(tactile_left, tactile_right)

        tokens, pad_mask = self._apply_checkpoint(tactile_embed_func, tactile_left, tactile_right)
        return tokens, pad_mask

    def embed_suffix(self, noisy_actions, timestep):
        """pi05-only version of PI0Pytorch.embed_suffix (the non-pi05 state-token branch is
        dropped -- this model always runs in pi05 mode, matching PI0Pytorch's own behavior when
        pi05=True, where the continuous `state` argument is likewise unused in the suffix).
        """
        embs = []
        pad_masks = []
        att_masks = []

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        def time_mlp_func(time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

        time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
        action_time_emb = action_emb
        adarms_cond = time_emb

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks, adarms_cond

    def _match_tactile_dtype(self, tactile_embs):
        """Casts tactile tokens to the tactile-expert branch's dtype, mirroring the prefix/suffix
        bf16 cast already done inline in forward()/sample_actions() -- HaptileTactileEncoder is a
        plain nn.Module outside paligemma_with_expert, so its output isn't auto-cast by
        FTP1PaliGemmaWithExpertModel's own to_bfloat16_for_selected_params.
        """
        if tactile_embs is None:
            return None
        branch_dtype = self.paligemma_with_expert.gemma_tactile_expert.model.layers[0].self_attn.q_proj.weight.dtype
        if tactile_embs.dtype != branch_dtype:
            tactile_embs = tactile_embs.to(dtype=branch_dtype)
        return tactile_embs

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            _state,
            tactile_left,
            tactile_right,
        ) = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, time)

        use_tactile_branch = self.config.use_tactile_input
        tactile_embs, tactile_pad_masks = self.embed_tactile(tactile_left, tactile_right)
        if use_tactile_branch and tactile_embs is None:
            raise ValueError(
                "config.use_tactile_input=True requires tactile_left_image/tactile_right_image "
                "on the observation, but embed_tactile returned None."
            )

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        if use_tactile_branch:
            tactile_embs = self._match_tactile_dtype(tactile_embs)

        layout = build_expert_attention_layout(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
            tactile_pad_masks=tactile_pad_masks if use_tactile_branch else None,
        )
        att_2d_masks_4d = self._prepare_attention_masks_4d(layout.att_2d_masks)

        def forward_func(prefix_embs, tactile_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            if use_tactile_branch:
                outputs, _ = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs, tactile_embs, suffix_embs],
                    use_cache=False,
                    adarms_cond=[None, None, adarms_cond],
                )
                return outputs[2]
            outputs, _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return outputs[1]

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, tactile_embs, suffix_embs, att_2d_masks_4d, layout.position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors).

        Replicates FTP1Pytorch.sample_actions's two-stage KV-cache build (ftp1-policy's
        ftp1_pytorch.py): the VLM prefix and the tactile tokens are each forwarded once, through
        two separate single-branch calls into FTP1PaliGemmaWithExpertModel, producing two
        per-layer K/V caches that are concatenated along the sequence dimension into one
        DynamicCache -- reused unchanged across every Euler denoising step below, so tactile
        tokens are computed exactly once per action-chunk prediction regardless of num_steps.
        """
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, _state, tactile_left, tactile_right = self._preprocess_observation(
            observation, train=False
        )

        use_tactile_branch = self.config.use_tactile_input
        tactile_embs, tactile_pad_masks = self.embed_tactile(tactile_left, tactile_right)
        if use_tactile_branch and tactile_embs is None:
            raise ValueError(
                "config.use_tactile_input=True requires tactile_left_image/tactile_right_image "
                "on the observation, but embed_tactile returned None."
            )
        if use_tactile_branch:
            tactile_embs = self._match_tactile_dtype(tactile_embs)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_layout = build_prefix_attention_layout(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_layout.att_2d_masks)

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001
        if use_tactile_branch:
            self.paligemma_with_expert.gemma_tactile_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        if use_tactile_branch:
            tactile_layout = build_tactile_attention_layout(
                tactile_pad_masks,
                position_offset=prefix_layout.position_ids[:, -1:] + 1,
            )
            tactile_att_2d_masks_4d = self._prepare_attention_masks_4d(tactile_layout.att_2d_masks)

            # Stage 1: cache the VLM prefix tokens.
            _, vlm_past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_layout.position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
            )
            # Stage 2: cache the tactile tokens (computed exactly once here, not per denoise step).
            _, tactile_past_key_values = self.paligemma_with_expert.forward(
                attention_mask=tactile_att_2d_masks_4d,
                position_ids=tactile_layout.position_ids,
                past_key_values=None,
                inputs_embeds=[None, tactile_embs, None],
                use_cache=True,
            )
            # Merge both per-layer K/V caches into one DynamicCache reused for every Euler step.
            past_key_values = DynamicCache()
            num_layers = len(vlm_past_key_values)
            for layer_idx in range(num_layers):
                vlm_kv = vlm_past_key_values[layer_idx]
                tactile_kv = tactile_past_key_values[layer_idx]
                past_key_values.update(
                    torch.cat([vlm_kv[0], tactile_kv[0]], dim=2),
                    torch.cat([vlm_kv[1], tactile_kv[1]], dim=2),
                    layer_idx,
                )
        else:
            _, past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_layout.position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                tactile_pad_masks if use_tactile_branch else None,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(self, prefix_pad_masks, tactile_pad_masks, past_key_values, x_t, timestep):
        """Apply one denoising step of the noise `x_t` at a given timestep, reusing the cache
        built once in sample_actions (tactile tokens are never recomputed here)."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep)

        use_tactile_branch = tactile_pad_masks is not None
        layout = build_action_denoise_layout(
            prefix_pad_masks,
            suffix_pad_masks,
            suffix_att_masks,
            tactile_pad_masks=tactile_pad_masks,
        )
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(layout.att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        if use_tactile_branch:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=layout.position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None, adarms_cond],
            )
            suffix_out = outputs_embeds[2]
        else:
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_masks_4d,
                position_ids=layout.position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1]

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
