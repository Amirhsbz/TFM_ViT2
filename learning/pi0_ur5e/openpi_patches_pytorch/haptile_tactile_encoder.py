"""Minimal, non-heterogeneous tactile ViT encoder for Haptile's tactile-expert model.

Phase-1 port of FTP1's tactile encoding strategy (ftp1-policy's t3_tactile_encoder.py +
ftp1_blocks.py's FTP1HptTactileEncoder), scoped down to exactly Haptile's two fixed,
always-present, same-type tactile cameras -- no heterogeneous-sensor registry, no 24-slot
function-area system, no shared "Stage 2" trunk.

This file is authored in the tele-amir repo and installed into
$OPENPI_ROOT/src/openpi/models_pytorch/ by scripts/install_openpi_pytorch_patch.py.
"""

from __future__ import annotations

import torch
from torch import nn

# Relocated verbatim from ftp1-policy's src/openpi/models_pytorch/ftp1_model_config.py (that
# file is the full FTP1ModelConfig, out of scope for this port -- only these two constants are
# needed, by t3_tactile_encoder.ViTEncoder._load_t3_pretrained_checkpoint).
T3_PRETRAINED_SENSOR_NAME_MAP = {
    "densetact": "densetact",
    "digit": "digit",
    "finray": "finray",
    "gs_black": "gs_black",
    "gs_tag": "gs_tag",
    "mini": "mini",
    "svelte": "svelte",
    "wedge": "wedge",
    "GelSightMini": "mini",
    "GelSightWedge": "wedge",
    "GelSightSvelte": "svelte",
    "GelSightFinray": "finray",
    "DIGIT": "digit",
    "DenseTact2": "densetact",
}


# "t3_large" is NOT the right size class despite the name suggesting it should match "T3-large" --
# verified by downloading and inspecting actual checkpoint tensor shapes: t3_large is
# embed_dim=1024/depth=6, which doesn't match this encoder's embed_dim=768/depth=3 at all (a
# hard shape-mismatch RuntimeError on load, not a silent issue). Checked all four published size
# classes (t3_tiny=192/3, t3_small=384/3, t3_medium=768/3, t3_large=1024/6) directly against
# downloaded checkpoints -- t3_medium is the one that actually matches this encoder's dimensions.
T3_PRETRAINED_TACTILE_ENCODER_CHECKPOINTS_BASE_URL = (
    "https://huggingface.co/datasets/alanz-mit/FoundationTactile/resolve/main/models/t3_medium/encoders/"
)

# Side tags for the learned left/right identity embedding.
_LEFT = 0
_RIGHT = 1


def _to_channels_first(image: torch.Tensor) -> torch.Tensor:
    """Normalize a tactile image tensor to (B, C, H, W), accepting either layout.

    Observation.tactile_left_image/tactile_right_image are stored channels-last ("*b h w c"),
    matching the rest of the pipeline, but PyTorch tensors arriving from different transform
    paths may already be channels-first -- mirrors the same is_channels_first detection already
    used in preprocessing_pytorch.py for the main camera images.
    """
    if image.ndim != 4:
        raise ValueError(f"Expected a 4D tactile image tensor (B,H,W,C) or (B,C,H,W), got shape {image.shape}")
    if image.shape[1] == 3 and image.shape[-1] != 3:
        return image
    return image.permute(0, 3, 1, 2).contiguous()


class HaptileTactileEncoder(nn.Module):
    """Encodes Haptile's two fixed tactile cameras into one CLS-pooled token each.

    One shared ViTEncoder (weights tied across both cameras) is run once per camera; each
    camera's CLS token is projected to `token_dim` and tagged with a learned left/right
    identity embedding so the tactile expert can tell the two tokens apart.
    """

    def __init__(
        self,
        token_dim: int,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        encoder_depth: int = 3,
        encoder_heads: int = 12,
        mlp_ratio: float = 4.0,
        sensor_name: str = "gs_tag",
        load_t3_pretrained_checkpoint: bool = False,
        cache_t3_pretrained_checkpoint_dir: str | None = None,
    ):
        super().__init__()
        # Imported lazily (not at module top level) so this file can be authored/linted in the
        # tele-amir repo before t3_tactile_encoder.py is installed alongside it in $OPENPI_ROOT.
        from openpi.models_pytorch.t3_tactile_encoder import ViTEncoder

        self.vit = ViTEncoder(
            tokenizer_name="haptile_tactile",
            sensor_name=sensor_name,
            embed_dim=embed_dim,
            num_heads=encoder_heads,
            mlp_ratio=mlp_ratio,
            depth=encoder_depth,
            img_size=(image_size, image_size),
            patch_size=patch_size,
            load_t3_pretrained_checkpoint=load_t3_pretrained_checkpoint,
            cache_t3_pretrained_checkpoint_dir=cache_t3_pretrained_checkpoint_dir,
        )
        # FTP1's unified_proj pattern (LayerNorm -> Linear -> GELU -> Linear), reference only.
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        self.side_embedding = nn.Embedding(2, token_dim)

    def _encode_one(self, image: torch.Tensor, side: int) -> torch.Tensor:
        image = _to_channels_first(image)
        features = self.vit(image)  # (B, num_patches + 1, embed_dim)
        cls_token = features[:, 0]  # (B, embed_dim)
        token = self.proj(cls_token)  # (B, token_dim)
        side_ids = torch.full((image.shape[0],), side, dtype=torch.long, device=image.device)
        return token + self.side_embedding(side_ids)

    def forward(self, tactile_left: torch.Tensor, tactile_right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tokens: (B, 2, token_dim), pad_mask: (B, 2) all-True)."""
        left_token = self._encode_one(tactile_left, _LEFT)
        right_token = self._encode_one(tactile_right, _RIGHT)
        tokens = torch.stack([left_token, right_token], dim=1)
        pad_mask = torch.ones(tokens.shape[0], 2, dtype=torch.bool, device=tokens.device)
        return tokens, pad_mask
