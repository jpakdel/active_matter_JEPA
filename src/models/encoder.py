"""Physics-aware encoder builder for active_matter.

Thin wrapper over V-JEPA's VisionTransformer that:
  - accepts arbitrary `in_chans` (we use 11 for active_matter, but Experiments
    A/B pass subsets, so this must be a config field);
  - fixes defaults for our data shape: num_frames=16, H=W=256, tubelet=2,
    patch_size=16 -> (T/2)*(H/16)*(W/16) = 8*16*16 = 2048 tokens per sample;
  - skips V-JEPA's MultiMaskWrapper since we do not do per-token masking.

We call `VisionTransformer(...)` directly rather than the `init_video_model`
builder because that function (a) doesn't forward `in_chans`, (b) forces the
MultiMaskWrapper + mask-token predictor path we don't want.

PatchEmbed3D (src/models/patch_embed.py) already accepts `in_chans` via its
Conv3d, so the "physics-aware patch embed" is nothing more than instantiating
it with the right channel count. We expose that wiring here.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.vit_encoder import VisionTransformer, VIT_EMBED_DIMS


# Active matter defaults. Change these in one place if the dataset convention
# changes (e.g. if we ever stride frames differently).
DEFAULT_IMG_SIZE = 256
DEFAULT_NUM_FRAMES = 16
DEFAULT_PATCH_SIZE = 16
DEFAULT_TUBELET_SIZE = 2


_SIZE_PRESETS = {
    "tiny":  dict(embed_dim=192,  depth=12, num_heads=3),
    "small": dict(embed_dim=384,  depth=12, num_heads=6),
    "base":  dict(embed_dim=768,  depth=12, num_heads=12),
    "large": dict(embed_dim=1024, depth=24, num_heads=16),
}


def build_encoder(
    in_chans: int,
    *,
    size: str = "small",
    img_size: int = DEFAULT_IMG_SIZE,
    num_frames: int = DEFAULT_NUM_FRAMES,
    patch_size: int = DEFAULT_PATCH_SIZE,
    tubelet_size: int = DEFAULT_TUBELET_SIZE,
    mlp_ratio: float = 4.0,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    uniform_power: bool = False,
) -> VisionTransformer:
    """Construct a V-JEPA VisionTransformer configured for active_matter.

    Returns a module that maps (B, C, T, H, W) -> (B, N, D) where
        N = (T // tubelet_size) * (H // patch_size) * (W // patch_size)
        D = embed_dim for the chosen size.
    """
    if size not in _SIZE_PRESETS:
        raise ValueError(f"unknown size '{size}', expected one of {list(_SIZE_PRESETS)}")
    preset = _SIZE_PRESETS[size]
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        in_chans=in_chans,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        norm_layer=nn.LayerNorm,
        uniform_power=uniform_power,
        **preset,
    )


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def num_tokens(
    *,
    img_size: int = DEFAULT_IMG_SIZE,
    num_frames: int = DEFAULT_NUM_FRAMES,
    patch_size: int = DEFAULT_PATCH_SIZE,
    tubelet_size: int = DEFAULT_TUBELET_SIZE,
) -> int:
    return (num_frames // tubelet_size) * (img_size // patch_size) ** 2


def embed_dim_for(size: str) -> int:
    return _SIZE_PRESETS[size]["embed_dim"]
