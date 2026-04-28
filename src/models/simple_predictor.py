"""Context-to-target predictor for D-JEPA, transplanted from V-JEPA.

V-JEPA's original predictor (src/models/predictor.py) expects token-level
`masks_ctxt` / `masks_tgt` and uses learned mask tokens as target queries with
positions scattered across the patch grid. We use a coarse context/target
split from the physics repo instead: both sides are dense token grids at the
*same* space-time positions but over different *channel* subsets. That makes
the mask-token machinery dead weight. This module keeps just the useful core:

    Linear(D_enc -> D_pred)
      -> (add 3D sincos position embedding)
      -> N transformer blocks
      -> LayerNorm
      -> Linear(D_pred -> D_enc)

Input  : context tokens from the encoder, shape (B, N, D_enc).
Output : predicted target tokens at the same N positions, shape (B, N, D_enc).

The predictor is deliberately shallower and narrower than the encoder, as in
V-JEPA (predictor_embed_dim < embed_dim, depth ~ 6).
"""

from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn

from src.models.modules import Block
from src.models.pos_embs import get_3d_sincos_pos_embed
from src.models.tensors import trunc_normal_


class SimplePredictor(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 256,
        patch_size: int = 16,
        num_frames: int = 16,
        tubelet_size: int = 2,
        embed_dim: int,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
    ):
        super().__init__()
        self.input_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.uniform_power = uniform_power

        grid_size = img_size // patch_size
        grid_depth = num_frames // tubelet_size
        num_patches = grid_depth * grid_size * grid_size
        self.num_patches = num_patches

        # Project encoder-dim tokens into predictor-dim tokens.
        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)

        # Sinusoidal 3D position embedding (identical grid to the encoder).
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False,
        )
        self._init_pos_embed(self.predictor_pos_embed.data)

        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=nn.GELU,
                norm_layer=norm_layer,
                grid_size=grid_size,
                grid_depth=grid_depth,
            )
            for _ in range(depth)
        ])

        self.predictor_norm = norm_layer(predictor_embed_dim)
        # Project predictor-dim tokens back to encoder-dim so the loss can
        # compare against the target encoder's output.
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()

    def _init_pos_embed(self, pos_embed: torch.Tensor) -> None:
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        sincos = get_3d_sincos_pos_embed(
            embed_dim,
            grid_size,
            grid_depth,
            cls_token=False,
            uniform_power=self.uniform_power,
        )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self) -> None:
        for layer_id, blk in enumerate(self.predictor_blocks, start=1):
            blk.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
            blk.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))

    def forward(self, ctxt: torch.Tensor) -> torch.Tensor:
        """Predict target-token representations from context-token representations.

        Args:
            ctxt: (B, N, D_enc). We assume N == self.num_patches.

        Returns:
            (B, N, D_enc), predicted target tokens at the same positions.
        """
        B, N, _ = ctxt.shape
        assert N == self.num_patches, (
            f"predictor expects {self.num_patches} tokens, got {N}"
        )

        x = self.predictor_embed(ctxt)
        x = x + self.predictor_pos_embed

        for blk in self.predictor_blocks:
            x = blk(x)

        x = self.predictor_norm(x)
        x = self.predictor_proj(x)
        return x


def build_predictor(
    *,
    embed_dim: int,
    img_size: int = 256,
    num_frames: int = 16,
    patch_size: int = 16,
    tubelet_size: int = 2,
    predictor_embed_dim: int = 384,
    depth: int = 6,
    num_heads: int = 6,
    mlp_ratio: float = 4.0,
) -> SimplePredictor:
    """Default predictor: shallower (depth=6) and narrower (384-d) than the encoder."""
    return SimplePredictor(
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=embed_dim,
        predictor_embed_dim=predictor_embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
