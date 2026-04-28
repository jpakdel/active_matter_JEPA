"""Dual-patch-embed encoder for cross-channel D-JEPA experiments.

Why this exists
---------------
The baseline uses a *shared* encoder for both the context and target branches:
same weights, same `in_chans`. That works because the baseline runs with
``context_channels == target_channels`` (e.g. both ``"all"``, or both ``"D"``).

For the cross-channel experiments in project_plan §6:

  * Experiment A: context = D (4ch), target = u (2ch)
  * Experiment B: context = ∇·D (2ch), target = Δu (2ch)

the two branches have different semantics (and sometimes different channel
counts). The first layer of a V-JEPA encoder is a ``PatchEmbed3D`` whose Conv3d
bakes the channel count into its weight tensor — a single encoder physically
cannot accept both tensor shapes.

The fix is the standard dual-patch-embed / shared-trunk pattern: give each
branch its own tiny projection into token space, and share everything
downstream (pos_embed, transformer blocks, final LayerNorm). This keeps the
project's "one transformer processes physics" story intact — only the
tokenizer is branch-specific.

Contract
--------
``DualPatchEncoder(x, branch="ctx" | "tgt")`` returns the same ``(B, N, D)``
tensor that a single ``VisionTransformer`` would. ``embed_dim`` is exposed as
a property so downstream code (predictor, loss) can stay agnostic.

Parameter sharing
-----------------
  * ``ctx_encoder``: a full ``VisionTransformer``. Owns pos_embed, blocks,
    norm — the shared trunk.
  * ``tgt_patch_embed``: a separate ``PatchEmbed3D`` used only when
    ``branch="tgt"``. Same embed_dim, same (patch, tubelet), different
    in_chans.

Why not swap ``self.patch_embed`` at runtime? Because ``state_dict()`` /
``load_state_dict()`` track module identity by name. Swapping modules in
``forward`` breaks checkpoint round-trips. Owning both modules explicitly is
safer and costs ~10 lines of duplicated V-JEPA forward.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.encoder import build_encoder
from src.models.patch_embed import PatchEmbed3D
from src.models.tensors import trunc_normal_


class DualPatchEncoder(nn.Module):
    """Shared-trunk ViT with per-branch patch embeds.

    Parameters
    ----------
    ctx_in_chans, tgt_in_chans : int
        Channel count for the context and target branches, respectively.
    All other kwargs are forwarded to ``build_encoder`` and must match between
    the two branches (they share the trunk).
    """

    def __init__(
        self,
        *,
        ctx_in_chans: int,
        tgt_in_chans: int,
        size: str,
        img_size: int,
        num_frames: int,
        patch_size: int,
        tubelet_size: int,
        mlp_ratio: float,
        drop_rate: float,
        attn_drop_rate: float,
        uniform_power: bool,
    ):
        super().__init__()
        # The "context encoder" owns the shared trunk (pos_embed, blocks,
        # norm). We re-route the patch-embed step in forward() so there's no
        # need to detach/swap its .patch_embed submodule.
        self.ctx_encoder = build_encoder(
            in_chans=ctx_in_chans,
            size=size,
            img_size=img_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            uniform_power=uniform_power,
        )
        # Target-branch patch embed. Same (patch, tubelet, embed_dim) as the
        # ctx branch so token count and dim line up; only in_chans differs.
        self.tgt_patch_embed = PatchEmbed3D(
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            in_chans=tgt_in_chans,
            embed_dim=self.ctx_encoder.embed_dim,
        )
        # Match V-JEPA's PatchEmbed3D init scheme (trunc_normal_ on weight,
        # zeros on bias). ``init_std`` is a float attribute set on the vendored
        # VisionTransformer at the end of __init__.
        trunc_normal_(self.tgt_patch_embed.proj.weight, std=self.ctx_encoder.init_std)
        if self.tgt_patch_embed.proj.bias is not None:
            nn.init.constant_(self.tgt_patch_embed.proj.bias, 0.0)

    # --- expose shared-trunk dims so downstream (predictor, loss) doesn't care
    @property
    def embed_dim(self) -> int:
        return self.ctx_encoder.embed_dim

    @property
    def num_heads(self) -> int:
        return self.ctx_encoder.num_heads

    # --- forward -------------------------------------------------------------
    def forward(self, x: torch.Tensor, branch: str = "ctx") -> torch.Tensor:
        """Tokenize with the branch-specific patch embed, then run the shared trunk.

        This duplicates the tokenize/pos-embed/blocks/norm sequence from
        ``VisionTransformer.forward`` because the vendored forward bakes in
        ``self.patch_embed``. Keeping the duplication local means we don't have
        to edit the vendored file.
        """
        if branch not in ("ctx", "tgt"):
            raise ValueError(f"branch must be 'ctx' or 'tgt', got {branch!r}")
        vit = self.ctx_encoder

        # 1. Interpolate pos_embed to match the input grid (same for both branches
        #    because they share shape after tokenization).
        pos_embed = vit.pos_embed
        if pos_embed is not None:
            pos_embed = vit.interpolate_pos_encoding(x, pos_embed)

        # 2. Branch-specific tokenization.
        if branch == "ctx":
            tokens = vit.patch_embed(x)
        else:  # "tgt"
            tokens = self.tgt_patch_embed(x)

        # 3. Shared pos_embed, transformer blocks, final norm. We don't
        #    forward a mask — D-JEPA does masking at the loss level by
        #    channel slicing, not per-token.
        if pos_embed is not None:
            tokens = tokens + pos_embed
        for blk in vit.blocks:
            tokens = blk(tokens, mask=None)
        if vit.norm is not None:
            tokens = vit.norm(tokens)
        return tokens
