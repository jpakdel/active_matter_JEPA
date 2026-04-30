"""CNN backbone — parametric replacement for the ViT.

Design constraints
------------------

* Output shape **must match the ViT's**: ``(B, N, D)`` where ``N`` is the
  number of tokens and ``D`` is ``embed_dim``. That way the predictor and the
  loss don't need to care which backbone produced their input.

* Token count ``N = T_out * H_out * W_out`` depends on:
    - temporal downsampling: ``T_out = num_frames // tubelet_size``
    - spatial downsampling: ``num_stages`` stride-2 stages, so
      ``H_out = img_size // 2 ** num_stages``
  At the defaults (``img_size=256``, ``num_frames=16``, ``tubelet_size=2``,
  ``num_stages=4``) this gives ``N = 8 * 16 * 16 = 2048``, identical to the
  ViT-small at ``patch_size=16``.

* Inspired by the teammate's ``ConvEncoder`` (factorized 2D spatial Conv +
  temporal-aggregation Conv1d-equivalent), but written fresh with our config
  schema. The point is shape-equivalence to the ViT, not behavioral fidelity
  to one specific upstream module.

Components
----------

* ``ConvStem3D`` — temporal-only downsample plus channel projection. Plays
  the role the ViT's PatchEmbed3D plays for time: reduce ``T`` by
  ``tubelet_size`` while leaving ``H, W`` alone, projecting input channels
  to ``base_channels``.

* ``Conv2DSpatialStage`` — stride-2 spatial downsample plus optional residual
  blocks. After the stem we treat ``T`` as a batch axis and run plain 2D
  convs per timestep; this matches the teammate's "factorized" recipe and is
  much cheaper than 3D convs of the same nominal size.

* ``ConvEncoderTrunk`` — the shared part of the encoder: stem + N spatial
  stages + projection to ``embed_dim`` + final flatten to tokens.

* ``ConvEncoder`` — single-branch encoder. Takes ``(B, in_chans, T, H, W)``,
  returns ``(B, N, D)``.

* ``DualConvEncoder`` — two stems (one per branch) feeding one shared trunk.
  Selected when the routing has different channel specs on context and
  target. Analogous to ``DualPatchEncoder`` for the ViT side.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


# ---- building blocks ---------------------------------------------------------

class ConvStem3D(nn.Module):
    """Conv3d stem: temporal stride = ``tubelet_size``, spatial stride = 1.

    Reduces ``T -> T // tubelet_size`` and projects ``in_chans -> out_chans``,
    leaving ``H, W`` untouched. Mirrors the role the ViT's PatchEmbed3D plays
    for the time axis specifically — spatial tokenization is handled by the
    downstream stride-2 stages.
    """

    def __init__(self, in_chans: int, out_chans: int, tubelet_size: int = 2):
        super().__init__()
        self.tubelet_size = int(tubelet_size)
        self.proj = nn.Conv3d(
            in_chans,
            out_chans,
            kernel_size=(tubelet_size, 3, 3),
            stride=(tubelet_size, 1, 1),
            padding=(0, 1, 1),
        )
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_chans)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)


class ResidualBlock2D(nn.Module):
    """Pre-norm residual: GroupNorm → 3x3 Conv → GELU → 3x3 Conv → +x.

    Operates on (B, C, H, W). When called inside the trunk, the temporal axis
    is folded into the batch so each timestep is processed independently.
    """

    def __init__(self, channels: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(self.drop(h))
        return x + h


class Conv2DSpatialStage(nn.Module):
    """One stride-2 spatial downsample followed by ``num_res_blocks`` residual blocks.

    Channel widths: ``in_ch -> out_ch`` (the downsample carries the channel ramp).
    """

    def __init__(self, in_ch: int, out_ch: int, num_res_blocks: int, dropout: float = 0.0):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(num_groups=min(8, out_ch), num_channels=out_ch)
        self.act = nn.GELU()
        self.blocks = nn.Sequential(
            *[ResidualBlock2D(out_ch, dropout=dropout) for _ in range(num_res_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, in_ch, H, W) — temporal axis folded into batch.
        x = self.down(x)
        x = self.act(self.norm(x))
        x = self.blocks(x)
        return x


class ConvEncoderTrunk(nn.Module):
    """The shared post-stem trunk of the CNN encoder.

    Takes ``(B, base_channels, T_out, H, W)`` from a stem, returns
    ``(B, N, embed_dim)`` token sequence.
    """

    def __init__(
        self,
        base_channels: int,
        embed_dim: int,
        num_stages: int,
        res_blocks_per_stage: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Channel ramp: base * 2^stage_index, capped at embed_dim. The final
        # stage projects to embed_dim regardless of where the ramp ended up,
        # so token features match the predictor's input dim exactly.
        widths = [base_channels]
        for s in range(num_stages):
            widths.append(min(embed_dim, base_channels * (2 ** (s + 1))))
        # Force the last width to be embed_dim, so the trailing 1x1 projection
        # can collapse channels precisely.
        widths[-1] = embed_dim

        self.stages = nn.ModuleList()
        for s in range(num_stages):
            self.stages.append(
                Conv2DSpatialStage(
                    in_ch=widths[s],
                    out_ch=widths[s + 1],
                    num_res_blocks=res_blocks_per_stage,
                    dropout=dropout,
                )
            )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W) — from the stem (T already downsampled).
        B, C, T, H, W = x.shape
        # Fold T into batch so 2D convs run per-timestep.
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        for stage in self.stages:
            x = stage(x)
        # Reshape back to (B, T, embed_dim, H_out, W_out)
        _, D, H_out, W_out = x.shape
        x = x.view(B, T, D, H_out, W_out)
        # Flatten to (B, N, D) where N = T * H_out * W_out.
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(B, T * H_out * W_out, D)
        return x


# ---- single-branch encoder ---------------------------------------------------

class ConvEncoder(nn.Module):
    """Single-branch CNN encoder. Output shape mirrors the ViT.

    Args:
        in_chans: input channel count (e.g. 11 for baseline).
        embed_dim: token feature dimension at the trunk's exit.
        base_channels: channel count after the stem.
        num_stages: number of stride-2 spatial downsample stages.
        res_blocks_per_stage: residual blocks per stage.
        tubelet_size: temporal stride at the stem.
        dropout: passed through to the residual blocks.
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int = 256,
        base_channels: int = 48,
        num_stages: int = 4,
        res_blocks_per_stage: int = 1,
        tubelet_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.stem = ConvStem3D(in_chans, base_channels, tubelet_size=tubelet_size)
        self.trunk = ConvEncoderTrunk(
            base_channels=base_channels,
            embed_dim=embed_dim,
            num_stages=num_stages,
            res_blocks_per_stage=res_blocks_per_stage,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.stem(x)
        return self.trunk(x)


# ---- dual-branch encoder -----------------------------------------------------

class DualConvEncoder(nn.Module):
    """Two stems (per channel-spec) feeding one shared trunk.

    Selected when the routing has different ``context_channels`` and
    ``target_channels``. Mirrors ``DualPatchEncoder`` on the ViT side.
    """

    def __init__(
        self,
        ctx_in_chans: int,
        tgt_in_chans: int,
        embed_dim: int = 256,
        base_channels: int = 48,
        num_stages: int = 4,
        res_blocks_per_stage: int = 1,
        tubelet_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ctx_in_chans = ctx_in_chans
        self.tgt_in_chans = tgt_in_chans
        self.embed_dim = embed_dim
        self.tubelet_size = tubelet_size
        self.ctx_stem = ConvStem3D(ctx_in_chans, base_channels, tubelet_size=tubelet_size)
        self.tgt_stem = ConvStem3D(tgt_in_chans, base_channels, tubelet_size=tubelet_size)
        self.trunk = ConvEncoderTrunk(
            base_channels=base_channels,
            embed_dim=embed_dim,
            num_stages=num_stages,
            res_blocks_per_stage=res_blocks_per_stage,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, branch: str = "ctx") -> torch.Tensor:
        if branch == "ctx":
            x = self.ctx_stem(x)
        elif branch == "tgt":
            x = self.tgt_stem(x)
        else:
            raise ValueError(f"branch must be 'ctx' or 'tgt', got {branch!r}")
        return self.trunk(x)


# ---- num-tokens helper, mirrors src.models.encoder.num_tokens ----------------

def cnn_num_tokens(
    *,
    img_size: int,
    num_frames: int,
    num_stages: int,
    tubelet_size: int,
) -> int:
    """Token count produced by ``ConvEncoder`` / ``DualConvEncoder``."""
    H_out = img_size // (2 ** num_stages)
    T_out = num_frames // tubelet_size
    return T_out * H_out * H_out
