"""EMA target-encoder updater.

BYOL/MoCo style momentum update. Given a paired ``online`` and ``target``
``nn.Module`` with identical architectures, after every optimizer step run::

    EMAUpdater(target, online, decay).step()

which performs in-place::

    p_target ← decay * p_target + (1 - decay) * p_online        for every param
    b_target ← decay * b_target + (1 - decay) * b_online        for every buffer
                                                                 (running stats)

The buffer update matters when the encoder has BatchNorm / running statistics —
without it, the target's BN stats stay frozen and drift away from the online's.
For our ViT encoders there are no BN buffers (LayerNorm is parameter-only), so
the buffer pass is a no-op there. For the CNN backbone we emit BatchNorm3d /
GroupNorm — GroupNorm has no running stats either, so again no-op. The pass is
kept for safety regardless.

Decay scheduling is supported but optional. The standard BYOL recipe ramps
``decay`` from ~0.996 toward 1.0 with a cosine schedule across training; you
can call ``set_decay`` per step to drive it externally, or just leave it at the
constant config value.
"""

from __future__ import annotations

import torch
from torch import nn


class EMAUpdater:
    def __init__(self, target: nn.Module, online: nn.Module, decay: float = 0.996):
        self.target = target
        self.online = online
        self.decay = float(decay)

    def set_decay(self, decay: float) -> None:
        self.decay = float(decay)

    @torch.no_grad()
    def step(self) -> None:
        d = self.decay
        # Parameters.
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.mul_(d).add_(p_o.data, alpha=1.0 - d)
        # Buffers (running stats etc.). Skip non-floating dtypes (e.g.
        # `num_batches_tracked` int64).
        for b_t, b_o in zip(self.target.buffers(), self.online.buffers()):
            if b_t.dtype.is_floating_point and b_o.dtype.is_floating_point:
                b_t.data.mul_(d).add_(b_o.data, alpha=1.0 - d)


def initialize_target_from_online(target: nn.Module, online: nn.Module) -> None:
    """Copy online weights+buffers into target. Run once at construction time
    so the first batch's predictor target is meaningful (not random)."""
    target.load_state_dict(online.state_dict(), strict=True)
    for p in target.parameters():
        p.requires_grad_(False)
