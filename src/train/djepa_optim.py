"""AdamW + param-group builder + LR/WD schedules for D-JEPA.

Follows V-JEPA's init_opt pattern (src/utils/schedulers.py + src/utils/optim.py):
  - Two param groups: bias/norm params with no weight decay, everything else with WD.
  - LR: linear warmup from start_lr -> lr, then cosine decay to final_lr.
  - WD: cosine ramp from weight_decay -> final_weight_decay.

Wraps the vendored V-JEPA schedulers in src/train/schedulers.py into one call
that steps both per optimizer.step().
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.train.schedulers import WarmupCosineSchedule, CosineWDSchedule


def _is_bias_or_norm(name: str, param: torch.Tensor) -> bool:
    """Classify a parameter as belonging to the 'no-WD' group.

    Matches V-JEPA's convention: exclude all 1D params (biases, LayerNorm scale)
    from weight decay. Identified by ndim == 1 or name ending in 'bias'.
    """
    return param.ndim == 1 or name.endswith(".bias") or name.endswith("bias")


def build_param_groups(
    modules: Iterable[nn.Module],
    *,
    wd_exclude_bias_and_norm: bool = True,
) -> list[dict]:
    """Collect trainable params from `modules` into two WD groups."""
    decay, no_decay = [], []
    for m in modules:
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if wd_exclude_bias_and_norm and _is_bias_or_norm(name, p):
                no_decay.append(p)
            else:
                decay.append(p)
    groups = [
        {"params": decay, "WD_exclude": False},
        {"params": no_decay, "WD_exclude": True, "weight_decay": 0.0},
    ]
    return groups


def build_optimizer_and_scheds(
    modules: Iterable[nn.Module],
    *,
    total_steps: int,
    warmup_steps: int,
    lr: float,
    start_lr: float,
    final_lr: float,
    weight_decay: float,
    final_weight_decay: float,
    wd_exclude_bias_and_norm: bool = True,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1.0e-8,
):
    """Return (optimizer, lr_scheduler, wd_scheduler).

    The LR scheduler drives lr for both groups; the WD scheduler only updates
    the `WD_exclude=False` group (the vendored CosineWDSchedule checks the flag).

    Both schedulers expose a .step() that must be called once per optimizer step.
    """
    param_groups = build_param_groups(
        modules, wd_exclude_bias_and_norm=wd_exclude_bias_and_norm
    )
    opt = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )
    lr_sched = WarmupCosineSchedule(
        opt,
        warmup_steps=warmup_steps,
        start_lr=start_lr,
        ref_lr=lr,
        T_max=total_steps,
        final_lr=final_lr,
    )
    wd_sched = CosineWDSchedule(
        opt,
        ref_wd=weight_decay,
        T_max=total_steps,
        final_wd=final_weight_decay,
    )
    return opt, lr_sched, wd_sched
