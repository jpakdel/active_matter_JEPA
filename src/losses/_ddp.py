"""Shared DDP helpers for the loss modules.

Previously duplicated inline in `sigreg.py` and `vicreg.py`. Same semantics
in both: no-op when torch.distributed isn't initialized.
"""
from __future__ import annotations

import torch
from torch import distributed as dist


def is_ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def all_reduce_avg(x: torch.Tensor) -> torch.Tensor:
    """Average a tensor across DDP ranks. No-op when DDP not initialized.

    Uses torch.distributed.nn.functional.all_reduce so the autograd graph is
    preserved through the reduction.
    """
    if is_ddp_active():
        import torch.distributed.nn as _dnn
        return _dnn.functional.all_reduce(x, dist.ReduceOp.AVG)
    return x


def world_size() -> int:
    return dist.get_world_size() if is_ddp_active() else 1
