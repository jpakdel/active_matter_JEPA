"""Per-step training metrics + the train_one_step function.

Split out from the old monolithic ``djepa_trainer.py`` so the loop module
(``trainer.py``) only contains the epoch/step loop, and so smoke tests can
import just this without dragging in the full launcher.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from typing import Optional

import torch

from src.losses.djepa_loss import DJepaLoss, DJepaLossOutput
from src.train.builders import select_channels, encoder_forward


@dataclasses.dataclass
class StepMetrics:
    step: int
    epoch: int
    pred_mse: float
    sigreg: float
    total: float
    lr: float
    wd: float
    fwd_ms: float
    bwd_ms: float
    step_ms: float
    data_ms: float


def now_ms() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() * 1000.0


# Back-compat alias.
_now_ms = now_ms


def train_one_step(
    batch: dict,
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    loss_fn: DJepaLoss,
    optimizer: torch.optim.Optimizer,
    lr_sched,
    wd_sched,
    scaler: Optional[torch.cuda.amp.GradScaler],
    cfg: dict,
    device: torch.device,
    target_encoder: Optional[torch.nn.Module] = None,
) -> tuple[DJepaLossOutput, dict]:
    """One optimizer step.

    If ``target_encoder`` is None, the original D-JEPA recipe is used: the
    same ``encoder`` runs both branches, with no_grad on the target side.

    If ``target_encoder`` is provided (EMA mode), the target branch runs
    through ``target_encoder`` instead, also under no_grad. The EMA *update*
    itself is the trainer's responsibility (one call after the optimizer step).
    """
    ctx_spec = cfg["data"]["context_channels"]
    tgt_spec = cfg["data"]["target_channels"]
    use_amp = cfg["optim"]["use_amp"] and device.type == "cuda"
    grad_clip = cfg["optim"]["grad_clip"]

    ctx_full = batch["context"].to(device, non_blocking=True)
    tgt_full = batch["target"].to(device, non_blocking=True)
    ctx = select_channels(ctx_full, ctx_spec)
    tgt = select_channels(tgt_full, tgt_spec)

    # Pick the module that produces z_tgt. EMA mode uses a separate frozen
    # encoder; shared-stop-grad mode uses the same `encoder` as the context
    # branch.
    tgt_module = target_encoder if target_encoder is not None else encoder

    optimizer.zero_grad(set_to_none=True)
    t_fwd0 = now_ms()
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if use_amp
        else contextlib.nullcontext()
    )
    with amp_ctx:
        z_ctx = encoder_forward(encoder, ctx, branch="ctx")
        with torch.no_grad():
            z_tgt = encoder_forward(tgt_module, tgt, branch="tgt").detach()
        z_hat = predictor(z_ctx)
        # Per project_plan §1: regularize the context-branch encoder output,
        # not the predictor output. This is what prevents encoder collapse.
        out = loss_fn(z_hat, z_tgt, z_for_reg=z_ctx)
    t_fwd1 = now_ms()

    if scaler is not None:
        scaler.scale(out.total).backward()
    else:
        out.total.backward()
    t_bwd1 = now_ms()

    if grad_clip is not None and grad_clip > 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(predictor.parameters()),
            grad_clip,
        )
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    lr = lr_sched.step()
    wd = wd_sched.step()
    t_step1 = now_ms()

    timings = {
        "fwd_ms": t_fwd1 - t_fwd0,
        "bwd_ms": t_bwd1 - t_fwd1,
        "step_ms": t_step1 - t_bwd1,
        "lr": lr,
        "wd": wd,
    }
    return out, timings
