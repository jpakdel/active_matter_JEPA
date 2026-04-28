"""Spot-instance-safe checkpointing for D-JEPA.

Atomic save: write to `<path>.tmp` then os.replace to `<path>`. This makes the
checkpoint file appear either (a) fully written or (b) not at all — never a
truncated file. Safe under sudden kill.

Captures everything needed to resume bit-exactly (up to nondeterminism in
CUDA kernels):
  - encoder, predictor state dicts
  - optimizer state dict
  - LR/WD scheduler internal steps
  - global_step, epoch
  - torch + numpy + python RNG states
  - config dict (for provenance)
  - amp GradScaler state (if used)
"""

from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


def _rng_state_dict() -> dict:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _load_rng_state_dict(state: dict) -> None:
    # RNG state tensors MUST live on CPU as torch.ByteTensor. If the
    # checkpoint was loaded with map_location=<cuda device> (the usual case
    # for training resume), torch.load will have moved every tensor in the
    # payload to GPU — including the RNG state — which then fails the type
    # check inside torch.set_rng_state. Force-coerce back to CPU bytes here.
    cpu_rng = state["torch_cpu"]
    if cpu_rng.device.type != "cpu" or cpu_rng.dtype != torch.uint8:
        cpu_rng = cpu_rng.to(device="cpu", dtype=torch.uint8)
    torch.set_rng_state(cpu_rng)
    cuda_rngs = state.get("torch_cuda_all")
    if cuda_rngs is not None and torch.cuda.is_available():
        # get_rng_state_all returns a list of ByteTensors (one per device);
        # same coercion applies per-device.
        cuda_rngs = [
            t.to(device="cpu", dtype=torch.uint8)
            if (t.device.type != "cpu" or t.dtype != torch.uint8)
            else t
            for t in cuda_rngs
        ]
        torch.cuda.set_rng_state_all(cuda_rngs)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def save_checkpoint(
    path: str | Path,
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_sched,
    wd_sched,
    global_step: int,
    epoch: int,
    config: dict,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    extra: Optional[dict] = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    payload = {
        "encoder": encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_sched_step": float(lr_sched._step),
        "wd_sched_step": float(wd_sched._step),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "rng": _rng_state_dict(),
        "config": config,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    *,
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_sched,
    wd_sched,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: str = "cpu",
    restore_rng: bool = True,
) -> dict:
    """Load checkpoint in-place. Returns the payload dict for any extra fields."""
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    encoder.load_state_dict(payload["encoder"])
    predictor.load_state_dict(payload["predictor"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_sched._step = payload["lr_sched_step"]
    wd_sched._step = payload["wd_sched_step"]
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if restore_rng and "rng" in payload:
        _load_rng_state_dict(payload["rng"])
    return payload


def find_latest_checkpoint(run_dir: str | Path) -> Optional[Path]:
    """Return the newest `ckpt_*.pt` in `run_dir/checkpoints/`, or None."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    return ckpts[-1] if ckpts else None


def prune_old_checkpoints(run_dir: str | Path, keep_last_n: int) -> None:
    """Delete all but the most recent `keep_last_n` checkpoints."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    if not ckpt_dir.exists():
        return
    ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda p: p.stat().st_mtime)
    for old in ckpts[:-keep_last_n]:
        try:
            old.unlink()
        except OSError:
            pass


def write_run_config(run_dir: str | Path, config: dict) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "config.json"
    with open(out, "w") as f:
        json.dump(config, f, indent=2, default=str)
    return out
