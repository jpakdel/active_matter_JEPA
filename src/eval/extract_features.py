"""Frozen-encoder feature extraction for the D-JEPA eval pipeline.

Given a trained checkpoint and a config, runs the encoder over every sample in
one or more splits (train / val / test) and dumps a `(N, D)` feature tensor
plus the `(N, 2)` (alpha, zeta) label tensor to disk.

Design choices:
  - Pooling: mean over the 2048 token axis. See module docstring in
    `src/models/encoder.py` — the encoder returns `(B, N, D)` with no built-in
    pool, and our evaluation protocol (§4) uses a single pooled vector per
    sample. We expose `pool` so we can also try `"cls"`-free alternatives later
    (e.g. mean over space then over time, or last-time mean) but the default is
    plain mean.
  - No gradient, `.eval()`, autocast for speed. One fwd per sample.
  - We ignore the predictor entirely — it's only needed for training. We still
    load it into a dummy module because `load_checkpoint` expects it (this is
    cheap; it's ~10M params on CPU).
  - Labels come directly from `batch["physical_params"]` which is already in
    (alpha, zeta) order per `src/data/channel_map.py`.

Output layout:
    <features_dir>/
        train.pt    {"features": (N_train, D), "labels": (N_train, 2)}
        val.pt      ...
        test.pt     ...            # only if the split exists on disk
        meta.json   {"checkpoint", "global_step", "pool", "embed_dim",
                     "num_tokens", "encoder_size", "context_channels"}

Usage from a script:
    from src.eval.extract_features import extract_all_splits
    extract_all_splits(
        ckpt_path="runs/<run_id>/checkpoints/ckpt_XXXX.pt",
        cfg_path="runs/<run_id>/config.json",
        out_dir="runs/<run_id>/features",
        splits=("train", "val", "test"),
        batch_size=8,
        num_workers=4,
        device="cuda",
    )
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader

from src.data.well_dataset import WellDatasetForJEPA
from src.models.encoder import embed_dim_for
from src.train.builders import (
    build_encoder_from_config,
    channels_for as _channels_for,
    encoder_forward as _encoder_forward,
    select_channels as _select_channels,
)


POOLS = ("mean",)  # extend as needed (e.g. "mean_space_then_time")


# ---- config helpers ----------------------------------------------------------

def _load_config(cfg_path: str | Path) -> dict:
    """Load the frozen per-run config. We support both JSON (saved by the
    trainer) and YAML (the source config in configs/)."""
    cfg_path = Path(cfg_path)
    if cfg_path.suffix == ".json":
        with open(cfg_path) as f:
            return json.load(f)
    # Fall back to YAML.
    from src.train.builders import load_yaml_config
    return load_yaml_config(cfg_path)


def _build_encoder_from_config(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Build just the encoder (no predictor, no loss) from a saved config.

    Thin wrapper over ``src.train.builders.build_encoder_from_config`` that also
    sets eval mode and freezes parameters. The routing rule (single vs
    dual-patch-embed) is shared with the trainer — if it diverges, state_dict
    loading would fail.
    """
    enc = build_encoder_from_config(cfg, device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def _load_encoder_state(encoder: torch.nn.Module, ckpt_path: str | Path,
                        map_location: str = "cpu") -> dict:
    """Load only the encoder state_dict from a full training checkpoint.

    We deliberately avoid the full `load_checkpoint` flow because that wants
    an optimizer, schedulers, etc. — none of which are meaningful at eval time.
    """
    payload = torch.load(str(ckpt_path), map_location=map_location, weights_only=False)
    incompatible = encoder.load_state_dict(payload["encoder"], strict=True)
    # `strict=True` already raises on mismatch, but the return value lists any
    # keys that were tolerated (there shouldn't be any if strict=True succeeds).
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"encoder state_dict mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "global_step": int(payload.get("global_step", -1)),
        "epoch": int(payload.get("epoch", -1)),
        "saved_config": payload.get("config", {}),
    }


# ---- data loader -------------------------------------------------------------

def _eval_loader(cfg: dict, split: str, batch_size: int, num_workers: int,
                 project_root: Optional[Path] = None) -> Optional[DataLoader]:
    """Return a no-shuffle, no-drop_last DataLoader for one split, or None if
    the split's directory doesn't exist (e.g. test split not present locally).

    `project_root` is used to resolve `cfg["data"]["data_dir"]` when it's a
    relative path (which it is for baseline configs). Defaults to the current
    working directory, matching the training launcher's convention.
    """
    d = cfg["data"]
    data_dir = Path(d["data_dir"])
    if not data_dir.is_absolute() and project_root is not None:
        data_dir = (project_root / data_dir).resolve()
    # WellDatasetForJEPA expands split -> data_dir / "data" / split (with "val"->"valid").
    split_on_disk = "valid" if split == "val" else split
    if not (data_dir / "data" / split_on_disk).exists():
        return None
    ds = WellDatasetForJEPA(
        data_dir=str(data_dir),
        num_frames=d["num_frames"],
        split=split,
        stride=d.get("stride"),
        noise_std=0.0,        # never add noise at eval
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


# ---- pooling -----------------------------------------------------------------

def pool_tokens(tokens: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """(B, N, D) -> (B, D)."""
    if mode == "mean":
        return tokens.mean(dim=1)
    raise ValueError(f"unknown pool mode {mode!r} (supported: {POOLS})")


# ---- main extraction ---------------------------------------------------------

@torch.no_grad()
def extract_one_split(
    encoder: torch.nn.Module,
    loader: DataLoader,
    *,
    context_channels: str,
    pool: str,
    device: torch.device,
    use_amp: bool = True,
    verbose: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the encoder over a whole split. Returns (features, labels).

    features: (N, D) float32 on CPU
    labels:   (N, 2) float32 on CPU, ordered [alpha, zeta]
    """
    encoder.eval()
    feats: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    amp_on = use_amp and device.type == "cuda"

    n_total = len(loader.dataset)
    seen = 0
    t0 = time.perf_counter()
    for batch in loader:
        # We encode the *context* — which for the baseline is the first half of
        # each 32-frame trajectory window. Evaluation at the representation
        # level doesn't care about the target; what matters is "what does the
        # encoder do when handed a physical snapshot".
        ctx = batch["context"].to(device, non_blocking=True)
        ctx = _select_channels(ctx, context_channels)
        ctx_mgr = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if amp_on else contextlib.nullcontext())
        with ctx_mgr:
            # Route through _encoder_forward so we pass branch="ctx" for
            # a DualPatchEncoder; a plain VisionTransformer ignores the
            # branch arg. Eval always encodes the context branch — the
            # target patch embed is never used downstream.
            z = _encoder_forward(encoder, ctx, branch="ctx")   # (B, N_tokens, D)
        pooled = pool_tokens(z.float(), pool)   # (B, D)
        feats.append(pooled.cpu())
        labs.append(batch["physical_params"].float().cpu())

        seen += ctx.size(0)
        if verbose and (seen % (loader.batch_size * 50) == 0 or seen == n_total):
            dt = time.perf_counter() - t0
            rate = seen / dt if dt > 0 else float("nan")
            print(f"  [{seen:>6d}/{n_total:>6d}] {rate:.1f} samples/s", flush=True)

    features = torch.cat(feats, dim=0)
    labels = torch.cat(labs, dim=0)
    assert features.shape[0] == labels.shape[0] == n_total, \
        f"mismatch: feat={features.shape} lab={labels.shape} ds={n_total}"
    return features, labels


def extract_all_splits(
    *,
    ckpt_path: str | Path,
    cfg_path: str | Path,
    out_dir: str | Path,
    splits: Iterable[str] = ("train", "val", "test"),
    batch_size: int = 8,
    num_workers: int = 4,
    pool: str = "mean",
    device: str = "cuda",
    use_amp: bool = True,
    project_root: Optional[Path] = None,
) -> Path:
    """Extract features for each requested split and persist as `<split>.pt`.

    Returns the `out_dir` path. Also writes `meta.json`.
    """
    if pool not in POOLS:
        raise ValueError(f"pool={pool!r} not in {POOLS}")

    device_t = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_config(cfg_path)
    encoder = _build_encoder_from_config(cfg, device_t)
    ckpt_info = _load_encoder_state(encoder, ckpt_path, map_location=str(device_t))

    D_enc = embed_dim_for(cfg["model"]["encoder_size"])
    written: dict[str, dict] = {}
    for split in splits:
        loader = _eval_loader(cfg, split, batch_size=batch_size, num_workers=num_workers,
                              project_root=project_root)
        if loader is None:
            print(f"[extract] skipping split={split!r}: directory not present", flush=True)
            continue
        print(f"[extract] split={split}  N={len(loader.dataset)}  D={D_enc}", flush=True)
        feats, labs = extract_one_split(
            encoder, loader,
            context_channels=cfg["data"]["context_channels"],
            pool=pool, device=device_t, use_amp=use_amp,
        )
        out_path = out_dir / f"{split}.pt"
        torch.save({"features": feats, "labels": labs}, out_path)
        written[split] = {"path": str(out_path), "N": int(feats.shape[0])}

    meta = {
        "checkpoint": str(ckpt_path),
        "config_path": str(cfg_path),
        "global_step": ckpt_info["global_step"],
        "epoch": ckpt_info["epoch"],
        "pool": pool,
        "embed_dim": int(D_enc),
        "encoder_size": cfg["model"]["encoder_size"],
        "context_channels": cfg["data"]["context_channels"],
        "splits": written,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[extract] wrote meta.json to {out_dir/'meta.json'}", flush=True)
    return out_dir


