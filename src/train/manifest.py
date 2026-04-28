"""TSV run tracker.

One row per run in `runs/manifest.tsv`. Columns are fixed (see COLUMNS below).
At run start: append a stub row with status=running. At run end (or on crash):
update that same row in place with final metrics and terminal status.

The file is rewritten from scratch each update — simple and safe for a local
solo workflow. If manifest.tsv doesn't exist we create it with a header.
"""

from __future__ import annotations

import csv
import datetime as _dt
from pathlib import Path
from typing import Optional

COLUMNS = [
    "run_id",
    "experiment",           # baseline | exp_a | exp_b | ...
    "context_channels",     # e.g. "all", "D", "divD", "D+phi"
    "target_channels",      # e.g. "all", "u", "lapU"
    "encoder_size",         # tiny | small | base | large
    "predictor_depth",
    "predictor_dim",
    "num_frames",
    "batch_size",
    "lr",
    "warmup_steps",
    "total_steps",
    "lambda_sigreg",
    "sigreg_num_slices",
    "seed",
    "status",               # running | done | crashed
    "wall_hours",
    "final_pred_mse",
    "final_sigreg",
    "final_total_loss",
    "val_pred_mse",
    "started_at",
    "ended_at",
    "notes",
]


def _ensure_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(COLUMNS)


def _read_all(path: Path) -> list[dict]:
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return list(r)


def _write_all(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in COLUMNS})
    tmp.replace(path)


def _row_from_config(run_id: str, cfg: dict) -> dict:
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    pred = model.get("predictor", {})
    loss = cfg.get("loss", {})
    opt = cfg.get("optim", {})
    return {
        "run_id": run_id,
        "experiment": cfg.get("experiment", ""),
        "context_channels": data.get("context_channels", ""),
        "target_channels": data.get("target_channels", ""),
        "encoder_size": model.get("encoder_size", ""),
        "predictor_depth": pred.get("depth", ""),
        "predictor_dim": pred.get("predictor_embed_dim", ""),
        "num_frames": data.get("num_frames", ""),
        "batch_size": opt.get("batch_size", ""),
        "lr": opt.get("lr", ""),
        "warmup_steps": "",          # filled after we compute from epochs
        "total_steps": "",
        "lambda_sigreg": loss.get("lambda_sigreg", ""),
        "sigreg_num_slices": loss.get("sigreg_num_slices", ""),
        "seed": cfg.get("seed", ""),
        "status": "running",
        "wall_hours": "",
        "final_pred_mse": "",
        "final_sigreg": "",
        "final_total_loss": "",
        "val_pred_mse": "",
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "ended_at": "",
        "notes": "",
    }


def append_run(
    manifest_path: str | Path,
    run_id: str,
    config: dict,
    *,
    total_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    notes: str = "",
) -> None:
    path = Path(manifest_path)
    _ensure_manifest(path)
    rows = _read_all(path)
    # Remove any existing row with this run_id (e.g. a re-run).
    rows = [r for r in rows if r.get("run_id") != run_id]
    row = _row_from_config(run_id, config)
    if total_steps is not None:
        row["total_steps"] = total_steps
    if warmup_steps is not None:
        row["warmup_steps"] = warmup_steps
    if notes:
        row["notes"] = notes
    rows.append(row)
    _write_all(path, rows)


def update_run(
    manifest_path: str | Path,
    run_id: str,
    **updates,
) -> None:
    """Patch fields on an existing row. No-op if run_id not found."""
    path = Path(manifest_path)
    if not path.exists():
        return
    rows = _read_all(path)
    for r in rows:
        if r.get("run_id") == run_id:
            for k, v in updates.items():
                if k in COLUMNS:
                    r[k] = v
            r["ended_at"] = _dt.datetime.now().isoformat(timespec="seconds")
            break
    _write_all(path, rows)
