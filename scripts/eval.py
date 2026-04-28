"""End-to-end eval runner for a completed D-JEPA training run.

Given a run directory (produced by scripts/active_matter/train_djepa.py), this
script:

  1. Finds the latest checkpoint in <run_dir>/checkpoints/ (or uses --ckpt).
  2. Extracts pooled (B, D) features for every sample in train/val/test using
     the frozen encoder, writes them to <run_dir>/features/<split>.pt.
  3. Fits a ridge-regression linear probe (alpha swept on val, evaluated on
     test).
  4. Fits a kNN regressor (k and metric swept on val, evaluated on test).
  5. Writes <run_dir>/eval_results.json with per-target MSE for α and ζ.

Features are cached — re-running with the same checkpoint is instant (skips
the encoder pass).

Usage:
    python scripts/active_matter/run_eval.py --run-dir runs/baseline_v0_20260421_152635
    python scripts/active_matter/run_eval.py --run-dir runs/... --ckpt path.pt --force
    python scripts/active_matter/run_eval.py --run-dir runs/... --batch-size 16

The resulting eval_results.json looks like:
    {
      "run_dir": "...",
      "checkpoint": "...",
      "meta": {...},               # from features/meta.json
      "linear_probe": {             # per src.eval.linear_probe.LinearProbeResult
        "target_names": ["alpha", "zeta"],
        "val_mse":  [0.43, 0.61],
        "test_mse": [0.45, 0.63],
        "best_alpha": 1.0,
        ...
      },
      "knn": {
        "target_names": ["alpha", "zeta"],
        "val_mse":  [...], "test_mse": [...],
        "best_k": 10, "best_metric": "cosine",
        ...
      }
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.eval.extract_features import extract_all_splits
from src.eval.knn_regression import fit_knn
from src.eval.linear_probe import fit_linear_probe
from src.eval.normalize_labels import fit_label_stats
from src.train.checkpoint import find_latest_checkpoint


# ---- helpers -----------------------------------------------------------------

def _require(cond: bool, msg: str):
    if not cond:
        raise SystemExit(msg)


def _load_split(feat_dir: Path, split: str):
    p = feat_dir / f"{split}.pt"
    if not p.exists():
        return None, None
    blob = torch.load(str(p), weights_only=True)
    return blob["features"].float(), blob["labels"].float()


def _print_table(title: str, target_names, val, test):
    print(f"\n=== {title} ===")
    header = f"  {'target':<8s}  {'val_mse':>10s}  {'test_mse':>10s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i, name in enumerate(target_names):
        te = f"{test[i]:>10.4f}" if test is not None else f"{'n/a':>10s}"
        print(f"  {name:<8s}  {val[i]:>10.4f}  {te}")


# ---- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="path to runs/<run_id>/ produced by train_djepa.py")
    ap.add_argument("--ckpt", default=None,
                    help="explicit checkpoint path; otherwise we pick the latest")
    ap.add_argument("--features-subdir", default="features",
                    help="where to cache feature tensors under <run_dir>/")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="eval batch size (forward only; can be larger than train)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pool", default="mean", choices=("mean",))
    ap.add_argument("--force", action="store_true",
                    help="re-extract features even if the cache exists")
    ap.add_argument("--skip-test", action="store_true",
                    help="only score on val (useful while iterating)")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    run_dir = (PROJECT_ROOT / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    _require(run_dir.exists(), f"run_dir does not exist: {run_dir}")

    cfg_path = run_dir / "config.json"
    _require(cfg_path.exists(), f"missing config.json in {run_dir}")

    if args.ckpt is not None:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_latest_checkpoint(run_dir)
    _require(ckpt_path is not None and ckpt_path.exists(),
             f"no checkpoint found in {run_dir}/checkpoints/")

    feat_dir = run_dir / args.features_subdir
    meta_path = feat_dir / "meta.json"
    have_cache = meta_path.exists() and all((feat_dir / f"{s}.pt").exists()
                                            for s in ("train", "val"))
    splits = ["train", "val"] if args.skip_test else ["train", "val", "test"]

    if args.force or not have_cache:
        print(f"[run_eval] extracting features  ckpt={ckpt_path.name}", flush=True)
        t0 = time.perf_counter()
        extract_all_splits(
            ckpt_path=ckpt_path,
            cfg_path=cfg_path,
            out_dir=feat_dir,
            splits=splits,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pool=args.pool,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_amp=not args.no_amp,
            project_root=PROJECT_ROOT,
        )
        print(f"[run_eval] extraction done in {time.perf_counter() - t0:.1f}s", flush=True)
    else:
        print(f"[run_eval] using cached features in {feat_dir}/  (--force to recompute)",
              flush=True)

    with open(meta_path) as f:
        meta = json.load(f)

    X_tr, y_tr = _load_split(feat_dir, "train")
    X_va, y_va = _load_split(feat_dir, "val")
    X_te, y_te = _load_split(feat_dir, "test")
    _require(X_tr is not None and X_va is not None, "train and val features are required")

    target_names = ("alpha", "zeta")
    print(f"\n[run_eval] shapes: X_tr={tuple(X_tr.shape)}  X_va={tuple(X_va.shape)}  "
          f"X_te={tuple(X_te.shape) if X_te is not None else 'n/a'}", flush=True)

    # Share train-split stats across both probes so their numbers are directly
    # comparable.
    stats = fit_label_stats(y_tr)

    print("\n[run_eval] fitting linear probe (ridge sweep)...", flush=True)
    lp = fit_linear_probe(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        target_names=target_names, stats=stats, verbose=args.verbose,
    )

    print("\n[run_eval] fitting kNN (k, metric sweep)...", flush=True)
    knn = fit_knn(
        X_tr, y_tr, X_va, y_va, X_te, y_te,
        target_names=target_names, stats=stats, verbose=args.verbose,
    )

    _print_table(
        f"linear probe  (alpha={lp.best_alpha:g})",
        lp.target_names, lp.val_mse, lp.test_mse,
    )
    _print_table(
        f"kNN  (k={knn.best_k}, metric={knn.best_metric})",
        knn.target_names, knn.val_mse, knn.test_mse,
    )

    out = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "meta": meta,
        "linear_probe": lp.to_dict(),
        "knn": knn.to_dict(),
    }
    out_path = run_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[run_eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
