"""Package the champion run's encoder weights + metadata for HF upload.

Reads the resume-bundle checkpoint (`ckpt_*.pt`) and emits a clean upload dir:

    hf_upload/
      encoder.pt              # online encoder state_dict (flat)
      target_encoder_ema.pt   # EMA target encoder state_dict (flat, optional)
      config.json             # full merged training config
      eval_results.json       # the headline alpha/zeta probe numbers

Strips optimizer state, scheduler steps, RNG state, and the predictor — the
predictor is not used in the frozen-encoder downstream story and re-attaching
it requires re-training, so we don't ship it.

Usage:
    python scripts/package_for_hf.py
        [--run-dir runs/<run_id>]
        [--out hf_upload/]
        [--ckpt ckpt_0010500.pt]   # default: pick newest in checkpoints/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


CHAMPION_RUN = "baseline_vit_ema_vicreg_lam001_20260430_170646"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Run directory to package. Defaults to the project champion.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Specific ckpt_*.pt filename (default: newest by mtime).")
    p.add_argument("--out", type=Path, default=Path("hf_upload"),
                   help="Output directory.")
    p.add_argument("--include-predictor", action="store_true",
                   help="Also save the predictor head (default: skip — not used downstream).")
    args = p.parse_args()

    here = Path(__file__).resolve().parent.parent  # REFACTORED_CODEBASE/
    run_dir = args.run_dir or (here / "runs" / CHAMPION_RUN)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    ckpt_dir = run_dir / "checkpoints"
    if args.ckpt:
        ckpt_path = ckpt_dir / args.ckpt
    else:
        ckpts = sorted(ckpt_dir.glob("ckpt_*.pt"), key=lambda x: x.stat().st_mtime)
        if not ckpts:
            raise SystemExit(f"no ckpt_*.pt found in {ckpt_dir}")
        ckpt_path = ckpts[-1]
    print(f"loading checkpoint: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # --- weights ---
    enc_sd = payload["encoder"]
    enc_path = out / "encoder.pt"
    torch.save(enc_sd, enc_path)
    enc_size_mb = enc_path.stat().st_size / 1e6
    print(f"  wrote encoder.pt ({enc_size_mb:.1f} MB, {len(enc_sd)} tensors)")

    if payload.get("target_encoder") is not None:
        tgt_sd = payload["target_encoder"]
        tgt_path = out / "target_encoder_ema.pt"
        torch.save(tgt_sd, tgt_path)
        tgt_size_mb = tgt_path.stat().st_size / 1e6
        print(f"  wrote target_encoder_ema.pt ({tgt_size_mb:.1f} MB, {len(tgt_sd)} tensors)")

    if args.include_predictor and payload.get("predictor") is not None:
        pred_sd = payload["predictor"]
        pred_path = out / "predictor.pt"
        torch.save(pred_sd, pred_path)
        print(f"  wrote predictor.pt ({pred_path.stat().st_size / 1e6:.1f} MB)")

    # --- metadata ---
    for name in ("config.json", "eval_results.json", "final.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy(src, out / name)
            print(f"  copied {name}")

    # provenance — record exactly which checkpoint we packaged
    (out / "PROVENANCE.txt").write_text(
        f"Packaged from: {ckpt_path}\n"
        f"Run dir:       {run_dir}\n"
        f"Global step:   {payload.get('global_step')}\n"
        f"Epoch:         {payload.get('epoch')}\n"
        f"Predictor included: {args.include_predictor}\n"
    )
    print(f"\ndone. Upload contents of {out.resolve()} to HF.")


if __name__ == "__main__":
    main()
