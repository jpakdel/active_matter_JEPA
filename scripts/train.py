"""D-JEPA training launcher (refactored, 4-axis layered configs).

Usage:
    python scripts/train.py --routing baseline --backbone vit --target shared --loss sigreg
    python scripts/train.py --routing exp_a --backbone vit --target ema --loss vicreg
    python scripts/train.py --routing exp_b --backbone cnn --target ema --loss vicreg_no_cov \\
                            --override optim.batch_size=4

Creates a per-run directory:
    runs/<run_name>_<YYYYmmdd_HHMMSS>/
        config.json          frozen copy of the merged config
        metrics.jsonl        one JSON per optimizer step
        final.json           end-of-run summary
        checkpoints/         rotating checkpoints

Also appends/updates a row in runs/manifest.tsv with final metrics and status.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_layered_config
from src.train.trainer import train


def _parse_overrides(items):
    """Convert ['a.b=1', 'c.d=true'] into {'a.b': '1', 'c.d': 'true'}."""
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--override expects key=value, got: {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routing", required=True,
                    choices=("baseline", "exp_a", "exp_b"),
                    help="data-routing axis (which channel groups feed ctx and tgt)")
    ap.add_argument("--backbone", default="vit",
                    help="backbone preset name; file stem under "
                         "configs/active_matter/backbones/")
    ap.add_argument("--target", default="shared",
                    help="target-encoder preset name; file stem under "
                         "configs/active_matter/targets/ (shared | ema)")
    ap.add_argument("--loss", required=True,
                    help="loss / regularizer preset name; file stem under "
                         "configs/active_matter/losses/")
    ap.add_argument("--override", action="append", default=[],
                    metavar="KEY=VALUE",
                    help="dotted-key override on the merged config; "
                         "repeatable. e.g. --override optim.batch_size=4")
    ap.add_argument("--run-name", default=None,
                    help="override run_name in config; timestamp is appended")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    cfg = load_layered_config(
        routing=args.routing,
        backbone=args.backbone,
        target=args.target,
        loss=args.loss,
        overrides=_parse_overrides(args.override),
    )

    base_name = args.run_name or (
        f"{cfg.get('experiment')}_{args.backbone}_{args.target}_{args.loss}"
    )
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{base_name}_{stamp}"

    out_root = PROJECT_ROOT / cfg["log"].get("out_dir", "runs")
    run_dir = out_root / run_id
    manifest_path = out_root / "manifest.tsv"

    print(f"routing:     {args.routing}")
    print(f"backbone:    {args.backbone}")
    print(f"target:      {args.target}")
    print(f"loss:        {args.loss}")
    print(f"run_id:      {run_id}")
    print(f"run_dir:     {run_dir}")
    print(f"manifest:    {manifest_path}")
    print(f"experiment:  {cfg.get('experiment')}")
    print(f"batch_size:  {cfg['optim']['batch_size']}")
    print(f"num_workers: {cfg['optim']['num_workers']}")
    print(f"num_epochs:  {cfg['optim']['num_epochs']} (override: {args.max_epochs})")
    print(f"reg_type:    {cfg['loss']['reg_type']}")
    print(f"target_type: {cfg['train']['target_type']}")

    summary = train(
        cfg,
        run_dir=run_dir,
        resume=not args.no_resume,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        manifest_path=manifest_path,
        run_id=run_id,
    )

    print("\n=== FINAL ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")


if __name__ == "__main__":
    main()
