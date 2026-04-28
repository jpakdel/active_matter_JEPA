# D-JEPA on `active_matter` (refactored)

Self-supervised representation learning on The Well's `active_matter` dataset. JEPA-style: a context encoder predicts a target encoder's embeddings under cross-channel masking, regularized either by SIGReg (slice-wise integrated-Gaussian distribution match) or by VICReg (variance + covariance hinge).

This is a refactor of the original codebase one directory up. Same models, same losses, same eval pipeline — just split into smaller files, deduplicated, and configured through a layered YAML scheme instead of one config file per sweep cell.

---

## Quick start

```bash
# train
python scripts/train.py --routing exp_a --reg vicreg

# eval a completed run
python scripts/eval.py --run-dir runs/exp_a_vicreg_20260424_005651

# representation analysis across multiple runs
python scripts/analyze.py
```

`--routing` is one of `baseline`, `exp_a`, `exp_b` (channel-routing presets in `configs/active_matter/`). `--reg` is the file stem of any preset under `configs/active_matter/presets/` — currently `sigreg`, `sigreg_lam001`, `sigreg_lam1`, `vicreg`, `vicreg_lam001`, `vicreg_lam1`, `vicreg_varw10`, `vicreg_varw50`, `vicreg_covw5`.

CLI overrides on the merged config are dotted-key:

```bash
python scripts/train.py --routing exp_b --reg vicreg \
       --override optim.batch_size=4 --override optim.num_epochs=50
```

---

## Repo layout

```
REFACTORED_CODEBASE/
├── REFACTOR_PLAN.md              # how this tree was derived from the original
├── README.md                     # you are here
├── ENV.md                        # vendored-commit pins, dataset location
├── requirements.txt
├── configs/active_matter/
│   ├── default.yaml              # base; full schema, all defaults
│   ├── baseline.yaml             # routing: all → all
│   ├── exp_a.yaml                # routing: D → u
│   ├── exp_b.yaml                # routing: ∇·D → Δu
│   └── presets/                  # regularizer presets, layered on top of routing
│       ├── sigreg.yaml
│       ├── sigreg_lam001.yaml
│       ├── sigreg_lam1.yaml
│       ├── vicreg.yaml
│       ├── vicreg_lam001.yaml
│       ├── vicreg_lam1.yaml
│       ├── vicreg_varw10.yaml
│       ├── vicreg_varw50.yaml
│       └── vicreg_covw5.yaml
├── src/
│   ├── config_loader.py          # layered YAML merger
│   ├── data/                     # WellDatasetForJEPA + derived fields ∇·D, Δu
│   ├── eval/                     # extract_features, linear_probe, knn_regression, normalize_labels
│   ├── losses/                   # DJepaLoss switch (sigreg | vicreg) + shared DDP helpers
│   ├── masks/utils.py            # apply_masks (used by ViT)
│   ├── models/                   # ViT encoder + DualPatchEncoder + simple predictor
│   └── train/                    # builders, step, trainer, optimizer, schedulers, checkpoint, manifest
└── scripts/
    ├── train.py                  # main launcher
    ├── eval.py                   # frozen-encoder eval (linear probe + kNN)
    └── analyze.py                # PCA + PC-correlation + cross-prediction residuals
```

---

## What changed from the original repo

| Concern | Before | After |
|---|---|---|
| Trainer file size | 550 lines, 6 concerns mixed | 3 files (`builders.py`, `step.py`, `trainer.py`) |
| Encoder construction | duplicated across train + eval | shared via `src.train.builders.build_encoder_from_config` |
| DDP helpers | duplicated in `sigreg.py` and `vicreg.py` | extracted to `src.losses._ddp` |
| Configs | 30 near-identical YAMLs | 1 default + 3 routings + 9 regularizer presets |
| `well_dataset.py` | 848 lines with 4 classes | 280 lines, only `WellDatasetForJEPA` |
| Dead vendored code | 14 files (~2,400 lines) reachable only through each other | dropped |

The full inventory and rationale are in `../REFACTOR_INVENTORY.md` (one directory up); the staged execution plan is in `REFACTOR_PLAN.md` (this directory).

---

## Hooks left for the team merge

Two follow-ups deliberately out of scope for this refactor — they belong to the team-merge pass with the parallel ConvNet-JEPA implementation.

1. **CNN backbone.** The original `src/train/model_utils.py` was dead code containing a ConvNet encoder/predictor stack from the physics repo. It's not migrated. When merging with the teammate's ConvNet implementation, choose between repurposing that file or vendoring the teammate's interface, then add `model.backbone: "cnn"` as a switch alongside the current `"vit"` path in `src.train.builders.build_encoder_from_config`.

2. **EMA target encoder.** The current loss has no momentum target — anti-collapse comes from SIGReg or VICReg. The teammate's pipeline runs an EMA target (BYOL-style) plus a variance hinge. Adding EMA is a localized change in `src.train.step.train_one_step`: replace the second `encoder_forward(...).detach()` call with a forward through a separate momentum-updated `target_encoder` whose params are an EMA of the context encoder. The hook would land in `src.train.trainer.train()` where it manages the EMA update each step.

---

## Compute budget reproducibility

Every run produces:

- `runs/<run_id>/config.json`  — frozen merged config
- `runs/<run_id>/metrics.jsonl` — one JSON per optimizer step (loss components + step timings)
- `runs/<run_id>/final.json`   — wall time, final losses, parameter counts, total tokens
- `runs/<run_id>/checkpoints/`  — rotating checkpoints (default keep-last-3)
- `runs/manifest.tsv`           — one row per run, status + wall hours + final metrics

The PDF's Reproducibility Checklist items are addressed by:
- Fixed seed in `default.yaml` (`seed: 0`); recorded in every run's `config.json`.
- Layered configs are deterministic; the exact merged config is dumped as `config.json`.
- Parameter counts in `final.json` (`encoder_params`, `predictor_params`).
- Wall time + GPU mem accounting in `metrics.jsonl` and `manifest.tsv`.
- Mixed-precision flag in `config.json` (`optim.use_amp`).
- No external data, no pretrained weights — see `ENV.md`.

---

## Caveats

- This refactor was created in a sibling directory (`REFACTORED_CODEBASE/`) under the original repo root. The original `src/`, `scripts/`, `configs/` are untouched and remain a working backup.
- `scripts/analyze.py` defaults to reading runs from `<parent>/runs/` so the previously-trained checkpoints stay accessible. Override with `DJEPA_RUNS_ROOT=<path>` env var.
- The `run_eval.py` → `scripts/eval.py` rename: behavior is identical, just relocated. Old run dirs from the parent repo work unchanged when passed via `--run-dir ../runs/<run_id>`.
