# Refactor Plan

A staged refactor of the project into `REFACTORED_CODEBASE/`. The original `src/`, `scripts/`, and `configs/` outside this folder are left untouched as a working backup.

The plan follows the inventory in `../REFACTOR_INVENTORY.md`. Stages are ordered so each is independently testable.

---

## Target Structure

```
REFACTORED_CODEBASE/
├── REFACTOR_PLAN.md              # this file
├── README.md                     # written in stage 8
├── ENV.md                        # copied verbatim
├── requirements.txt              # copied verbatim
├── configs/
│   └── active_matter/
│       ├── default.yaml          # base config, full schema, all defaults
│       ├── baseline.yaml         # routing override: all → all
│       ├── exp_a.yaml            # routing override: D → u
│       ├── exp_b.yaml            # routing override: divD → lapU
│       └── presets/              # regularizer presets, layered on top
│           ├── sigreg.yaml         # default λ=0.1
│           ├── sigreg_lam001.yaml
│           ├── sigreg_lam1.yaml
│           ├── vicreg.yaml         # default λ=0.1, var=25, cov=1
│           ├── vicreg_lam001.yaml
│           ├── vicreg_lam1.yaml
│           ├── vicreg_varw10.yaml
│           ├── vicreg_varw50.yaml
│           └── vicreg_covw5.yaml
├── src/
│   ├── __init__.py
│   ├── config_loader.py          # NEW — layered YAML merger (default + routing + preset)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── channel_map.py        # copy verbatim
│   │   ├── derived_fields.py     # copy verbatim
│   │   └── well_dataset.py       # TRIMMED to WellDatasetForJEPA only (~280 lines)
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── _ddp.py               # NEW — extracted DDP helpers
│   │   ├── djepa_loss.py         # patched to use _ddp
│   │   ├── sigreg.py             # patched to use _ddp
│   │   └── vicreg.py             # patched to use _ddp
│   ├── masks/
│   │   ├── __init__.py
│   │   └── utils.py              # only apply_masks; rest of masks/ dropped
│   ├── models/
│   │   ├── __init__.py
│   │   ├── dual_patch_encoder.py
│   │   ├── encoder.py
│   │   ├── modules.py
│   │   ├── patch_embed.py
│   │   ├── pos_embs.py
│   │   ├── simple_predictor.py
│   │   ├── tensors.py
│   │   └── vit_encoder.py
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── extract_features.py   # patched to import shared builders from src.train.builders
│   │   ├── knn_regression.py
│   │   ├── linear_probe.py
│   │   └── normalize_labels.py
│   └── train/
│       ├── __init__.py
│       ├── builders.py           # NEW — build_from_config, build_loader, channel routing helpers
│       ├── checkpoint.py
│       ├── djepa_optim.py
│       ├── manifest.py
│       ├── schedulers.py
│       ├── step.py               # NEW — StepMetrics, train_one_step
│       └── trainer.py            # NEW — train() loop only; previous djepa_trainer.py shell
├── scripts/
│   ├── train.py                  # was scripts/active_matter/train_djepa.py
│   ├── eval.py                   # was scripts/active_matter/run_eval.py
│   └── analyze.py                # was scripts/active_matter/analyze_representations.py
└── tests/                        # smoke tests retained as a tests dir
    ├── test_dataset.py
    ├── test_encoder.py
    ├── test_predictor.py
    ├── test_dual_encoder.py
    ├── test_trainer.py
    ├── test_resume.py
    ├── test_sigreg.py
    ├── test_vicreg.py
    ├── overfit_one_batch.py
    └── validation/
        ├── sanity_check_derived_fields.py
        └── stokes_correlation_analysis.py
```

**Files dropped entirely** (relative to original `src/`):
- `data/data_utils.py`
- `masks/default.py`, `masks/multiblock3d.py`, `masks/random_tube.py`
- `models/attentive_pooler.py`, `models/attentive_pooler_modules.py`, `models/init_model.py`, `models/multimask.py`, `models/physics_model_shim.py`, `models/physics_tensors.py`, `models/predictor.py`
- `train/hydra_compose.py`, `train/misc.py`, `train/model_summary.py`, `train/model_utils.py`, `train/train_utils.py`, `train/trainer.py`, `train/trainer_base.py`
- All 30 old `configs/active_matter/*.yaml` (replaced by the layered scheme).

That's 17 dead files + 30 redundant configs = **47 files deleted, ~4,700 lines gone**.

`model_utils.py` (the dead ConvNet stack) is **not migrated** for now. When the team merge happens we'll vendor a fresh CNN backbone from the teammate's repo or rewrite to match — flagged as an explicit follow-up rather than carried as dead code.

---

## Stages

### Stage 1 — Skeleton + verbatim copies
Mechanical: copy the 22 files that are KEEP-without-modification into the new tree. No content changes.

Files copied:
```
src/data/channel_map.py
src/data/derived_fields.py
src/eval/knn_regression.py
src/eval/linear_probe.py
src/eval/normalize_labels.py
src/losses/djepa_loss.py
src/losses/sigreg.py
src/losses/vicreg.py
src/masks/utils.py
src/models/dual_patch_encoder.py
src/models/encoder.py
src/models/modules.py
src/models/patch_embed.py
src/models/pos_embs.py
src/models/simple_predictor.py
src/models/tensors.py
src/models/vit_encoder.py
src/train/checkpoint.py
src/train/djepa_optim.py
src/train/manifest.py
src/train/schedulers.py
ENV.md
requirements.txt
```

Plus empty `__init__.py` markers under each package.

After Stage 1, the new tree is incomplete — `src/data/well_dataset.py`, the trainer, the eval entrypoint, and configs are not yet present, so it's not runnable.

### Stage 2 — Trim `well_dataset.py`
Read the original 848 lines. Write back only:
- File header + imports.
- `WellDatasetForJEPA` class (lines 23–~300).
- Drop everything else (`EmbeddingsDataset`, `DISCOLatentDataset`, `WellDatasetForMPP`, hydra-style loaders).

Expected output: ~280 lines.

### Stage 3 — Consolidate DDP helpers
Create `src/losses/_ddp.py` containing the three helpers `_is_ddp_active`, `_all_reduce_avg`, `_world_size`. Patch `sigreg.py` and `vicreg.py` to import them from there instead of defining them inline.

### Stage 4 — Split `djepa_trainer.py`
The 550-line file becomes three:

- `src/train/builders.py` (~250 lines): `_select_channels`, `_channels_for`, `_encoder_forward`, `build_from_config`, `build_loader`, `load_yaml_config`. **Public**, importable by both training and eval. This kills the awkward private-helper imports `extract_features.py` currently makes.
- `src/train/step.py` (~120 lines): `StepMetrics`, `_now_ms`, `train_one_step`.
- `src/train/trainer.py` (~200 lines): `train()` function only.

### Stage 5 — Patch `extract_features.py` to use shared builders
Replace its private `_build_encoder_from_config` with `from src.train.builders import build_from_config` (or a model-only sibling). Drop the import of `_channels_for`, `_encoder_forward`, `_select_channels` from `djepa_trainer.py` — they live in `builders.py` now and are public.

### Stage 6 — Collapse configs into layered scheme
Write `src/config_loader.py` — a small utility (~60 lines) that takes a CLI like:
```
python scripts/train.py --routing exp_a --reg vicreg
python scripts/train.py --routing exp_b --reg vicreg_lam001
python scripts/train.py --routing baseline --reg sigreg
```
and produces a merged dict by deep-merging:
```
configs/active_matter/default.yaml
+ configs/active_matter/<routing>.yaml
+ configs/active_matter/presets/<reg>.yaml
+ any --override key=value flags
```

Write the 13 YAMLs (1 default + 3 routings + 9 presets). Each routing file is ~5 lines; each preset is ~5–10 lines. Default carries all the heavy schema (model, optim, log, full data block).

### Stage 7 — Rewrite top-level scripts
- `scripts/train.py` — uses the new config_loader; preserves the existing run_dir / manifest behavior.
- `scripts/eval.py` — same, just for run_eval.
- `scripts/analyze.py` — copy of `analyze_representations.py` with path constants updated.

### Stage 8 — README + tests dir
- `README.md` — quick-start, target audience is teammate or grader. Includes the dual-backbone note.
- Move smoke tests into `tests/`, drop unmigrated ones, drop the `_add_origin_headers.py` / `_rewrite_imports.py` one-offs.

### Stage 9 — Smoke verification
After all stages: from `REFACTORED_CODEBASE/` run
```
python scripts/eval.py --run-dir ../runs/baseline_v0_20260421_152635
```
to confirm the eval pipeline reproduces the cached results within ±1%. Run `tests/test_resume.py` to confirm a save/load roundtrip works.

---

## Open Decisions Deferred to the Team Merge

1. **CNN backbone**: not vendored from `model_utils.py`. When the team merge happens we either pull from the teammate's repo or rewrite. The refactored code structure leaves room — `build_from_config` switches on a `model.backbone` config field that currently only accepts `"vit"`. Adding `"cnn"` later is a localized change.

2. **EMA target encoder**: not yet implemented in our code (`DJepaLoss` does not maintain a momentum target). Adding it is a Stage-4-adjacent change but I'm leaving it out of this refactor — the merge with the teammate's EMA logic deserves its own pass.

3. **Sample/crop scheme**: this refactor preserves your current 256-pixel native-resolution windowing. The decision about whether to switch to PDF-prescribed 224-pixel cropping is a config knob and can be flipped after the merge.

---

## Risk and Rollback

The original `src/`, `scripts/`, `configs/` are not touched. If anything in `REFACTORED_CODEBASE/` breaks, the working backup is one directory up.

`runs/` is shared between both versions — `REFACTORED_CODEBASE/scripts/eval.py --run-dir ../runs/<id>/` reads from your existing trained checkpoints, so eval can be tested without retraining anything.
