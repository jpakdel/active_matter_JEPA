# D-JEPA on `active_matter`

Self-supervised representation learning on The Well's `active_matter` simulation dataset, with linear-probe and kNN evaluation against two underlying physical parameters: **α** (active dipole strength) and **ζ** (steric alignment).

The setup is JEPA-style — a context encoder produces a latent representation, a predictor maps it forward, and the prediction is matched to a target encoder's output of a sibling sample. Anti-collapse is provided by either a **variance-hinge regularizer** (VICReg-style, with optional decorrelation), a **distribution-match regularizer** (SIGReg from LeJEPA), or an **EMA target encoder** (BYOL-style), or any combination.

The project's core question: **does any architectural / loss / target-encoder configuration produce frozen features from which α and ζ are linearly recoverable?**

---

## Headline results

After 34 trained-and-evaluated runs across a 4-axis design space, the project leaders are:

| Metric | Best run | Test MSE (z-scored, lower=better) |
|---|---|---|
| **α kNN** | `baseline + cnn + ema + vicreg_no_cov` | **0.0131** |
| **α linear** | `baseline + vit + ema + vicreg_lam001` | **0.0063** |
| **ζ kNN** | `baseline + vit + ema + vicreg_lam001` | **0.102** |
| **ζ linear** | `baseline + vit + ema + vicreg_lam001` | **0.068** |

`baseline + vit + ema + vicreg_lam001` (the round-2 surrogate-driven cell — see Phase 6 in [`COMPLETED.md`](COMPLETED.md)) sweeps three of the four metrics and ties on the fourth (α kNN, where it loses to the CNN cell by 0.0016). On α linear it is **3× better** than the CNN cell (0.0063 vs 0.0195), so the CNN cell's α-kNN edge does not generalize: cell 2's *features* contain α more cleanly. CNN keeps a narrow α-kNN advantage; for any other use, the ViT recipe wins. Full table: [`results/PARETO.md`](results/PARETO.md).

For all 34 runs, every metric, every hyperparameter: [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md).

---

## Quick start

```bash
# Train one cell — pick one value from each axis
python scripts/train.py \
    --routing baseline \
    --backbone cnn \
    --target ema \
    --loss vicreg_no_cov

# Evaluate a finished run (linear probe + kNN against alpha/zeta)
python scripts/eval.py --run-dir runs/<run_id>

# Cross-run representation analysis (PCA, PC-correlations, residual leakage)
python scripts/analyze.py
```

### The 4 axes

| Axis | Values | Meaning |
|---|---|---|
| `--routing` | `baseline` / `exp_a` / `exp_b` | Which channels feed ctx and tgt: `baseline=all→all`, `exp_a=D→u`, `exp_b=∇·D→Δu` |
| `--backbone` | `vit` / `cnn` | ViT-small (3D tubelet patch embed, ~23M params) or 2D-spatial CNN (~3.8M params) |
| `--target` | `shared` / `ema` | Single shared encoder with stop-grad target, or BYOL-style EMA target encoder |
| `--loss` | various | Loss preset under `configs/active_matter/losses/` (see below) |

### Loss presets

10 named presets parameterizing the regularizer family + knobs:

- `sigreg`, `sigreg_lam001`, `sigreg_lam1` — SIGReg with outer scale 0.1 / 0.01 / 1.0
- `vicreg`, `vicreg_lam001`, `vicreg_lam1` — VICReg full (variance hinge + cov term) at outer scale 0.1 / 0.01 / 1.0
- `vicreg_no_cov` — VICReg with `cov_weight=0` (variance hinge only — matches teammate's recipe)
- `vicreg_varw10`, `vicreg_varw50` — VICReg with variance hinge weight 10 or 50 (default 25)
- `vicreg_covw5` — VICReg with covariance weight 5 (default 1)

### CLI overrides

Any leaf in the merged config can be overridden:

```bash
python scripts/train.py \
    --routing exp_b --backbone cnn --target ema --loss vicreg \
    --override optim.batch_size=4 \
    --override optim.num_epochs=50
```

---

## Repo layout

```
REFACTORED_CODEBASE/
├── README.md                     # this file
├── COMPLETED.md                  # full project timeline + refactor history
├── REFACTOR_PLAN.md              # design plan from the refactor pass
├── ABLATION_PLAN.md              # tier-1/tier-2 cell ordering for the queue
├── LESSONS_LEARNED.md            # bug log from the project (9 entries)
├── ENV.md                        # vendored upstream commits, dataset path
├── requirements.txt
│
├── configs/active_matter/
│   ├── default.yaml              # base; data, optim, log, schedule (the non-axis defaults)
│   ├── {baseline,exp_a,exp_b}.yaml          # 3 routing presets
│   ├── backbones/{vit,cnn}.yaml             # 2 backbone presets
│   ├── targets/{shared,ema}.yaml            # 2 target-encoder presets
│   └── losses/                              # 10 regularizer presets
│
├── src/                          # ~5,750 lines, 28 live modules
│   ├── config_loader.py          # layered YAML merger (default + 4 axes + CLI overrides)
│   ├── data/                     # WellDatasetForJEPA, derived fields (∇·D, Δu), channel map
│   ├── eval/                     # frozen-encoder feature extract → ridge + kNN
│   ├── losses/                   # DJepaLoss switch (sigreg | vicreg) + shared DDP helpers
│   ├── masks/utils.py            # apply_masks (used by ViT trunk)
│   ├── models/                   # ViT encoder, CNN encoder, dual variants, predictor, pos embs
│   └── train/                    # builders, train loop, optimizer, schedulers, EMA, checkpoint, manifest
│
├── scripts/
│   ├── train.py                  # main launcher
│   ├── eval.py                   # frozen-encoder eval (linear + kNN probe)
│   ├── analyze.py                # PCA + PC-correlation + cross-prediction residuals across runs
│   ├── recover_evals.sh          # mop-up: run eval on any run that has final.json but no eval_results.json
│   ├── run_tier1_cnn.sh          # batch launcher for the 6-cell tier-1 CNN sweep (preserved as a template)
│   └── _gen_results_docs.py      # rebuilds RUN_INVENTORY / METRICS / PARETO from live run dirs
│
├── results/                      # all metrics in one place
│   ├── README.md                 # index for this folder
│   ├── PARETO.md                 # top-5 by each of 4 metrics + Pareto frontier
│   ├── ABLATION_EFFECTS.md       # paired-comparison ablation tables for every design axis
│   ├── RUN_INVENTORY.md          # all 34 runs with all metrics + hyperparameters
│   ├── METRICS.md                # wall time, parameter counts, final losses
│   └── _runs.json                # machine-readable raw inventory
│
├── tests/                        # gitignored — local smoke tests only
│   ├── test_imports.py           # all 28 modules import cleanly
│   ├── test_config_loader.py     # layered config merge correctness
│   ├── test_build_and_shapes.py  # CPU shape contract for every (backbone × routing) cell
│   └── test_gpu_smoke.py         # GPU full-config one-step + branch routing assertion
│
└── runs/                         # per-run artifacts (checkpoints gitignored, manifest+meta tracked)
    ├── manifest.tsv              # one row per run
    └── <run_id>/
        ├── config.json           # frozen merged config
        ├── metrics.jsonl         # per-step training trajectory
        ├── final.json            # wall time, final losses, param counts
        ├── eval_results.json     # alpha/zeta probe scores
        ├── checkpoints/          # gitignored
        └── features/{train,val,test}.pt + meta.json   # gitignored except meta.json
```

---

## Reading the results

Start with [`results/PARETO.md`](results/PARETO.md) for top-5s and the joint (α, ζ) Pareto frontier.

For per-axis design questions ("does adding the cov term help α on exp_a?"), see [`results/ABLATION_EFFECTS.md`](results/ABLATION_EFFECTS.md). It enumerates every paired comparison in the dataset where exactly one axis is toggled.

For the full record (all 34 runs as one sortable table + per-routing breakouts + hyperparameters), see [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md).

For training metadata (wall time, parameter counts, compute accounting), see [`results/METRICS.md`](results/METRICS.md).

---

## Reproducibility

Every run produces:

- `runs/<run_id>/config.json` — the exact merged config used (all 4 axes resolved + every override)
- `runs/<run_id>/metrics.jsonl` — one JSON line per optimizer step (loss components + step-timing)
- `runs/<run_id>/final.json` — wall time, final losses, parameter counts, num_tokens
- `runs/<run_id>/checkpoints/` — rotating last-N checkpoints with full RNG state
- `runs/<run_id>/eval_results.json` — α / ζ test MSE for ridge and kNN
- `runs/manifest.tsv` — cross-run summary, one row per run

The PDF's reproducibility checklist is satisfied by:

- **Fixed seed.** Default `seed: 0` in `default.yaml`, recorded per run.
- **Exact data preprocessing.** All preprocessing flows through `src.data.well_dataset.WellDatasetForJEPA` and the derived-field kernels in `src.data.derived_fields`. Both deterministic.
- **Full config dumped per run.** `config.json` is the resolved merge of every preset + override. Re-running with the same `config.json` gives the same trained model up to CUDA nondeterminism.
- **Parameter counts in final.json.** Always under the 100M cap (ViT-small ~34M total; CNN ~8.6M total).
- **Compute accounting.** `wall_s` per run; aggregate per-backbone in `results/METRICS.md`.
- **Mixed precision flag.** `optim.use_amp` is in every config; ViT runs use AMP fp16, CNN runs use fp32 (NaN issues with AMP+CNN — see `LESSONS_LEARNED.md`).
- **No external data or pretrained weights.** All vendored upstream commits pinned in `ENV.md`.

---

## What's vendored from where

- **V-JEPA** (https://github.com/facebookresearch/jepa @ `51c59d518fc63c08464af6de585f78ac0c7ed4d5`) — ViT encoder + transformer blocks + 3D patch embed + sincos position embeddings + cosine LR / WD schedulers. See `src/models/vit_encoder.py`, `src/models/modules.py`, `src/models/patch_embed.py`, `src/models/pos_embs.py`, `src/train/schedulers.py`.
- **physical-representation-learning** (https://github.com/helenqu/physical-representation-learning @ `bb77f7b5b506ba793ca7e746e1d0e3c12f70c0db`) — `WellDatasetForJEPA`. See `src/data/well_dataset.py`.
- **LeJEPA** (Balestriero & LeCun, arXiv 2511.08544; `rbalestr-lab/lejepa` @ `c293d291ca87cd4fddee9d3fffe4e914c7272052`) — SIGReg implementation. See `src/losses/sigreg.py`.
- **VICReg** (Bardes, Ponce, LeCun. ICLR 2022. arXiv 2105.04906) — variance-invariance-covariance loss. Re-implemented in `src/losses/vicreg.py`.
- **BYOL** (Grill et al. NeurIPS 2020. arXiv 2006.07733) — EMA target-encoder pattern. Implemented in `src/train/ema.py`.
- **CNN backbone** — written fresh in `src/models/cnn_encoder.py`, parametrically following the structure used in the parallel project (`Tenoic/NYU-DL2026-FP`).

Per-file headers reference the upstream path and commit when applicable.

---

## Project history

For the chronological project narrative — what was done, what was tried, what worked, what failed — see [`COMPLETED.md`](COMPLETED.md). For the refactor design rationale, see [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md). For the bug ledger from the development process, see [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md).
