# Completed Work

Chronological narrative of what was done in this project, from first run through final results consolidation. Cross-referenced to other docs where the detail lives.

---

## Phase 1 — SIGReg-only sweep (2026-04-21 / 22)

The original D-JEPA setup: ViT-small backbone (~23M params) + simple predictor + SIGReg as the only anti-collapse mechanism. No EMA target. Three channel routings: `baseline` (all 11 channels both branches, temporal split only), `exp_a` (D → u, learns Stokes inversion), `exp_b` (∇·D → Δu, the "local-PDE preprocessing" hypothesis from §6.1 of the original plan).

5 runs total:

- `baseline_v0` — α=0.175 lin / 0.231 kNN, ζ=0.442 / 0.383
- `exp_a_v0` — α=0.214 / 0.335, ζ=0.314 / 0.398
- `exp_b_v0` — α=0.593 / 0.540, ζ=0.425 / 0.534
- `exp_b_lam001_v0` — λ=0.01, partial collapse (kNN ζ=1.46)
- `exp_b_lam1_v0` — λ=1.0, α=0.488 (best Exp B, still worst routing)

**Outcome.** The §6.1 "Exp B exposes α linearly" hypothesis was decisively refuted — Exp B was the worst of the three on α. The λ sweep showed it wasn't a tuning artifact: α stayed in [0.488, 0.593] across 100× in λ.

---

## Phase 2 — VICReg pivot (2026-04-24)

Hypothesis: the regularizer family matters. Switched SIGReg's slice-wise distribution-matching to VICReg's variance-hinge + covariance-decorrelation, kept everything else identical. 8 runs covering the three routings + 5-knob sweep on Exp B (λ ∈ {0.01, 0.1, 1.0}, var_weight ∈ {10, 25, 50}, cov_weight ∈ {1, 5}).

Headline:

- `djepa_baseline_vicreg_v0` — **collapsed**. pred_mse → 0.0015, kNN ζ=1.98 (worse than constant-mean).
- `djepa_exp_a_vicreg_v0` — α=0.087 (best α probe in the project at that time).
- `djepa_exp_b_vicreg_v0` — also collapsed. α=0.753.
- 5-knob sweep on Exp B: α stayed in [0.659, 0.757] across the entire sweep.

**Outcome.** VICReg + Exp A (D → u routing) gave the best α probe yet. But VICReg's variance hinge alone wasn't enough to prevent encoder collapse on the easy routings (baseline, where ctx == tgt is mostly identity prediction; exp_b, where ∇·D and Δu are low-magnitude derived fields).

---

## Phase 3 — Refactor (2026-04-27)

The original repo had grown ad-hoc: 30 near-identical config YAMLs (one per sweep cell), a 550-line `djepa_trainer.py` mixing 6 concerns, ~2,400 lines of dead vendored code reachable only through unused entry points, and duplicated helpers between training and eval.

A staged refactor pulled the live code into a clean tree at `REFACTORED_CODEBASE/`. Original `src/`, `scripts/`, `configs/` left intact one directory up as a working backup.

| Concern | Before | After |
|---|---|---|
| Trainer file | 550 lines, 6 mixed concerns | 3 files: `builders.py`, `step.py`, `trainer.py` |
| Encoder construction | duplicated train + eval | shared via `src.train.builders.build_encoder_from_config` |
| DDP helpers | duplicated in `sigreg.py` and `vicreg.py` | extracted to `src.losses._ddp` |
| Configs | 30 near-identical YAMLs | layered: 1 default + 3 routings + 9 regularizer presets |
| `well_dataset.py` | 848 lines, 4 dataset classes | 294 lines, only `WellDatasetForJEPA` |
| Dead vendored code | 14 files (~2,400 lines) reachable only through each other | dropped entirely |

Full inventory and rationale: [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md). The original inventory across both repos: `../REFACTOR_INVENTORY.md` (in the parent dir).

After the refactor, the layered config scheme allowed a 4-axis CLI (`--routing`, `--backbone`, `--target`, `--loss`) instead of one YAML per cell.

---

## Phase 4 — EMA target encoder + CNN backbone + ablation queue (2026-04-28 / 29)

The hypothesis from comparing my project to the parallel ConvNet-JEPA project (`Tenoic/NYU-DL2026-FP`): **EMA target encoders + CNN backbone** were both missing from this codebase, and either or both might be doing real work in the teammate's results.

Two architectural additions landed:

- **EMA target encoder** (`src/train/ema.py`) — a separate frozen encoder whose parameters are an exponential moving average of the online encoder. BYOL-style. Switchable via `--target {shared, ema}`.
- **CNN backbone** (`src/models/cnn_encoder.py`) — fresh implementation parametric in `embed_dim`, `base_channels`, `num_stages`, `res_blocks_per_stage`. Designed so its token output `(B, N=2048, D=256)` matches the ViT's, so the predictor and loss work unchanged. Switchable via `--backbone {vit, cnn}`.

Plus 13 new ablation cells run as a queue:

**ViT × EMA × {SIGReg, VICReg} × 3 routings = 6 cells** (plus one extra: `exp_b_vit_ema_vicreg_no_cov`).

Most-significant result: **`baseline_vit_ema_vicreg`** rescued the original collapsed `baseline_vicreg` from α=0.417 to **α=0.032 / ζ=0.144** — a 13× α improvement just from adding EMA. Confirms EMA is doing real anti-collapse work where the variance hinge alone failed.

**CNN × EMA × {VICReg, vicreg_no_cov} × 3 routings = 6 cells.**

Most-significant result: **`baseline_cnn_ema_vicreg_no_cov`** at α=0.0195 lin / **0.0131 kNN** — the project leader. The "no_cov" choice (variance-only VICReg, matching the teammate's exact recipe) is genuinely better than the cov-on variant for α on the baseline routing, because the cov term destroys a clean low-rank α direction when prediction is structurally easy.

The cov term turns out to be **routing-dependent**:

- baseline (easy): cov OFF is best for α (0.0131 kNN with cov off → 0.348 with cov on)
- exp_a (D → u, harder): cov ON is best (0.391 → 0.203)
- exp_b (∇·D → Δu, hard): cov ON is best (0.541 → 0.410)

When the prediction task is non-trivial and the encoder needs to spread information across many features, the decorrelation pressure helps. When prediction is easy, the encoder finds a clean low-rank α direction without any decorrelation push, and cov destroys it.

Detailed cross-axis analysis: [`results/ABLATION_EFFECTS.md`](results/ABLATION_EFFECTS.md).

---

## Phase 5 — Bloat audit + results consolidation (2026-04-29)

A second-pass cleanup of the refactored codebase. 690 lines removed across 10 files: dead V-JEPA factory functions (`vit_huge`, `vit_giant`, `vit_gigantic`), dead transformer modules (`CrossAttention`, `CrossAttentionBlock`), back-compat aliases in `builders.py`, dead helpers in `tensors.py` and `pos_embs.py`, retired one-time launcher and test scripts.

In parallel, all analytical results were consolidated under `results/`:

- [`results/README.md`](results/README.md) — index for the folder
- [`results/PARETO.md`](results/PARETO.md) — top-5 by each of 4 metrics + Pareto frontier
- [`results/ABLATION_EFFECTS.md`](results/ABLATION_EFFECTS.md) — paired comparisons for every design axis
- [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md) — all 26 runs with all metrics + hyperparameters
- [`results/METRICS.md`](results/METRICS.md) — wall time, parameter counts, final losses
- `results/_runs.json` — machine-readable raw inventory

The generator at `scripts/_gen_results_docs.py` rebuilds the four data-driven docs from live run dirs. Re-run any time new training runs land.

Final state: ~5,750 lines of source across 28 live modules; 26 trained-and-evaluated runs spanning the 4-axis design space (incomplete coverage — see "Coverage gaps" below).

---

## Coverage gaps in the design space

The full 4-axis Cartesian product is `3 routings × 2 backbones × 2 targets × ~10 losses = 120 cells`. We ran 26. The deliberate exclusions:

- **CNN × shared-encoder** (no EMA). Skipped — the teammate's parallel project showed CNN without an EMA target underperforms; no a-priori reason to expect it to help here.
- **CNN × SIGReg.** Skipped — would tell us whether LeJEPA-style anti-collapse generalizes to the CNN backbone, but lower-priority than the VICReg variants.
- **VICReg knob sweeps (varw10/varw50/covw5/lam001/lam1) × {non-exp_b routings}**. The original sweep was scoped to Exp B only because that was where the §6.1 hypothesis lived. The new EMA + non-exp_b cells use only the default `vicreg` and `vicreg_no_cov` presets.
- **Patch-size ablations on ViT.** Discussed with my teammate, deferred — patch_size=4 would give 16× more tokens at ~256× attention compute, expensive and not the main story.

These would be straightforward Tier-2 additions if more compute is available — the launcher and resume logic handle adding new cells without re-running existing ones.

---

## Compute summary

| Backbone | Encoder params | Predictor params | Wall (avg) | AMP |
|---|---|---|---|---|
| ViT-small | ~22.5M | ~10.9M | 46.9 min/run | fp16 |
| CNN | ~3.8M | ~4.9M | 46.1 min/run | fp32 |

Total compute: **20 ViT runs + 6 CNN runs ≈ 20.4 hours of GPU on a single RTX 4070 SUPER.**

Per-run record in `results/METRICS.md`.

---

## What's left

- Final report writeup. The headline finding is the architecture × parameter interaction (CNN best for α, ViT best for ζ). Supporting axes-of-variation analysis is in `results/ABLATION_EFFECTS.md`. Recommended structure: methods → 4-axis design space → results table → ablation discussion → conclusion.
- Optional: probe-over-training learning curves. Several individual runs (notably the 3-epoch Exp A short-checkpoint, α=0.085) hint that early-training representations may be different — and possibly better — than fully-converged ones for α probing. Deferred from the original plan as a §6.3 open item.
- Optional: a CNN × shared (non-EMA) cell to isolate whether EMA is doing the work or whether CNN is doing it on its own. One cell, ~30 min.

---

## Documents in this repo

- [`README.md`](README.md) — project description, headline results, quick start
- [`COMPLETED.md`](COMPLETED.md) — this file
- [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) — design plan from the Phase-3 refactor
- [`ABLATION_PLAN.md`](ABLATION_PLAN.md) — tier-1/tier-2 cell ordering for the Phase-4 queue
- [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) — bug ledger from the development process (9 entries)
- [`ENV.md`](ENV.md) — vendored upstream commits + dataset location
- [`results/`](results/) — all analytical outputs, see `results/README.md`
