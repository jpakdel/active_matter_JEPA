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

Most-significant result: **`baseline_cnn_ema_vicreg_no_cov`** at α=0.0195 lin / **0.0131 kNN** — the project leader at the time. The "no_cov" choice (variance-only VICReg, matching the teammate's exact recipe) outperformed the cov-on variant for α on the baseline routing in this set.

The cov term turns out to be **routing-dependent**:

- baseline (easy): cov OFF is best for α (0.0131 kNN with cov off → 0.348 with cov on)
- exp_a (D → u, harder): cov ON is best (0.391 → 0.203)
- exp_b (∇·D → Δu, hard): cov ON is best (0.541 → 0.410)

In our cells, cov-on helped α on exp_a and on exp_b (CNN); cov-off was better on baseline.

Detailed cross-axis analysis: [`results/ABLATION_EFFECTS.md`](results/ABLATION_EFFECTS.md).

---

## Phase 5 — Bloat audit + results consolidation (2026-04-29)

A second-pass cleanup of the refactored codebase. 690 lines removed across 10 files: dead V-JEPA factory functions (`vit_huge`, `vit_giant`, `vit_gigantic`), dead transformer modules (`CrossAttention`, `CrossAttentionBlock`), back-compat aliases in `builders.py`, dead helpers in `tensors.py` and `pos_embs.py`, retired one-time launcher and test scripts.

In parallel, all analytical results were consolidated under `results/`:

- [`results/README.md`](results/README.md) — index for the folder
- [`results/PARETO.md`](results/PARETO.md) — top-5 by each of 4 metrics + Pareto frontier
- [`results/ABLATION_EFFECTS.md`](results/ABLATION_EFFECTS.md) — paired comparisons for every design axis
- [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md) — all runs with all metrics + hyperparameters
- [`results/METRICS.md`](results/METRICS.md) — wall time, parameter counts, final losses
- `results/_runs.json` — machine-readable raw inventory

The generator at `scripts/_gen_results_docs.py` rebuilds the four data-driven docs from live run dirs. Re-run any time new training runs land.

State at end of Phase 5: ~5,750 lines of source across 28 live modules; 26 trained-and-evaluated runs spanning the 4-axis design space.

---

## Phase 6 — Round-2 sweep (2026-04-30)

A second batch of 8 cells was queued via `scripts/run_round2_recommended.sh` and ran to completion in 6h31m (16:17 → 22:48 on 2026-04-30). All 8 cells finished without NaN or failure.

| # | cell | α_lin | ζ_lin | α_kNN | ζ_kNN |
|---|---|---|---|---|---|
| 1 | baseline+vit+ema+vicreg_no_cov | 0.099 | 0.714 | 0.192 | 0.533 |
| **2** | **baseline+vit+ema+vicreg_lam001** | **0.0063** ★ | **0.0680** ★ | 0.0147 | **0.1017** ★ |
| 3 | baseline+vit+ema+sigreg_lam1 | 0.131 | 0.372 | 0.222 | 0.454 |
| 4 | baseline+vit+ema+sigreg_lam001 | 0.209 | 0.327 | 0.173 | 0.404 |
| 5 | baseline+vit+ema+vicreg_varw10 | 0.037 | 0.294 | 0.046 | 0.400 |
| 6 | exp_a+vit+ema+vicreg_lam001 | 0.070 | 0.217 | 0.162 | 0.807 |
| 7 | exp_b+cnn+ema+vicreg_lam001 | 0.236 | 0.205 | 0.475 | 0.364 |
| 8 | exp_a+vit+shared+vicreg_lam001 | 0.090 | 0.255 | 0.202 | 0.670 |

**Headline:** `baseline+vit+ema+vicreg_lam001` (cell 2) is the new project champion on three of four metrics. It loses α-kNN to `baseline+cnn+ema+vicreg_no_cov` (0.0131) by 0.0016 but wins α-linear by 3× (0.0063 vs 0.0195). On ζ it improves on the prior best by 29% (kNN) and 49% (linear).

**Two of cell 2's axes are cleanly ablated within this batch**:

- Routing flip (cell 2 vs cell 6, both `vit+ema+vicreg_lam001`, only routing differs) → α 11× worse, ζ 8× worse going baseline → exp_a
- Loss perturbations (cells 3, 4, 5) all underperform cell 2 — α 3–15× worse, ζ 4–5× worse depending on the perturbation; `vicreg_lam001` is the sweet spot in this set

The target axis is **not** cleanly ablated from cell 2 within this batch: cell 8 differs from cell 2 in both routing and target. The closest target-only flip in the dataset is cell 6 vs cell 8 (both `exp_a+vit+vicreg_lam001`): target=shared is 25% worse on α-kNN (0.202 vs 0.162) but 17% better on ζ-kNN (0.670 vs 0.807). Separately, the `baseline+vit+vicreg` recipe (without `lam001`) shows target=shared → ema is a 13× α-linear improvement, so the target axis is load-bearing on at least one related recipe.

**Negative finding**: `final_pred_mse` (the JEPA training loss at end-of-training) is not a reliable proxy for probe quality. Cell 5 reached `pred_mse=0.019` (15× lower than cell 2's 0.288) but produced markedly worse probes than cell 2.

State at end of Phase 6: **34 trained-and-evaluated runs**.

---

## Phase 7 — Visualization studies (2026-05-01)

Built [`visualizations/`](visualizations/) — five self-contained scripts that produce paper / slide figures from the saved `runs/<id>/features/test.pt` tensors and the raw input fields. See [`visualizations/README.md`](visualizations/README.md) for the file index and [`visualizations/ANALYSIS.md`](visualizations/ANALYSIS.md) for the per-figure description and underlying numbers.

A few measurable properties of the trained-encoder features for the chosen exp_b cell (`exp_b_vit_ema_vicreg_20260428_142657`) compared to the chosen baseline cell (`baseline_vit_ema_vicreg_lam001_20260430_170646`):

- exp_b's test-feature singular values are ~1000× smaller in absolute magnitude than baseline's across the entire spectrum (see `outputs/feature_spectrum.png`).
- Participation ratio: exp_b 28.5, baseline 10.0.
- Top-3 PCs explain 21% of exp_b's variance, 48% of baseline's.

These are descriptive of the two specific cells; they do not generalize to all exp_b runs. The full per-routing range of trained-cell numbers is in [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md).

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

Total compute through Phase 6: **27 ViT runs + 7 CNN runs ≈ 26.7 hours of GPU on a single RTX 4070 SUPER.**

Per-run record in `results/METRICS.md`.

---

## What's left

- Final report writeup. After Phase 6, the headline shifted: a single ViT recipe (`baseline+vit+ema+vicreg_lam001`) now dominates 3 of 4 metrics; the prior "CNN-for-α / ViT-for-ζ" dichotomy survives only as a narrow α-kNN advantage for the CNN cell that doesn't reproduce on α-linear. Recommended report structure: methods → 4-axis design space → results table → ablation discussion → conclusion.
- Optional: probe-over-training learning curves. Several individual runs (notably the 3-epoch Exp A short-checkpoint, α=0.085) hint that early-training representations may be different — and possibly better — than fully-converged ones for α probing. Deferred from the original plan as a §6.3 open item.
- Optional: a CNN × shared (non-EMA) cell to isolate whether EMA is doing the work or whether CNN is doing it on its own. One cell, ~30 min.

---

## Documents in this repo

- [`README.md`](README.md) — project description, headline results, quick start
- [`COMPLETED.md`](COMPLETED.md) — this file
- [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md) — design plan from the Phase-3 refactor
- [`ABLATION_PLAN.md`](ABLATION_PLAN.md) — tier-1/tier-2 cell ordering for the Phase-4 queue + round-2 record
- [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) — bug ledger from the development process (9 entries)
- [`ENV.md`](ENV.md) — vendored upstream commits + dataset location
- [`results/`](results/) — all analytical outputs, see `results/README.md`
- [`visualizations/`](visualizations/) — paper / slide figures, see `visualizations/README.md`
