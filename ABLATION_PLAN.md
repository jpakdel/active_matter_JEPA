# Ablation Plan

What gets trained when the GPU comes free.

## Constraints

- One training run takes ~46 min on the RTX 4070 SUPER at the existing config.
- Eval (feature extract + linear probe + kNN) takes ~5 min per run.
- A run starts to finish (train + eval) is ~51 min.
- 4-axis CLI: `--routing {baseline,exp_a,exp_b}`, `--backbone {vit,cnn}`, `--target {shared,ema}`, `--loss {…}`.

## Already in the can (won't re-run)

The following 14 runs from 2026-04-21 / 04-22 / 04-24 already exist under the original repo's `runs/`. Numbers carry forward; trust them.

| Routing | Backbone | Target | Loss | Run dir |
|---|---|---|---|---|
| baseline | vit | shared | sigreg | `baseline_v0_20260421_152635` |
| exp_a | vit | shared | sigreg | `exp_a_v0_20260421_185035` |
| exp_b | vit | shared | sigreg | `exp_b_v0_20260421_200801` |
| exp_b | vit | shared | sigreg_lam001 | `exp_b_lam001_v0_20260422_113047` |
| exp_b | vit | shared | sigreg_lam1 | `exp_b_lam1_v0_20260422_121613` |
| baseline | vit | shared | vicreg | `djepa_baseline_vicreg_v0_20260424_000719` |
| exp_a | vit | shared | vicreg | `djepa_exp_a_vicreg_v0_20260424_005651` |
| exp_b | vit | shared | vicreg | `djepa_exp_b_vicreg_v0_20260424_014519` |
| exp_b | vit | shared | vicreg_lam001 | `djepa_exp_b_vicreg_lam001_v0_20260424_023349` |
| exp_b | vit | shared | vicreg_lam1 | `djepa_exp_b_vicreg_lam1_v0_20260424_032214` |
| exp_b | vit | shared | vicreg_varw10 | `djepa_exp_b_vicreg_varw10_v0_20260424_041038` |
| exp_b | vit | shared | vicreg_covw5 | `djepa_exp_b_vicreg_covw5_v0_20260424_045839` |
| exp_b | vit | shared | vicreg_varw50 | `djepa_exp_b_vicreg_varw50_v0_20260424_054642` |

These cover ViT × shared × {3 routings × {sigreg, vicreg defaults}} plus the exp_b regularizer sweep.

## Tier 1 — 12 cells, ~10.2 hours (high-priority)

Ordered by information value: most-informative first. If the queue gets killed midway, what's already done is the most useful subset.

| # | Routing | Backbone | Target | Loss | Question answered |
|---|---|---|---|---|---|
| 1 | baseline | cnn | ema | vicreg_no_cov | Replicate teammate's recipe on my data — baseline cell |
| 2 | exp_b | cnn | ema | vicreg_no_cov | Replicate teammate where he beats me (his α=0.145) |
| 3 | exp_a | cnn | ema | vicreg_no_cov | Replicate teammate where I beat him (his α=0.291) |
| 4 | exp_a | vit | ema | vicreg | Does EMA improve my best probe (existing α=0.087)? |
| 5 | exp_b | vit | ema | vicreg | Does EMA rescue ViT exp_b under VICReg (was collapsed, α=0.753)? |
| 6 | baseline | vit | ema | vicreg | Does EMA rescue ViT baseline under VICReg (was 0.417)? |
| 7 | exp_a | vit | ema | sigreg | SIGReg + EMA — novel combination on my best route |
| 8 | exp_b | vit | ema | sigreg | SIGReg + EMA on the route SIGReg already had α=0.593 |
| 9 | baseline | vit | ema | sigreg | SIGReg + EMA on the cleanest route |
| 10 | exp_b | cnn | ema | vicreg | CNN with full VICReg (cov term on) — does cov hurt or help? |
| 11 | exp_a | cnn | ema | vicreg | CNN + full VICReg, exp_a |
| 12 | baseline | cnn | ema | vicreg | CNN + full VICReg, baseline |

After cells 1–3 are in the can (~2.5h), I have a head-to-head with the teammate.
After cells 4–6 (~5h), I know whether EMA fixes my VICReg failures.
After cells 7–9 (~7.5h), I have the SIGReg+EMA cell across all routings.
After 10–12 (~10.2h), I know how the cov term behaves on CNN.

## Tier 2 — 4 stretch cells, +3.4 hours = ~13.6 total

| # | Routing | Backbone | Target | Loss | Question answered |
|---|---|---|---|---|---|
| 13 | exp_b | cnn | ema | sigreg | CNN + SIGReg — does the LeJEPA loss help on the CNN backbone? |
| 14 | exp_a | cnn | ema | sigreg | (same q, exp_a) |
| 15 | baseline | cnn | ema | sigreg | (same q, baseline) |
| 16 | exp_b | vit | ema | vicreg_no_cov | Apples-to-apples with cell #2 (same recipe, different backbone) |

If the queue runs to completion, this gives a full 16-cell sweep covering both backbones across all three routings under both regularizer families, all with EMA on. Combined with the existing 14 shared-encoder runs, total dataset = 30 cells.

## Round 2 — Surrogate-driven (2026-04-30, 8 cells, ~6.5 hours, completed)

After 26 cells were in the can, a random-forest surrogate over the one-hot 4-axis design space (`predict_configs.py` at the project root) predicted probe quality at the 94 untested cells and produced a curated top-5 per metric (filter: hamming=1 from observed, no Mode-3 collapse, no Mode-1-without-EMA-rescue). The deduped union — 8 cells — was queued via [`scripts/run_round2_recommended.sh`](scripts/run_round2_recommended.sh).

| # | Routing | Backbone | Target | Loss | Why picked |
|---|---|---|---|---|---|
| 1 | baseline | vit | ema | vicreg_no_cov | curated α top-1 (predicted 0.136) |
| 2 | baseline | vit | ema | vicreg_lam001 | curated α top-3 + ζ top-1 |
| 3 | baseline | vit | ema | sigreg_lam1 | curated α top-2 + ζ top-5 |
| 4 | baseline | vit | ema | sigreg_lam001 | curated α top-4 |
| 5 | baseline | vit | ema | vicreg_varw10 | curated α top-5 |
| 6 | exp_a | vit | ema | vicreg_lam001 | curated ζ top-3 |
| 7 | exp_b | cnn | ema | vicreg_lam001 | curated ζ top-2 (M4 risk flagged) |
| 8 | exp_a | vit | shared | vicreg_lam001 | curated ζ top-4 |

Sweep ran 16:17–22:48 on 2026-04-30, all 8 cells finished without NaN or eval failure. **Cell 2 (`baseline+vit+ema+vicreg_lam001`) became the new project champion on three of four metrics.** Full per-cell results in [`COMPLETED.md`](COMPLETED.md) (Phase 6) and [`results/PARETO.md`](results/PARETO.md).

## Round 3 — Not yet planned

Refresh the surrogate against the 34-cell observation set (the Phase-6 surrogate had α-Spearman ρ=0.60 on 26 cells; with 34 cells it should rank more reliably) and re-pick the top recommendations. The launcher and resume logic handle this with no changes; the only step is to re-run `predict_configs.py --runs-dir runs/ --runs-dir REFACTORED_CODEBASE/runs/`.

## Not run (deliberate)

- shared-encoder runs on cells already done. No new info.
- patch-size ablations on ViT. User dropped this axis to save compute.
- shared-encoder CNN runs. Less informative — no a-priori reason to think CNN works without EMA, and the teammate's exp_a (CNN no-EMA) was his weakest setup.
- baseline replicates of pre-refactor runs. Skip; trust the old numbers.

## What runs at 2:03 AM

The cron fires `bash REFACTORED_CODEBASE/scripts/run_all_ablations.sh` (under `run_in_background=true`). That script:

1. cd into `REFACTORED_CODEBASE/`
2. Verify code via `python tests/test_build_and_shapes.py` first; if it fails, the script aborts before burning GPU on a broken pipeline.
3. Run the 16 cells in order (Tier 1 then Tier 2). Per-cell:
   - `python scripts/train.py --routing X --backbone Y --target Z --loss W` → logs to `runs/<cell>.train.log`
   - `python scripts/eval.py --run-dir <new_run_dir>` → logs to `runs/<cell>.eval.log`
   - Continue on failure (`set -u` but not `set -e`); failures are visible in `runs/manifest.tsv` (`status=crashed`).
4. After all 16: write `runs/ablation_summary.txt` collating final.json + eval_results.json from each.

If something breaks structurally (an OOM at a particular cell config, a module-level bug), the queue continues past it. Worst case we lose one cell out of 16.
