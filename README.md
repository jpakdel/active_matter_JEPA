# D-JEPA on `active_matter`

Self-supervised representation learning on The Well's `active_matter` simulation, with frozen-encoder evaluation against two underlying physical parameters: **α** (active dipole strength) and **ζ** (steric alignment).

The architecture is JEPA-style — a context encoder produces a latent, a predictor maps it forward, and the prediction is matched to a target encoder's output. Anti-collapse is provided by either **VICReg** (variance hinge, optionally with off-diagonal covariance penalty), **SIGReg** (slice-wise distributional matching to N(0,1), from LeJEPA), or a **BYOL-style EMA target encoder**, or any combination.

The project trains across a 4-axis design space — routing × backbone × target × loss — and reports linear-probe and kNN test MSE on z-scored α and ζ.

---

## Headline result

After 34 trained-and-evaluated runs, the project champion is `baseline + vit + ema + vicreg_lam001`:

| Metric | Test MSE (z-scored, lower=better) |
|---|---|
| α linear probe | **0.0063** |
| α kNN | 0.0147 *(narrow #2 to `baseline + cnn + ema + vicreg_no_cov` at 0.0131)* |
| ζ linear probe | **0.068** |
| ζ kNN | **0.102** |

A constant-mean predictor scores 1.0; lower is better. Full inventory: [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md). Top-5s and Pareto frontier: [`results/PARETO.md`](results/PARETO.md).

**Pretrained weights for the champion are published on Hugging Face:** [`JJP9216NYUBB/JepaPhysics`](https://huggingface.co/JJP9216NYUBB/JepaPhysics) — both the online encoder and its EMA twin, plus a self-contained `model.py` with no dependencies beyond `torch` and `numpy`. See the [Use the pretrained weights](#use-the-pretrained-weights-from-hugging-face) section below for a load snippet.

---

## Quick start

### Install

```bash
# 1. Install GPU PyTorch first per your CUDA version (https://pytorch.org)
# 2. Then install the rest:
pip install -r requirements.txt
```

Tested on Python 3.10 / 3.11. See [`ENV.md`](ENV.md) for the upstream commit pins of every vendored module.

### Data

The Well's `active_matter` HDF5 shards are expected at `<project_root>/data/active_matter/data/{train,valid,test}/*.hdf5` — i.e. one directory above this codebase. The paths and channel layout are documented in [`ENV.md`](ENV.md). Sizes: 45 train / 16 valid / 21 test files, ~49 GB total.

### Use the pretrained weights from Hugging Face

The champion's weights are published at [`JJP9216NYUBB/JepaPhysics`](https://huggingface.co/JJP9216NYUBB/JepaPhysics). To skip training entirely and use the encoder for downstream feature extraction:

```bash
pip install huggingface_hub torch numpy
```

```python
import torch, json
from huggingface_hub import snapshot_download
import importlib.util
from pathlib import Path

local = Path(snapshot_download("JJP9216NYUBB/JepaPhysics"))
spec = importlib.util.spec_from_file_location("hf_model", local / "model.py")
hf_model = importlib.util.module_from_spec(spec); spec.loader.exec_module(hf_model)

cfg = json.load(open(local / "config.json"))
m = cfg["model"]["encoder"]
encoder = hf_model.build_encoder(
    in_chans=11,
    size=cfg["model"]["encoder_size"],
    img_size=m["img_size"], num_frames=m["num_frames"],
    patch_size=m["patch_size"], tubelet_size=m["tubelet_size"],
    mlp_ratio=m["mlp_ratio"],
)
encoder.load_state_dict(torch.load(local / "encoder.pt", map_location="cpu"))
encoder.eval()

# x: (B, 11, 16, 256, 256) — 16-frame windows of the 11-channel active_matter fields
with torch.no_grad():
    z = encoder(x)            # (B, 2048, 384)
    pooled = z.mean(dim=1)    # (B, 384)
```

Both the online encoder (`encoder.pt`) and the EMA target encoder (`target_encoder_ema.pt`) are shipped. The full HF model card has the architecture details, intended use, and limitations.

### Train one cell

```bash
# Reproduce the project champion (baseline + vit + ema + vicreg_lam001):
python scripts/train.py \
    --routing baseline \
    --backbone vit \
    --target ema \
    --loss vicreg_lam001
```

The four axes are independently selectable (`--routing {baseline, exp_a, exp_b}`, `--backbone {vit, cnn}`, `--target {shared, ema}`, `--loss <preset>`). Loss presets live under [`configs/active_matter/losses/`](configs/active_matter/losses/) — there are 10 of them (sigreg / sigreg_lam001 / sigreg_lam1 / vicreg / vicreg_lam001 / vicreg_lam1 / vicreg_no_cov / vicreg_varw10 / vicreg_varw50 / vicreg_covw5).

A single cell takes ~46 min on an RTX 4070 SUPER (ViT, AMP fp16) or ~46 min (CNN, fp32). Spot-instance restart is supported — re-launching with the same `--routing/--backbone/--target/--loss` resumes from the latest checkpoint.

### Evaluate a finished run

```bash
python scripts/eval.py --run-dir runs/<run_id>
```

Extracts pooled frozen-encoder features for train/val/test, fits a closed-form ridge linear probe (α swept on val) and a kNN regressor (k and metric swept on val), writes `runs/<run_id>/eval_results.json` with per-target test MSE for α and ζ.

### Reproduce all 34 runs from the manifest

```bash
# Re-run the entire 4-axis sweep — see runs/manifest.tsv for the cells trained.
# Each cell takes ~46 min; the full sweep is ~26 GPU-hours.
for cell in $(cat runs/manifest.tsv | tail -n+2 | cut -f1); do
    # Decode the routing/backbone/target/loss from the run_id and call train.py.
    # Example: baseline_vit_ema_vicreg_lam001_20260430_170646
    #   -> --routing baseline --backbone vit --target ema --loss vicreg_lam001
    ...
done
```

`scripts/run_tier1_cnn.sh` and `scripts/run_round2_recommended.sh` are preserved as queue-launcher templates. Each finished run produces a `config.json` (the merged config used) which can be re-run by passing the same axis values.

---

## Compliance with assignment constraints

| Constraint | How it's satisfied |
|---|---|
| **Train from scratch — no pretrained weights** | All weights initialized via `trunc_normal_` / `kaiming_normal_`. No `load_state_dict` from any third-party checkpoint. See `src/models/{vit_encoder,cnn_encoder,simple_predictor}.py` and the absence of any `--pretrained` flag. |
| **Only `active_matter` dataset — no external data** | Only data source is `WellDatasetForJEPA` reading the HDF5 shards under `data/active_matter/`. See `src/data/well_dataset.py`. |
| **No labels during representation learning** | Training uses `(context, target)` tensors and an unsupervised JEPA objective (`pred_mse + λ·reg`). The α/ζ labels are loaded only by the eval pipeline (`src/eval/`), never by `src/train/`. |
| **Frozen encoder for evaluation** | `src/eval/extract_features.py` calls `encoder.eval()` and runs under `torch.no_grad()`. Linear probe and kNN operate on the cached `(N, D)` feature tensors only — gradients never reach the encoder. |
| **Both linear probe and kNN regression** | `src/eval/linear_probe.py` (closed-form ridge with α swept on val) and `src/eval/knn_regression.py` (k and L2/cosine metric swept on val). Both run on every cell; per-run results in `runs/<run_id>/eval_results.json`. |
| **Regression with MSE only — no classification, no MLP heads** | Linear probe is a single `Linear` layer with closed-form ridge. kNN is non-parametric averaging. Both report z-scored MSE per target. |
| **Model under 100M parameters** | ViT-small encoder (~23.5M) + predictor (~10.9M) ≈ 34.4M total. CNN encoder (~3.8M) + predictor (~4.9M) ≈ 8.6M total. Per-run param counts recorded in `runs/<run_id>/final.json`; aggregate in [`results/METRICS.md`](results/METRICS.md). |
| **Train/val/test splits used correctly** | The dataset has fixed splits at the file level — `train/`, `valid/`, `test/` HDF5 directories. Probe ridge α swept on **val only**; final test MSE reported on **test only** (val labels never touch the test prediction). Train-split z-score statistics applied to val/test. See `src/eval/normalize_labels.py:fit_label_stats`. |
| **Checkpointing and restart support (spot instances)** | Atomic save via tmp→rename in `src/train/checkpoint.py`. Full RNG state (CPU, CUDA, NumPy, Python) is saved and restored. `--resume` reloads optimizer, scheduler, and step counters bit-exact. |

---

## Repo layout

```
REFACTORED_CODEBASE/
├── README.md                # this file
├── ENV.md                   # upstream commit pins, dataset path, environment notes
├── requirements.txt
│
├── configs/active_matter/
│   ├── default.yaml                     # base: data, optim, log, schedule
│   ├── {baseline,exp_a,exp_b}.yaml      # 3 routing presets
│   ├── backbones/{vit,cnn}.yaml         # 2 backbone presets
│   ├── targets/{shared,ema}.yaml        # 2 target-encoder presets
│   └── losses/                          # 10 regularizer presets
│
├── src/                     # ~5,750 lines, 28 live modules
│   ├── config_loader.py     # layered YAML merger (default + 4 axes + CLI overrides)
│   ├── data/                # WellDatasetForJEPA, derived fields (∇·D, Δu), channel map
│   ├── eval/                # frozen-encoder feature extract → ridge + kNN
│   ├── losses/              # DJepaLoss switch (sigreg | vicreg) + DDP helpers
│   ├── masks/utils.py       # apply_masks (used by ViT trunk)
│   ├── models/              # ViT encoder, CNN encoder, dual variants, predictor, pos embs
│   └── train/               # builders, train loop, optimizer, schedulers, EMA, checkpoint, manifest
│
├── scripts/
│   ├── train.py                            # main launcher
│   ├── eval.py                             # frozen-encoder eval (linear + kNN probe)
│   ├── analyze.py                          # cross-run PCA / PC-correlation / residual leakage
│   ├── _gen_results_docs.py                # rebuilds RUN_INVENTORY / METRICS / PARETO
│   ├── recover_evals.sh                    # mop-up: eval any run with final.json but no eval_results.json
│   ├── run_tier1_cnn.sh                    # batch launcher template (preserved)
│   ├── run_round2_recommended.sh           # batch launcher template (preserved)
│   └── stokes_validation/                  # physics validation of the derived-field kernels
│       ├── sanity_check_derived_fields.py
│       └── stokes_correlation_analysis.py
│
├── results/                 # all metrics, machine + human readable
│   ├── README.md            # index for this folder
│   ├── PARETO.md            # top-5 by each metric + Pareto frontier
│   ├── ABLATION_EFFECTS.md  # paired-comparison ablation tables per design axis
│   ├── RUN_INVENTORY.md     # all 34 runs with all metrics + hyperparameters
│   ├── METRICS.md           # wall time, parameter counts, final losses
│   └── _runs.json           # machine-readable raw inventory
│
├── visualizations/          # paper / slide figures
│   ├── README.md            # index of scripts, what each one renders, how to run
│   ├── ANALYSIS.md          # per-figure descriptions and the empirical numbers
│   └── 0[1-7]_*.py          # routing concepts, 3D PCA on encoder features, spectrum, landscape, raw-input PCA
│
└── runs/                    # per-run artifacts
    ├── manifest.tsv         # one row per run (cross-run summary)
    └── <run_id>/
        ├── config.json           # frozen merged config used for this run
        ├── metrics.jsonl         # one JSON line per optimizer step (loss components, timing)
        ├── final.json            # wall time, final losses, parameter counts, num_tokens
        ├── eval_results.json     # α/ζ test MSE for ridge and kNN
        ├── checkpoints/          # gitignored (>100 MB each)
        └── features/{meta.json}  # split sizes; the .pt feature tensors are gitignored
```

---

## Reproducibility

Every run produces a frozen artifact set that is sufficient to verify and reproduce its result:

| Artifact | Content |
|---|---|
| `config.json` | Exact merged config: every axis preset + every CLI override resolved. Re-running with the same `config.json` reproduces the trained model up to CUDA nondeterminism. |
| `metrics.jsonl` | Per-step training trajectory: `pred_mse`, `reg`, `total`, lr, wd, grad-norm, step time. |
| `final.json` | Wall time, final loss components, encoder/predictor parameter counts (in millions), `num_tokens` (= 2048 for all configs). |
| `eval_results.json` | Linear-probe α-test MSE, kNN α-test MSE, both ζ test MSEs, plus the val-selected ridge α and kNN k for each target. |
| `manifest.tsv` | One row per run with the above summarized — sortable, audit-friendly. |

Fixed seed (`seed: 0` by default, recorded in `config.json`). Mixed precision flag is recorded per run (`optim.use_amp`); ViT runs use AMP fp16, CNN runs use fp32 (a documented numerical-stability boundary).

No external data, no pretrained weights, no labels during pretraining — see the constraints table above. The compute totals across all 34 runs: ~26.7 GPU-hours on a single RTX 4070 SUPER.

---

## Where to start reading

- **Pretrained weights on Hugging Face**: [`JJP9216NYUBB/JepaPhysics`](https://huggingface.co/JJP9216NYUBB/JepaPhysics) — load and use without training
- **Headline numbers**: [`results/PARETO.md`](results/PARETO.md)
- **Per-axis ablation evidence**: [`results/ABLATION_EFFECTS.md`](results/ABLATION_EFFECTS.md)
- **Full run table**: [`results/RUN_INVENTORY.md`](results/RUN_INVENTORY.md)
- **Compute / parameter accounting**: [`results/METRICS.md`](results/METRICS.md)
- **Figures suitable for the report**: [`visualizations/README.md`](visualizations/README.md) (file index) + [`visualizations/ANALYSIS.md`](visualizations/ANALYSIS.md) (per-figure description with underlying numbers)
- **Physics validation of the derived-field kernels (`∇·D`, `Δu`)**: see the `STOKES_TEST.md` outside this codebase or [`scripts/stokes_validation/`](scripts/stokes_validation/) for the scripts themselves.

---

## Reproducibility audit

Each item below is a checklist requirement, the evidence in this codebase that satisfies it, and any honest gaps.

### 1. Fixed seeds and determinism flags; seeds recorded in logs

**Status**: seeds fixed, recorded; cuDNN/AMP nondeterminism not suppressed.

- **Seed source**: `seed` field in [`configs/active_matter/default.yaml`](configs/active_matter/default.yaml) — default `0`. Override via `--override seed=N` on the CLI.
- **Where it's set**: [`src/train/trainer.py:104`](src/train/trainer.py) — `torch.manual_seed(cfg.get("seed", 0))` at the start of every `train()` call. SIGReg additionally seeds its own `torch.Generator` ([`src/losses/sigreg.py:126-130`](src/losses/sigreg.py)) so the random projection slices are deterministic per-step.
- **Where it's recorded**:
  - Per-run `runs/<run_id>/config.json` — full merged config, `seed` field at top level
  - Per-run `runs/<run_id>/checkpoints/ckpt_*.pt` — full RNG state (CPU torch + per-device CUDA torch + numpy + python `random`) saved by [`src/train/checkpoint.py:31-37`](src/train/checkpoint.py) and restored on `--resume`
  - Cross-run `runs/manifest.tsv` — `seed` column ([`src/train/manifest.py:33`](src/train/manifest.py))
- **Determinism flags NOT set**: this codebase does not call `torch.backends.cudnn.deterministic = True` or `torch.use_deterministic_algorithms(True)`. Combined with AMP fp16 on ViT runs, two re-runs from the same seed will land within a small numerical window but are not bit-identical. The seed-and-RNG-state restoration is sufficient for `--resume` continuity but not for cross-machine bit-exactness.

### 2. Data preprocessing and augmentations

**Status**: deterministic preprocessing pipeline, no training augmentations by default.

- **Channel layout** (fixed, documented in [`src/data/channel_map.py`](src/data/channel_map.py)): 11 channels per frame in this order — `phi` (1), `u_1, u_2` (2), `D_11..D_22` (4), `E_11..E_22` (4). Physical params per trajectory: `(alpha, zeta)`.
- **Preprocessing** ([`src/data/well_dataset.py`](src/data/well_dataset.py)):
  1. Read `(num_objs, T, H, W, *components)` directly from HDF5 via `h5py.File.read_direct` into a preallocated buffer (no per-step Python copies)
  2. Slice a 2·`num_frames` time window starting at `t0` (stride = `num_frames` by default → non-overlapping ctx/tgt windows); split into ctx (first half) and tgt (second half)
  3. Permute to `(C, T, H, W)` and (optionally) bilinearly interpolate spatial dims to the configured `resolution`
  4. (Default `noise_std=0.0`) — Gaussian noise injection is supported but off in every shipped config
- **Channel routing** ([`src/train/builders.py:select_channels`](src/train/builders.py)): a string spec selects channels per branch — `"all"` (11 ch), `"D"` (4 ch), `"u"` (2 ch), `"divD"` (2 ch, computed via `divergence_D` in [`src/data/derived_fields.py`](src/data/derived_fields.py)), `"lapU"` (2 ch, computed via `laplacian_u`).
- **Derived-field kernels** ([`src/data/derived_fields.py`](src/data/derived_fields.py)): central differences with periodic boundary conditions via `torch.roll` (5-point stencil for the Laplacian, 2-point for the divergence). Validated against analytical reference and the Stokes pointwise relation — see [`scripts/stokes_validation/`](scripts/stokes_validation/) and the project-root `STOKES_TEST.md`.
- **Label normalization** ([`src/eval/normalize_labels.py:fit_label_stats`](src/eval/normalize_labels.py)): per-target z-score with mean/std fit on the **train** split only and applied unchanged to val/test. Constant-mean baseline scores exactly 1.0 by construction.
- **Augmentations**: none. No flips, crops, jitter, masking, or stochastic time-windowing. The training stochasticity is entirely from per-batch sampling order.

### 3. Configuration files and command lines for reproduction

**Status**: all configs in-repo; merged config frozen per run; one CLI command per cell.

- **Layered config tree** under [`configs/active_matter/`](configs/active_matter/):
  - 1 `default.yaml` — non-axis defaults (data, optim, log)
  - 3 routing presets — `baseline.yaml`, `exp_a.yaml`, `exp_b.yaml`
  - 2 backbone presets — `backbones/{vit,cnn}.yaml`
  - 2 target presets — `targets/{shared,ema}.yaml`
  - 10 loss presets — `losses/{sigreg, sigreg_lam001, sigreg_lam1, vicreg, vicreg_lam001, vicreg_lam1, vicreg_no_cov, vicreg_varw10, vicreg_varw50, vicreg_covw5}.yaml`
- **Merge order** ([`src/config_loader.py`](src/config_loader.py)): `default` ← `routing` ← `backbone` ← `target` ← `loss` ← `--override key=value` CLI flags. Last writer wins per leaf.
- **Per-run frozen config**: every run writes the resolved merged config to `runs/<run_id>/config.json` ([`src/train/checkpoint.py:160`](src/train/checkpoint.py) `write_run_config`). Re-running from this exact JSON reproduces the trained model up to the cuDNN/AMP caveat in §1.
- **Reproduce the project champion**:
  ```bash
  python scripts/train.py --routing baseline --backbone vit --target ema --loss vicreg_lam001
  python scripts/eval.py --run-dir runs/<resulting_run_id>
  ```
- **Reproduce all 34 runs**: each run's `config.json` decodes to a `--routing/--backbone/--target/--loss` quadruple plus its seed; `runs/manifest.tsv` lists every cell. Templates in [`scripts/run_tier1_cnn.sh`](scripts/run_tier1_cnn.sh) and [`scripts/run_round2_recommended.sh`](scripts/run_round2_recommended.sh) document the exact ordering used for the original sweeps.

### 4. Parameter count report (< 100M cap)

**Status**: every cell well under cap; exact counts recorded per run.

| Backbone | Encoder params | Predictor params | Total |
|---|---|---|---|
| ViT-small (`baseline` routing, 11 ctx ch) | 23.46 M | 10.94 M | **34.40 M** |
| ViT-small (`exp_a` routing, 4 ctx ch) | 22.47 M | 10.94 M | **33.42 M** |
| ViT-small (`exp_b` routing, 2 ctx ch) | 22.08 M | 10.94 M | **33.02 M** |
| CNN | 3.75–3.76 M | 4.87 M | **~8.62 M** |

- **Where it's computed**: [`src/models/encoder.py:count_parameters`](src/models/encoder.py) (`sum(p.numel() for p in module.parameters() if p.requires_grad)`) called at training start.
- **Where it's recorded**: every run's `runs/<run_id>/final.json` has `encoder_params` and `predictor_params` as raw integers.
- **Aggregate table**: [`results/METRICS.md`](results/METRICS.md) — one row per run with both counts in millions plus the total.
- **Compliance**: max single-run total is **34.40 M = 34.4% of the 100 M cap**. The CNN cells are 4× smaller still.

### 5. Compute accounting

**Status**: GPU/hours/AMP recorded per run; **peak memory not recorded** (gap, see below).

- **Hardware**: single NVIDIA RTX 4070 SUPER (12 GB VRAM, single-GPU, no DDP). All 34 runs on this one device.
- **Per-run wall time**: `wall_s` in `runs/<run_id>/final.json`. Aggregated in [`results/METRICS.md`](results/METRICS.md):
  - **ViT**: 27 runs, avg 46.8 min/run
  - **CNN**: 7 runs, avg 46.0 min/run
  - **Total: ~26.7 GPU-hours** across all 34 runs
- **Mixed precision**:
  - **ViT runs use AMP fp16** (`optim.use_amp: true` in [`configs/active_matter/default.yaml`](configs/active_matter/default.yaml))
  - **CNN runs use fp32** (`use_amp: false` overridden in [`configs/active_matter/backbones/cnn.yaml`](configs/active_matter/backbones/cnn.yaml)) — the CNN backward pass NaN's under AMP fp16; documented in [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md). Forward activations stayed healthy in fp16 (~26 peak vs the 6.5e4 overflow point); the overflow is in fp16 gradients (likely GroupNorm or attention softmax). fp32 is ~2× slower per step but stable.
  - The flag is recorded per run in `config.json` (`optim.use_amp`) and the AMP `GradScaler` state is checkpointed alongside model weights so resume bit-matches the pre-crash trajectory.
- **Eval compute**: `scripts/eval.py` is dominated by feature extraction (one forward pass over train+val+test) plus closed-form ridge and a small kNN sweep. Sub-minute per run on the same GPU; not separately tracked because it's one-shot per cell.
- **Peak memory: not recorded.** Neither `final.json` nor `metrics.jsonl` calls `torch.cuda.max_memory_allocated()`, and there's no rolling-max instrumentation in the trainer. The `default.yaml` header comment notes "~2.3 GB peak on RTX 4070 SUPER at size=small" for the ViT cell at `batch_size=2`, but that's a one-shot anecdote, not a logged measurement. **Adding it would be a one-line addition to `train()` near the `final.json` write.**
