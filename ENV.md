# Environment and upstream pinning

## Python environment

Python 3.10 or 3.11. Create a fresh venv or conda env, then:

1. Install GPU PyTorch first per your CUDA version (https://pytorch.org/get-started/locally/).
2. Install the rest:

```
pip install -r requirements.txt
```

Tested on PyTorch 2.x with CUDA 12.x.

## Vendored upstream codebases

Source files under `src/` that were copied from upstream carry a header naming the upstream path and commit:

```
# === VENDORED: origin ===
# Upstream repo: ...
# Upstream path: ...
# Upstream commit: ...
```

Pinned commits:

| Repo | URL | Commit | What's used |
|------|-----|--------|---|
| V-JEPA | https://github.com/facebookresearch/jepa | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` | ViT encoder, transformer blocks, 3D patch embed (tubelet), sincos position embeddings, cosine LR / WD schedulers |
| Physics JEPA | https://github.com/helenqu/physical-representation-learning | `bb77f7b5b506ba793ca7e746e1d0e3c12f70c0db` | `WellDatasetForJEPA` HDF5 loader (`src/data/well_dataset.py`) |
| LeJEPA | https://github.com/rbalestr-lab/lejepa | `c293d291ca87cd4fddee9d3fffe4e914c7272052` | SIGReg loss (Balestriero & LeCun, arXiv 2511.08544) — `src/losses/sigreg.py` |

## Re-implemented from papers (no upstream commit pin)

| Component | Source paper | Where |
|---|---|---|
| VICReg | Bardes, Ponce, LeCun. ICLR 2022. arXiv 2105.04906 | `src/losses/vicreg.py` |
| BYOL EMA target encoder | Grill et al. NeurIPS 2020. arXiv 2006.07733 | `src/train/ema.py` |
| CNN backbone | Written fresh, structurally analogous to a parallel project's CNN (no shared code) | `src/models/cnn_encoder.py` |

## Dataset

The Well's `active_matter` shards are expected at:

```
<project_root>/data/active_matter/data/{train,valid,test}/*.hdf5
```

where `<project_root>` is the directory **containing** this codebase (one level up). Pass `data_dir = <project_root>/data/active_matter` to `WellDatasetForJEPA`; it appends `/data/<split>/` internally. The default config does this with the relative path `../data/active_matter`.

Sizes: 45 train / 16 valid / 21 test files, 3 trajectories per file, 81 timesteps × 256×256 grid. Total ~49 GB.

Each filename encodes `L`, `zeta`, `alpha`; those values are also stored in the HDF5 root attributes and under `scalars/`. Fields inside each HDF5 file:

| Field path | Shape | Meaning |
|---|---|---|
| `t0_fields/concentration` | `(3, 81, 256, 256)` | ϕ |
| `t1_fields/velocity` | `(3, 81, 256, 256, 2)` | u₁, u₂ |
| `t2_fields/D` | `(3, 81, 256, 256, 2, 2)` | orientation tensor |
| `t2_fields/E` | `(3, 81, 256, 256, 2, 2)` | strain-rate tensor |
| `scalars/{L,alpha,zeta}` | scalar | per-trajectory physical parameters |

## Compute

Per-cell training: ~46 min on a single RTX 4070 SUPER. Full 34-run sweep: ~26.7 GPU-hours.

Mixed precision is per-backbone (recorded per run in `runs/<run_id>/config.json` under `optim.use_amp`):

- ViT: AMP fp16
- CNN: fp32 (AMP triggered NaN gradients in the CNN encoder; documented as a numerical-stability boundary)

Spot-instance restart: every run is resumable via `python scripts/train.py --routing X --backbone Y --target Z --loss W` — the launcher detects an existing run dir for the same axis tuple and resumes from the latest checkpoint.
