# Environment and upstream pinning

## Python environment

Use Python 3.10 or 3.11. Create a fresh venv or conda env, then:

```
pip install -r requirements.txt
```

GPU PyTorch should be installed first per the CUDA version on your machine.

## Upstream codebases (vendored)

Source files under `src/` that were copied from upstream carry a header like:

```
# === VENDORED: origin ===
# Upstream repo: ...
# Upstream path: ...
# Upstream commit: ...
```

Pinned commits:

| Repo | URL | Commit |
|------|-----|--------|
| V-JEPA | https://github.com/facebookresearch/jepa | `51c59d518fc63c08464af6de585f78ac0c7ed4d5` |
| Physics JEPA | https://github.com/helenqu/physical-representation-learning | `bb77f7b5b506ba793ca7e746e1d0e3c12f70c0db` |

Local clones of both repos live at `C:/Users/Jubin/refs/jepa` and `C:/Users/Jubin/refs/physical-representation-learning` for diffing against upstream.

## Dataset location

The Well `active_matter` shards live at:

```
<project_root>/data/active_matter/data/{train,valid,test}/*.hdf5
```

Pass `data_dir = <project_root>/data/active_matter` to `WellDatasetForJEPA`; it appends `/data/<split>/` internally.

Sizes: 45 train / 16 valid / 21 test files, 3 trajectories per file, 81 timesteps × 256×256 grid. Total ~49 GB. Each filename encodes `L`, `zeta`, `alpha`; those values are also stored in the HDF5 root attributes and under `scalars/`.

Fields inside each HDF5 file:
- `t0_fields/concentration` — (3, 81, 256, 256), ϕ
- `t1_fields/velocity` — (3, 81, 256, 256, 2), u₁ u₂
- `t2_fields/D` — (3, 81, 256, 256, 2, 2), orientation tensor
- `t2_fields/E` — (3, 81, 256, 256, 2, 2), strain-rate tensor
- `scalars/{L,alpha,zeta}` — scalar physical parameters

## Compute

- Checkpoint dir: `/scratch/$NETID/` on HPC; local `outputs/` otherwise.
- All SLURM jobs use `#SBATCH --requeue` for spot-instance support.
