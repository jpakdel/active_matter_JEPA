# Visualizations

Figures for the report and slides. Each script is self-contained — pick representative cells, load their saved features (`runs/<run_id>/features/test.pt`), produce one or more outputs in `outputs/`.

For the per-figure description and underlying numbers, see [`ANALYSIS.md`](ANALYSIS.md).

## Scripts and outputs

| Script | What it produces |
|---|---|
| `01_routing_concepts.py` | `outputs/routing_concepts.png` — 3-panel conceptual cartoon of what each routing asks the encoder to learn. No data; pure schematic. |
| `02_features_pca_3d.py` | `outputs/pca3d_<routing>_<param>.gif` (×6) + `outputs/pca3d_panel.png` — 3D PCA scatter of test-set encoder features per routing, colored by α and by ζ. Rotating GIFs (16 s revolution) plus a static panel. |
| `03_features_spectrum.py` | `outputs/feature_spectrum.png` + `outputs/feature_spectrum_table.tsv` — singular-value spectrum of test features per routing on a log y-axis, annotated with participation ratio. |
| `04_energy_landscape.py` | `outputs/landscape_<routing>_<param>.gif` (×6) + `outputs/landscape_panel.png` — α (or ζ) plotted as a height surface over the top-2 PCs of the encoder's test features, with a translucent RBF surface fit. |
| `07_raw_field_pca.py` | `outputs/raw_pca_<routing>_<param>.{gif,png}` (×6 each) + `outputs/raw_pca_panel.png` — PCA directly on the routing-sliced raw ctx fields. No encoder, no summary statistics. The honest "what's in the raw input data" view. Empirically: the top-3 raw-data PCs do *not* show α or ζ stratification for any routing, so any encoder that recovers the parameters does so via low-variance or non-linear combinations the encoder learns. |

## Representative cells

| routing | run_id | α_lin | ζ_lin | α_kNN | ζ_kNN |
|---|---|---|---|---|---|
| baseline | `baseline_vit_ema_vicreg_lam001_20260430_170646` | 0.006 | 0.068 | 0.015 | 0.102 |
| exp_a | `exp_a_vit_ema_vicreg_lam001_20260430_202310` | 0.070 | 0.217 | 0.162 | 0.807 |
| exp_b | `exp_b_vit_ema_vicreg_20260428_142657` | 0.716 | 0.889 | 0.751 | 1.067 |

All ViT + EMA. The exp_b cell falls back to the standard `vicreg` recipe because no `vit_ema_vicreg_lam001` exp_b cell was trained.

## How to run

```bash
cd REFACTORED_CODEBASE
python visualizations/01_routing_concepts.py
python visualizations/02_features_pca_3d.py
python visualizations/03_features_spectrum.py
python visualizations/04_energy_landscape.py
python visualizations/07_raw_field_pca.py
```

Each script writes to `visualizations/outputs/` and prints what it wrote. `02`, `04`, and `07` each render six 16-second GIFs at 130 dpi — about 2 minutes per GIF. `07` walks the active-matter dataset at 64×64 resolution (one pass, ~30 s).

`scipy>=1.7` is required for `04` (RBF surface fitting).

## What the (x, y, z) coordinates are

| Script | What the coordinates are | Encoder used? |
|---|---|---|
| `02` | Top-3 PCA coords of frozen-encoder test features | Yes |
| `03` | Singular values of frozen-encoder test features | Yes |
| `04` | Top-2 PCA coords of frozen-encoder features (x, y); ground-truth α or ζ as height (z) | Yes |
| `07` | Top-3 PCA coords of *raw* routing-sliced ctx fields (no encoder, no summary) | No |

## Adding more

`02`, `04`, and `07` take their representative cells / routings from a small list at the top of the file — edit that list to swap in different runs. To use a CNN cell with `02` / `04`, no code change needed (the loader uses the saved features directly, which are 256-dim for CNN vs 384-dim for ViT — PCA handles either).

## 2D companions and CSV exports

Every rotating 3D GIF in scripts `02 / 04 / 07` has two companion files of the same basename:

| extension | what it is |
|---|---|
| `.gif` | rotating 3D scatter |
| `.png` | flattened 2D top-down scatter — same dots, drops one axis |
| `.csv` | underlying coordinates per snippet, including the dropped axis, plus the color value |

The 2D PNGs and CSVs are always re-rendered when you run a script (each takes <1 s). The GIFs are skipped when they already exist on disk (each takes ~2 min). **Delete a GIF to force its re-render.** Shared helpers live in `_utils.py`.
