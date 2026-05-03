# Visualizations — what each figure shows

Index of the figures produced under [`outputs/`](outputs/) and what they depict, with the per-routing empirical numbers behind them. For the file-list / how-to-run, see [`README.md`](README.md).

---

## Empirical observation

Across 34 trained cells (4-axis design space: routing × backbone × target × loss), the channel routing correlates with probe quality but does not determine it. Per-routing α_kNN MSE on the held-out test split:

| routing | n cells | min α_kNN | median α_kNN | max α_kNN | min ζ_kNN | max ζ_kNN |
|---|---|---|---|---|---|---|
| baseline | 11 | 0.013 | 0.192 | **0.783** | 0.102 | 1.980 |
| exp_a | 8 | 0.162 | 0.203 | 0.391 | 0.336 | 0.947 |
| exp_b | 15 | 0.410 | 0.636 | 0.815 | 0.189 | 1.631 |

Two facts to read off this table before any interpretation:

- **exp_b's best α_kNN cell (0.41) is worse than baseline's median (0.19)** — but baseline's worst cell (0.78) lands inside exp_b's range, so the routings overlap. Routing is correlated with α_kNN MSE but does not cleanly partition cells by performance.
- **For ζ_kNN, the per-routing means are close** (baseline 0.55, exp_a 0.59, exp_b 0.71) and the within-routing spread is large (each routing has cells from ~0.2 up to >1.0). Routing carries little signal for ζ.

The full per-cell breakdown lives in [`../results/RUN_INVENTORY.md`](../results/RUN_INVENTORY.md).

---

## Representative cells used by the figures

Picked to control axes where possible — same backbone (ViT) and same target (EMA) and same loss (`vicreg_lam001`) wherever the dataset has it.

| routing | run_id | α_lin | ζ_lin | α_kNN | ζ_kNN |
|---|---|---|---|---|---|
| baseline | `baseline_vit_ema_vicreg_lam001_20260430_170646` | 0.006 | 0.068 | 0.015 | 0.102 |
| exp_a | `exp_a_vit_ema_vicreg_lam001_20260430_202310` | 0.070 | 0.217 | 0.162 | 0.807 |
| exp_b | `exp_b_vit_ema_vicreg_20260428_142657` | 0.716 | 0.889 | 0.751 | 1.067 |

All three are ViT + EMA. The exp_b cell falls back to the standard `vicreg` recipe because no `vit_ema_vicreg_lam001` exp_b cell was trained.

---

## Figure index

| File | What it shows |
|---|---|
| `routing_concepts.png` | 3-panel conceptual cartoon: what each routing asks the encoder to learn. No data. |
| `pca3d_panel.png` and `pca3d_<routing>_<param>.gif` ×6 | Top-3 PCs of the test-set features for the representative cell of each routing, colored by α (top row) and ζ (bottom row). The static panel is one viewpoint; the GIFs rotate (16-second revolution). For the baseline cell α-color separates along one direction; for the exp_b cell colors mix throughout. |
| `feature_spectrum.png` + `feature_spectrum_table.tsv` | Singular values of the test features per routing, log y-axis. Annotates participation ratio. The exp_b cell's singular values are ~10³× smaller in absolute magnitude than baseline's; participation ratio is 28.5 (exp_b) vs 10.0 (baseline). |
| `landscape_panel.png` and `landscape_<routing>_<param>.gif` ×6 | α (or ζ) plotted as a height surface over the top-2 PCs of the encoder's test features, with a translucent RBF surface fit. Geometric view of what the linear probe is fitting. |
| `raw_pca_panel.png` and `raw_pca_<routing>_<param>.{png,gif}` ×6 | PCA directly on the routing-sliced raw ctx fields (no encoder, no summary statistics). Honest analog of `pca3d_*` but on the input data instead of encoder features. **The empirical answer**: top-3 PCs of the raw input do *not* show α or ζ stratification for any routing — the parameter structure is in the data but not in its top variance directions, so any encoder that recovers α has to do so via lower-variance or non-linear combinations. |

For the spectrum figure, the singular values are dumped to a companion `.tsv` alongside the PNG.

---

## 2D companions and CSV exports

Every rotating 3D GIF has companions of the same basename:

| extension | what it is |
|---|---|
| `.gif` | rotating 3D scatter |
| `.png` | flattened 2D top-down scatter — same dots, drops one axis |
| `.csv` | underlying coordinates per snippet, plus the color value, for re-plotting |

CSV column conventions:

- `pca3d_*.csv`     — `snippet_idx, z_pc1, z_pc2, z_pc3, alpha` (or `zeta`) — encoder feature PCs
- `landscape_*.csv` — `snippet_idx, z_pc1, z_pc2, alpha` (or `zeta`) — encoder feature PCs + label
- `raw_pca_*.csv`   — `snippet_idx, raw_pc1, raw_pc2, raw_pc3, alpha` (or `zeta`) — raw input PCs
