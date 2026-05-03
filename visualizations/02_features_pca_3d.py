"""02_features_pca_3d.py — 3D PCA scatter of test-set encoder features.

For each representative cell, project the test-set features to 3D PCA, then
scatter colored by alpha (one figure) and by zeta (another). Saved as a
slowly-rotating GIF for slides AND a static 3-routing × 2-color panel for
the paper.

This is the hero figure: when the GIF spins, exp_b's "ζ smear" is impossible
to miss (the encoder didn't encode ζ, so coloring by ζ shows random structure).
The α coloring on exp_b shows a thin ridge — the 1D collapse the conceptual
figure predicted.

Output (per representative cell × per parameter):
    visualizations/outputs/pca3d_<routing>_<alpha|zeta>.gif   - rotating 3D
    visualizations/outputs/pca3d_<routing>_<alpha|zeta>.png   - flattened 2D
    visualizations/outputs/pca3d_<routing>_<alpha|zeta>.csv   - raw coords
    visualizations/outputs/pca3d_panel.png                    - 3-routing × 2-color static

Run from REFACTORED_CODEBASE/:
    python visualizations/02_features_pca_3d.py

If a GIF already exists in outputs/, it is reused (saves ~2 min/GIF). The 2D
PNG and CSV companions are always re-rendered. Delete the GIF to force regen.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path(__file__).resolve().parent.parent / "runs"

# Local helper module (sibling)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import render_2d_companion, write_scatter_csv  # noqa: E402


@dataclass
class Cell:
    routing: str
    run_id: str
    label: str  # short caption


CELLS: List[Cell] = [
    Cell("baseline", "baseline_vit_ema_vicreg_lam001_20260430_170646",
         "ViT + EMA + vicreg_lam001\nα_kNN = 0.015 / ζ_kNN = 0.102"),
    Cell("exp_a",    "exp_a_vit_ema_vicreg_lam001_20260430_202310",
         "ViT + EMA + vicreg_lam001\nα_kNN = 0.162 / ζ_kNN = 0.807"),
    Cell("exp_b",    "exp_b_vit_ema_vicreg_20260428_142657",
         "ViT + EMA + vicreg\nα_kNN = 0.751 / ζ_kNN = 1.067"),
]


def load_test(run_id: str):
    f = torch.load(RUNS / run_id / "features" / "test.pt", weights_only=False)
    feats = f["features"].numpy()
    alpha = f["labels"][:, 0].numpy()
    zeta = f["labels"][:, 1].numpy()
    return feats, alpha, zeta


def project_3d(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA-3D projection + the variance ratios of the first 3 components."""
    pca = PCA(n_components=3)
    proj = pca.fit_transform(feats)
    return proj, pca.explained_variance_ratio_


def static_scatter(ax, proj, color, title, cmap, ratios, cbar_label):
    """Render one 3D scatter onto an existing axes."""
    sc = ax.scatter(
        proj[:, 0], proj[:, 1], proj[:, 2],
        c=color, cmap=cmap, s=22, edgecolors="k", linewidths=0.3, alpha=0.92,
    )
    ax.set_xlabel(f"PC1 ({ratios[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ratios[1]*100:.1f}%)", fontsize=9)
    ax.set_zlabel(f"PC3 ({ratios[2]*100:.1f}%)", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=7)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
    cbar.set_label(cbar_label, fontsize=9)
    cbar.ax.tick_params(labelsize=7)
    # Pick a good default angle that shows 3D structure clearly.
    ax.view_init(elev=22, azim=35)
    return sc


def render_gif(proj, color, cmap, ratios, cbar_label, title_lines, out_path,
               n_frames=240, fps=15, dpi=130, figsize=(7.5, 7.0)):
    """Render a slowly-rotating 3D scatter as a GIF.

    Defaults: 240 frames at 15 fps = 16 s for one full revolution (22.5°/s).
    That's 1/4 the angular speed and 4x the duration of the original render.
    1.5° per frame at 15 fps is smooth-feeling without burning enormous file size.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        proj[:, 0], proj[:, 1], proj[:, 2],
        c=color, cmap=cmap, s=32, edgecolors="k", linewidths=0.35, alpha=0.92,
    )
    ax.set_xlabel(f"PC1 ({ratios[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({ratios[1]*100:.1f}%)", fontsize=11)
    ax.set_zlabel(f"PC3 ({ratios[2]*100:.1f}%)", fontsize=11)
    ax.set_title("\n".join(title_lines), fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label(cbar_label, fontsize=11)

    # Constant elevation, sweeping azimuth.
    elev = 22

    def update(i):
        azim = (i / n_frames) * 360.0
        ax.view_init(elev=elev, azim=azim)
        return (sc,)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main():
    print("Loading representative cells and computing PCA projections...")
    projections = {}
    labels_alpha = {}
    labels_zeta = {}
    ratios = {}
    for cell in CELLS:
        feats, alpha, zeta = load_test(cell.run_id)
        proj, r = project_3d(feats)
        projections[cell.routing] = proj
        labels_alpha[cell.routing] = alpha
        labels_zeta[cell.routing] = zeta
        ratios[cell.routing] = r
        print(f"  {cell.routing:9s}  feats {feats.shape}  PC1-3 var: "
              f"{r[0]:.3f}/{r[1]:.3f}/{r[2]:.3f}  (sum {r.sum():.3f})")

    # ---- 1. The static 3×2 panel ---------------------------------------------
    print("\nRendering static 3-routing × 2-color panel...")
    fig = plt.figure(figsize=(15, 9))
    for col, cell in enumerate(CELLS):
        # Top row: colored by alpha
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        static_scatter(
            ax, projections[cell.routing], labels_alpha[cell.routing],
            f"{cell.routing}\n{cell.label}",
            cmap="viridis", ratios=ratios[cell.routing],
            cbar_label="α (active dipole)",
        )
        # Bottom row: colored by zeta
        ax = fig.add_subplot(2, 3, 3 + col + 1, projection="3d")
        static_scatter(
            ax, projections[cell.routing], labels_zeta[cell.routing],
            "(same encoder, colored by ζ)",
            cmap="plasma", ratios=ratios[cell.routing],
            cbar_label="ζ (steric)",
        )
    fig.suptitle(
        "PCA-3D of frozen-encoder test-set features  ·  rows = probe target  ·  columns = routing",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Read across each row: as routing degrades the encoder (baseline → exp_a → exp_b), "
        "the colored gradient becomes harder to see — that's the linear/kNN probe failing in pictures. "
        "Bottom-right (exp_b ζ) is essentially noise: the encoder discarded ζ entirely.",
        ha="center", va="bottom", fontsize=9, style="italic", color="#52647a",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    panel_path = OUT / "pca3d_panel.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {panel_path}")

    # ---- 2. Per-cell × per-parameter trio: rotating GIF + 2D PNG + CSV ------
    # The 2D PNG drops the PC3 axis (top-down view onto PC1 vs PC2) but keeps
    # the same coloring — informationally equivalent for the linear-probe
    # separability question. The CSV stores all three PCs + the color value
    # so downstream tools can re-plot or re-analyse.
    print("\nRendering rotating GIFs + 2D companions + CSVs...")
    for cell in CELLS:
        proj = projections[cell.routing]   # (N, 3)
        rats = ratios[cell.routing]
        for param, color, cmap, label in [
            ("alpha", labels_alpha[cell.routing], "viridis", "α (active dipole)"),
            ("zeta",  labels_zeta[cell.routing],  "plasma",  "ζ (steric)"),
        ]:
            stem = f"pca3d_{cell.routing}_{param}"
            gif_path = OUT / f"{stem}.gif"
            png_path = OUT / f"{stem}.png"
            csv_path = OUT / f"{stem}.csv"

            # GIF (skip if exists — they take ~2 min each)
            if gif_path.exists():
                print(f"  skip GIF (exists): {gif_path.name}")
            else:
                render_gif(
                    proj, color, cmap, rats,
                    cbar_label=label,
                    title_lines=[
                        f"{cell.routing}  ·  PCA-3D of test features  ·  colored by {param}",
                        cell.label,
                    ],
                    out_path=gif_path,
                )
                print(f"  wrote GIF: {gif_path.name}")

            # 2D companion (always re-render — fast)
            render_2d_companion(
                proj[:, 0], proj[:, 1], color,
                cmap=cmap,
                x_label=f"z PC1 ({rats[0]*100:.1f}%)",
                y_label=f"z PC2 ({rats[1]*100:.1f}%)",
                color_label=label,
                title=(f"{cell.routing}  ·  PCA-2D top-down view  ·  colored by {param}\n"
                       f"{cell.label}"),
                out_path=png_path,
            )
            print(f"  wrote 2D : {png_path.name}")

            # CSV: snippet_idx, pc1, pc2, pc3, color_value
            rows = [
                (i, float(proj[i, 0]), float(proj[i, 1]), float(proj[i, 2]), float(color[i]))
                for i in range(len(proj))
            ]
            write_scatter_csv(
                csv_path,
                ("snippet_idx", "z_pc1", "z_pc2", "z_pc3", param),
                rows,
            )
            print(f"  wrote CSV: {csv_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
