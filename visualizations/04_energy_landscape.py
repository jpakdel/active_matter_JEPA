"""04_energy_landscape.py — α and ζ as a 3D height surface over the latent.

The "EBM-landscape" view: for each routing, plot the (PC1, PC2) of the encoder's
test features as the ground plane, then put the *physical parameter* (α or ζ)
on the vertical axis. Points are the actual test samples; the translucent
surface is a smooth fit of the parameter over the latent (RBF interpolation).

How to read it:
- Good encoder: the points lie close to a smooth, non-flat surface — the
  parameter is a continuous, recoverable function of the latent. The probe
  works because the encoder organized the latent so that the parameter varies
  smoothly across it.
- Collapsed encoder: the points scatter vertically with no surface structure;
  the smooth fit is forced to be near-flat (= predicting the mean). The probe
  fails because the encoder didn't put parameter info into the latent.

The lesson the existing PCA-color view (02) and the spectrum (03) hint at,
this view makes literal: it shows the function the linear probe is trying
to fit.

Output (per representative cell × per parameter):
    visualizations/outputs/landscape_<routing>_<alpha|zeta>.gif   - rotating 3D
    visualizations/outputs/landscape_<routing>_<alpha|zeta>.png   - flattened 2D
    visualizations/outputs/landscape_<routing>_<alpha|zeta>.csv   - raw coords
    visualizations/outputs/landscape_panel.png                    - 3-routing × 2-param static

Run from REFACTORED_CODEBASE/:
    python visualizations/04_energy_landscape.py

If a GIF already exists, it's reused. The 2D PNG and CSV are always re-rendered.
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
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path(__file__).resolve().parent.parent / "runs"

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
    return f["features"].numpy(), f["labels"][:, 0].numpy(), f["labels"][:, 1].numpy()


def fit_surface(pc12: np.ndarray, values: np.ndarray, n_grid: int = 40):
    """RBF interpolate `values` over the (PC1, PC2) plane and return a grid.

    Returns (XX, YY, ZZ) suitable for ax.plot_surface.
    """
    # Generate grid bounded by the data with a small margin.
    pad = 0.10
    x_min, x_max = pc12[:, 0].min(), pc12[:, 0].max()
    y_min, y_max = pc12[:, 1].min(), pc12[:, 1].max()
    x_pad = (x_max - x_min) * pad
    y_pad = (y_max - y_min) * pad
    xs = np.linspace(x_min - x_pad, x_max + x_pad, n_grid)
    ys = np.linspace(y_min - y_pad, y_max + y_pad, n_grid)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    # Thin-plate-spline RBF; smoothing keeps it from overfitting jagged scatter.
    rbf = RBFInterpolator(pc12, values, kernel="thin_plate_spline", smoothing=1.0)
    ZZ = rbf(grid_pts).reshape(XX.shape)
    return XX, YY, ZZ


def render_landscape(ax, pc12, values, ratios, cmap, value_label, title,
                     show_surface=True):
    """Draw a single (PC1, PC2, value) 3D scatter + smooth surface."""
    # Smooth surface (translucent)
    if show_surface:
        try:
            XX, YY, ZZ = fit_surface(pc12, values, n_grid=40)
            # Clip surface to data range so it doesn't blow up at the corners.
            zmin, zmax = values.min(), values.max()
            zspan = zmax - zmin
            ZZ = np.clip(ZZ, zmin - 0.3 * zspan, zmax + 0.3 * zspan)
            ax.plot_surface(
                XX, YY, ZZ, cmap=cmap, alpha=0.30, linewidth=0,
                antialiased=True, shade=True,
            )
        except Exception as e:
            print(f"    (surface fit failed: {e})")

    # Scatter the actual test samples
    sc = ax.scatter(
        pc12[:, 0], pc12[:, 1], values,
        c=values, cmap=cmap, s=34, edgecolors="k", linewidths=0.35, alpha=0.95,
    )
    ax.set_xlabel(f"PC1 ({ratios[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({ratios[1]*100:.1f}%)", fontsize=10)
    ax.set_zlabel(value_label, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.tick_params(labelsize=7)
    return sc


def render_landscape_gif(pc12, values, ratios, cmap, value_label, title_lines, out_path,
                         n_frames=240, fps=15, dpi=130, figsize=(7.5, 7.0)):
    """Rotating GIF of the (PC1, PC2, value) landscape."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    render_landscape(ax, pc12, values, ratios, cmap, value_label, "\n".join(title_lines))

    elev = 26

    def update(i):
        azim = (i / n_frames) * 360.0
        ax.view_init(elev=elev, azim=azim)
        return ()

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    ani.save(out_path, writer=writer, dpi=dpi)
    plt.close(fig)


def main():
    print("Loading representative cells and computing PC1-2 projections...")
    pc12 = {}
    alphas = {}
    zetas = {}
    ratios2d = {}
    for cell in CELLS:
        feats, alpha, zeta = load_test(cell.run_id)
        pca = PCA(n_components=2)
        proj = pca.fit_transform(feats)
        pc12[cell.routing] = proj
        alphas[cell.routing] = alpha
        zetas[cell.routing] = zeta
        ratios2d[cell.routing] = pca.explained_variance_ratio_
        print(f"  {cell.routing:9s}  feats {feats.shape}  "
              f"PC1-2 var: {ratios2d[cell.routing][0]:.3f}/{ratios2d[cell.routing][1]:.3f}")

    # ---- 1. Static 3-routing × 2-param panel --------------------------------
    print("\nRendering static 3×2 landscape panel...")
    fig = plt.figure(figsize=(15.5, 9.5))
    for col, cell in enumerate(CELLS):
        # α landscape
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        render_landscape(
            ax, pc12[cell.routing], alphas[cell.routing], ratios2d[cell.routing],
            cmap="viridis", value_label="α (active dipole)",
            title=f"{cell.routing}\n{cell.label}",
        )
        ax.view_init(elev=26, azim=35)
        # ζ landscape
        ax = fig.add_subplot(2, 3, 3 + col + 1, projection="3d")
        render_landscape(
            ax, pc12[cell.routing], zetas[cell.routing], ratios2d[cell.routing],
            cmap="plasma", value_label="ζ (steric)",
            title="(same encoder, ζ surface)",
        )
        ax.view_init(elev=26, azim=35)

    fig.suptitle(
        "Energy-landscape view: physical parameter as a height surface over (PC1, PC2)\n"
        "rows = probe target  ·  columns = routing  ·  surface = RBF fit, points = test samples",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Read the verticality of the points and the curvature of the surface: a clean smooth surface with "
        "points lying on it = the parameter is a recoverable function of the latent (probe wins). "
        "A near-flat surface with points scattered vertically = the encoder didn't put the parameter into the latent (probe fails).",
        ha="center", va="bottom", fontsize=9, style="italic", color="#52647a",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    panel_path = OUT / "landscape_panel.png"
    fig.savefig(panel_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {panel_path}")

    # ---- 2. Per-cell × per-parameter trio: rotating GIF + 2D PNG + CSV ------
    # The 2D PNG drops the height (which equals the color) and shows
    # (PC1, PC2) colored by α (or ζ). Same separability question as the GIF
    # but viewable in one frame.
    print("\nRendering rotating landscape GIFs + 2D companions + CSVs...")
    for cell in CELLS:
        proj = pc12[cell.routing]    # (N, 2)
        rats = ratios2d[cell.routing]
        for param, vals, cmap, label in [
            ("alpha", alphas[cell.routing], "viridis", "α (active dipole)"),
            ("zeta",  zetas[cell.routing],  "plasma",  "ζ (steric)"),
        ]:
            stem = f"landscape_{cell.routing}_{param}"
            gif_path = OUT / f"{stem}.gif"
            png_path = OUT / f"{stem}.png"
            csv_path = OUT / f"{stem}.csv"

            if gif_path.exists():
                print(f"  skip GIF (exists): {gif_path.name}")
            else:
                render_landscape_gif(
                    proj, vals, rats, cmap, label,
                    title_lines=[
                        f"{cell.routing}  ·  {param} as a height over (PC1, PC2)",
                        cell.label,
                    ],
                    out_path=gif_path,
                )
                print(f"  wrote GIF: {gif_path.name}")

            render_2d_companion(
                proj[:, 0], proj[:, 1], vals,
                cmap=cmap,
                x_label=f"z PC1 ({rats[0]*100:.1f}%)",
                y_label=f"z PC2 ({rats[1]*100:.1f}%)",
                color_label=label,
                title=(f"{cell.routing}  ·  flattened landscape  ·  colored by {param}\n"
                       f"{cell.label}"),
                out_path=png_path,
            )
            print(f"  wrote 2D : {png_path.name}")

            rows = [
                (i, float(proj[i, 0]), float(proj[i, 1]), float(vals[i]))
                for i in range(len(proj))
            ]
            write_scatter_csv(
                csv_path,
                ("snippet_idx", "z_pc1", "z_pc2", param),
                rows,
            )
            print(f"  wrote CSV: {csv_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
