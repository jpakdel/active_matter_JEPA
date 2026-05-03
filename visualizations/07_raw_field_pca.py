"""07_raw_field_pca.py — PCA directly on raw test fields per routing.

The honest analog of script 02 (PCA on encoder features) applied to the
*raw simulation fields* themselves. NO summary statistics, NO CCA, NO
neural net. Just: take the routing's ctx fields, flatten, PCA to 3 dims,
scatter colored by α (and by ζ).

What this tests: is α (or ζ) linearly recoverable from the top variance
directions of the raw input data, before any encoder touches it? If yes,
the data has the structure — the encoder just has to preserve it. If no,
no linear encoder operating on this input can recover it via its top
variance directions.

This is the most direct possible "what's in the data" view. The only
preprocessing is a 256→64 spatial downsample (for tractability) inside
the dataset loader. All channel and frame structure is preserved.

Output (per routing × per parameter):
    visualizations/outputs/raw_pca_<routing>_<alpha|zeta>.gif   - rotating 3D
    visualizations/outputs/raw_pca_<routing>_<alpha|zeta>.png   - flattened 2D
    visualizations/outputs/raw_pca_<routing>_<alpha|zeta>.csv   - raw coords
    visualizations/outputs/raw_pca_panel.png                    - 3-routing × 2-color static

Run from REFACTORED_CODEBASE/:
    python visualizations/07_raw_field_pca.py

If a GIF already exists in outputs/, it's reused (saves ~2 min/GIF).
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

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.data.well_dataset import WellDatasetForJEPA
from src.train.builders import select_channels

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _utils import render_2d_companion, write_scatter_csv  # noqa: E402


# ---- routings (same as scripts 05/06) ---------------------------------------

@dataclass
class Routing:
    name: str
    ctx_spec: str
    color: str  # for plot accents


ROUTINGS: List[Routing] = [
    Routing("baseline", "all",  "#1f77b4"),
    Routing("exp_a",    "D",    "#2ca02c"),
    Routing("exp_b",    "divD", "#d62728"),
]


# ---- data loading -----------------------------------------------------------

def load_test_dataset(num_frames: int = 16, resolution=(64, 64)):
    """Iterate the test split, return raw (ctx, tgt) tensors and (α, ζ) labels.

    Returns:
        ctx_all: (104, 11, T, H, W) tensor — raw ctx fields, all 11 channels
        tgt_all: (104, 11, T, H, W) tensor — raw tgt fields, all 11 channels
        params:  (104, 2) tensor — (α, ζ) per snippet
    """
    data_dir = PROJECT.parent / "data" / "active_matter"
    ds = WellDatasetForJEPA(
        data_dir=str(data_dir), num_frames=num_frames, split="test",
        resolution=resolution,
    )
    n = len(ds)
    print(f"  loading {n} test snippets at {resolution[0]}x{resolution[1]} resolution...")
    ctxs, tgts, params = [], [], []
    for i in range(n):
        item = ds[i]
        ctxs.append(item["context"])
        tgts.append(item["target"])
        params.append(item["physical_params"])
    return torch.stack(ctxs), torch.stack(tgts), torch.stack(params).float()


def routing_raw_flat(full_ctx: torch.Tensor, ctx_spec: str) -> np.ndarray:
    """Slice channels, then flatten each snippet to a single long vector.

    Returns: (N, D) array where D = C_routing * T * H * W.
    For baseline ctx_spec='all' at 16f×64×64: D = 11 * 16 * 64 * 64 = 720,896.
    For exp_a   ctx_spec='D':                 D =  4 * 16 * 64 * 64 = 262,144.
    For exp_b   ctx_spec='divD':              D =  2 * 16 * 64 * 64 = 131,072.
    """
    ctx_r = select_channels(full_ctx, ctx_spec)         # (N, C, T, H, W)
    flat = ctx_r.reshape(ctx_r.shape[0], -1).numpy()    # (N, C*T*H*W)
    return flat


# ---- PCA + render -----------------------------------------------------------

def pca_3d_on_raw(flat: np.ndarray):
    """Center per-feature, then PCA to 3 components.

    With N=104 samples and D huge (~700k), sklearn picks randomized SVD
    automatically. Returns (N, 3) projection + (3,) explained variance ratios.
    """
    # Center each feature — standard PCA preprocessing. Don't divide by std
    # (raw pixel intensities have different physical units across channels;
    # variance scaling would inappropriately equalize them).
    flat_c = flat - flat.mean(axis=0, keepdims=True)
    pca = PCA(n_components=3, svd_solver="auto", random_state=0)
    proj = pca.fit_transform(flat_c)
    return proj, pca.explained_variance_ratio_


def render_gif(proj, color, cmap, ratios, cbar_label, title_lines, out_path,
               n_frames=240, fps=15, dpi=130, figsize=(7.5, 7.0)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        proj[:, 0], proj[:, 1], proj[:, 2],
        c=color, cmap=cmap, s=32, edgecolors="k", linewidths=0.35, alpha=0.92,
    )
    ax.set_xlabel(f"raw PC1 ({ratios[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"raw PC2 ({ratios[1]*100:.1f}%)", fontsize=11)
    ax.set_zlabel(f"raw PC3 ({ratios[2]*100:.1f}%)", fontsize=11)
    ax.set_title("\n".join(title_lines), fontsize=12)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.10)
    cbar.set_label(cbar_label, fontsize=11)

    elev = 22

    def update(i):
        ax.view_init(elev=elev, azim=(i / n_frames) * 360.0)
        return (sc,)

    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)
    ani.save(out_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)


def render_static_panel(routings_data, alphas, zetas, out_path):
    """3-routing × 2-color static panel."""
    fig = plt.figure(figsize=(15.5, 9.5))
    for col, (routing, color, proj, ratios) in enumerate(routings_data):
        # Top row: α
        ax = fig.add_subplot(2, 3, col + 1, projection="3d")
        sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                        c=alphas, cmap="viridis",
                        s=32, edgecolors="k", linewidths=0.3, alpha=0.92)
        ax.set_xlabel(f"raw PC1 ({ratios[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"raw PC2 ({ratios[1]*100:.1f}%)", fontsize=9)
        ax.set_zlabel(f"raw PC3 ({ratios[2]*100:.1f}%)", fontsize=9)
        ax.set_title(f"{routing}  ·  raw ctx PCA  ·  colored by α", fontsize=10, color=color)
        ax.view_init(elev=22, azim=35)
        ax.tick_params(labelsize=7)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
        cbar.set_label("α", fontsize=9)

        # Bottom row: ζ
        ax = fig.add_subplot(2, 3, 3 + col + 1, projection="3d")
        sc = ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2],
                        c=zetas, cmap="plasma",
                        s=32, edgecolors="k", linewidths=0.3, alpha=0.92)
        ax.set_xlabel(f"raw PC1 ({ratios[0]*100:.1f}%)", fontsize=9)
        ax.set_ylabel(f"raw PC2 ({ratios[1]*100:.1f}%)", fontsize=9)
        ax.set_zlabel(f"raw PC3 ({ratios[2]*100:.1f}%)", fontsize=9)
        ax.set_title("(same projection, colored by ζ)", fontsize=10)
        ax.view_init(elev=22, azim=35)
        ax.tick_params(labelsize=7)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.10)
        cbar.set_label("ζ", fontsize=9)

    fig.suptitle(
        "PCA on the raw routing-sliced ctx fields  ·  no encoder, no summary statistics\n"
        "Same projection per column, colored by α (top) and ζ (bottom)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Each dot = one test snippet. Coordinates = top-3 PCs of the FLATTENED raw ctx field for that routing. "
        "If α (or ζ) colors stratify cleanly, the parameter is linearly recoverable from the input's top variance directions — "
        "no encoder needed. If colors mix, the parameter isn't carried by raw input variance and any encoder must work harder.",
        ha="center", va="bottom", fontsize=9, style="italic", color="#52647a",
    )
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- main -------------------------------------------------------------------

def main():
    print("Loading raw test dataset (one pass for all routings)...")
    full_ctx, _full_tgt, params = load_test_dataset()
    alphas = params[:, 0].numpy()
    zetas  = params[:, 1].numpy()

    print("\nComputing per-routing PCA on raw ctx fields...")
    routings_data = []
    for r in ROUTINGS:
        flat = routing_raw_flat(full_ctx, r.ctx_spec)
        print(f"  {r.name:9s}  flat ctx shape: {flat.shape}  ({flat.nbytes / 1e6:.1f} MB)")
        proj, ratios = pca_3d_on_raw(flat)
        routings_data.append((r.name, r.color, proj, ratios))
        print(f"    PC1-3 var: {ratios[0]:.3f} / {ratios[1]:.3f} / {ratios[2]:.3f}  "
              f"(sum {ratios.sum():.3f})")

    print("\nRendering static 2-row × 3-col panel...")
    render_static_panel(routings_data, alphas, zetas, OUT / "raw_pca_panel.png")

    print("\nRendering rotating GIFs + 2D companions + CSVs (3 routings × 2 metrics)...")
    for routing, color, proj, ratios in routings_data:
        for label_arr, label_name, cmap, suffix in [
            (alphas, "α (active dipole)", "viridis", "alpha"),
            (zetas,  "ζ (steric)",        "plasma",  "zeta"),
        ]:
            stem = f"raw_pca_{routing}_{suffix}"
            gif_path = OUT / f"{stem}.gif"
            png_path = OUT / f"{stem}.png"
            csv_path = OUT / f"{stem}.csv"

            if gif_path.exists():
                print(f"  skip GIF (exists): {gif_path.name}")
            else:
                render_gif(
                    proj, label_arr, cmap, ratios,
                    cbar_label=label_name,
                    title_lines=[
                        f"{routing}  ·  PCA on raw ctx fields  ·  colored by {suffix}",
                        f"no encoder, no summary statistics",
                    ],
                    out_path=gif_path,
                )
                print(f"  wrote GIF: {gif_path.name}")

            render_2d_companion(
                proj[:, 0], proj[:, 1], label_arr,
                cmap=cmap,
                x_label=f"raw PC1 ({ratios[0]*100:.1f}%)",
                y_label=f"raw PC2 ({ratios[1]*100:.1f}%)",
                color_label=label_name,
                title=(f"{routing}  ·  raw ctx PCA top-down  ·  colored by {suffix}\n"
                       f"flattened (PC1, PC2), drops PC3"),
                out_path=png_path,
            )
            print(f"  wrote 2D : {png_path.name}")

            rows = [
                (i, float(proj[i, 0]), float(proj[i, 1]), float(proj[i, 2]), float(label_arr[i]))
                for i in range(len(proj))
            ]
            write_scatter_csv(
                csv_path,
                ("snippet_idx", "raw_pc1", "raw_pc2", "raw_pc3", suffix),
                rows,
            )
            print(f"  wrote CSV: {csv_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
