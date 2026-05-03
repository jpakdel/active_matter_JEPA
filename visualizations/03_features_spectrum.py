"""03_features_spectrum.py — singular-value spectrum of frozen test features.

Quantitative backup for the visual collapse seen in 02's PCA scatter. For
each representative cell, plot σ_i of the test-feature matrix on a log y-axis
and annotate the participation ratio (effective rank) of the features.

Participation ratio (PR) = (Σ λᵢ)² / (Σ λᵢ²)  where λᵢ = σᵢ²
A perfectly white representation has PR = D (the embed dim). A 1D collapsed
representation has PR ≈ 1. Real encoders sit somewhere in between.

Output:
    visualizations/outputs/feature_spectrum.png
    visualizations/outputs/feature_spectrum_table.tsv

Run from REFACTORED_CODEBASE/:
    python visualizations/03_features_spectrum.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = Path(__file__).resolve().parent.parent / "runs"


@dataclass
class Cell:
    routing: str
    run_id: str
    color: str  # matplotlib color


CELLS: List[Cell] = [
    Cell("baseline", "baseline_vit_ema_vicreg_lam001_20260430_170646", "#1f77b4"),
    Cell("exp_a",    "exp_a_vit_ema_vicreg_lam001_20260430_202310",    "#2ca02c"),
    Cell("exp_b",    "exp_b_vit_ema_vicreg_20260428_142657",            "#d62728"),
]


def participation_ratio(eigvals: np.ndarray) -> float:
    """Effective rank of a covariance spectrum.

    PR = (Σ λᵢ)² / (Σ λᵢ²). Equals D for an isotropic / white covariance,
    1 for a rank-1 covariance.
    """
    eigvals = np.asarray(eigvals, dtype=np.float64)
    return float(eigvals.sum() ** 2 / np.maximum((eigvals ** 2).sum(), 1e-30))


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.6))

    rows = []  # for the TSV
    for cell in CELLS:
        f = torch.load(RUNS / cell.run_id / "features" / "test.pt", weights_only=False)
        feats = f["features"].numpy()  # (N, D)
        # Center, then SVD. Singular values of (N, D)-centered features.
        feats_c = feats - feats.mean(axis=0, keepdims=True)
        s = np.linalg.svd(feats_c, compute_uv=False)
        # Sample-covariance eigenvalues = s^2 / (N-1)
        eig = s ** 2 / max(len(feats_c) - 1, 1)
        pr = participation_ratio(eig)
        # Cumulative variance fraction
        cum = np.cumsum(eig) / eig.sum()
        # Number of components for 95% variance
        n95 = int(np.searchsorted(cum, 0.95)) + 1

        ax.plot(
            np.arange(1, len(s) + 1), s,
            label=f"{cell.routing}   PR={pr:.1f}, 95%-rank={n95}",
            color=cell.color, linewidth=2.0, alpha=0.9,
        )
        rows.append((cell.routing, cell.run_id, len(feats), feats.shape[1], pr, n95))
        print(f"  {cell.routing:9s}  N={len(feats):3d}  D={feats.shape[1]:3d}  "
              f"PR={pr:6.2f}  95%-rank={n95:3d}")

    ax.set_yscale("log")
    ax.set_xlabel("singular-value index", fontsize=11)
    ax.set_ylabel("singular value (log)", fontsize=11)
    ax.set_title(
        "Magnitude collapse with routing\n"
        "Test-feature singular values per routing  ·  PR = participation ratio (effective rank)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper right", frameon=True)
    ax.grid(True, which="both", alpha=0.3)

    fig.text(
        0.5, 0.005,
        "Read the absolute scale: exp_b's largest singular value is ~1000× smaller than baseline's smallest. The encoder outputs are nearly\n"
        "constant on test inputs. The PR is high (28 vs baseline's 10) because among those tiny variations, no single direction dominates —\n"
        "but absolute magnitude is collapsed everywhere. Either way: the residual structure is uninformative for α and ζ probes.",
        ha="center", va="bottom", fontsize=8.5, style="italic", color="#52647a",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1.0])

    out_path = OUT / "feature_spectrum.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out_path}")

    tsv_path = OUT / "feature_spectrum_table.tsv"
    with open(tsv_path, "w") as fh:
        fh.write("routing\trun_id\tN\tD\tparticipation_ratio\tn_95_variance\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"wrote {tsv_path}")


if __name__ == "__main__":
    main()
