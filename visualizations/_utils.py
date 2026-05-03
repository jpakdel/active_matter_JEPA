"""Shared helpers for visualization scripts.

Every 3D scatter rendered as a rotating GIF should have two companion files
of the same basename:

    <basename>.gif   - rotating 3D view (original)
    <basename>.png   - flattened 2D top-down scatter, same coloring; lets the
                       reader judge separability without watching the rotation
    <basename>.csv   - the underlying coordinates for re-plotting / analysis

The 2D companion is "informationally equivalent" to the GIF in the sense that
it shows the same dots with the same colors — just dropping one axis. If the
colors stratify cleanly in 2D, that's the linear-probe success signal in
geometric form.

These helpers also include a paired (side-by-side) variant for script 06's
task-vs-encoder figures.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


def render_2d_companion(
    x: np.ndarray,
    y: np.ndarray,
    color: np.ndarray,
    *,
    cmap: str,
    x_label: str,
    y_label: str,
    color_label: str,
    title: str,
    out_path: Path,
    figsize=(7.5, 6.5),
    dpi=180,
):
    """Single 2D scatter colored by `color`."""
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        x, y, c=color, cmap=cmap, s=64, edgecolors="k", linewidths=0.5, alpha=0.92,
    )
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_label, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_2d_companion_paired(
    left:  dict,   # keys: x, y, color, cmap, x_label, y_label, color_label, title
    right: dict,   # same keys
    *,
    suptitle: str,
    out_path: Path,
    figsize=(13.0, 6.0),
    dpi=180,
):
    """Two 2D scatters side by side. Used by script 06 (task vs encoder)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, panel in zip(axes, [left, right]):
        sc = ax.scatter(
            panel["x"], panel["y"], c=panel["color"],
            cmap=panel["cmap"], s=60, edgecolors="k", linewidths=0.5, alpha=0.92,
        )
        ax.set_xlabel(panel["x_label"], fontsize=11)
        ax.set_ylabel(panel["y_label"], fontsize=11)
        ax.set_title(panel["title"], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(panel["color_label"], fontsize=11)
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_scatter_csv(out_path: Path, columns: Sequence[str], rows: Iterable):
    """Write a small CSV file with given column headers and row tuples.

    Each row's values can be Python floats / ints / strings — they're written
    via csv.writer, which calls str(...) on each cell.
    """
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(columns))
        for row in rows:
            w.writerow(row)
