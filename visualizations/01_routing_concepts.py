"""01_routing_concepts.py — what each routing demands of the encoder.

A 3-panel conceptual cartoon (no data) showing the field-to-field mapping
each routing presents to the JEPA objective, and a one-line caption naming
what the encoder must learn to satisfy that mapping.

Output:
    visualizations/outputs/routing_concepts.png

Run from REFACTORED_CODEBASE/:
    python visualizations/01_routing_concepts.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def field_box(ax, x, y, w, h, label, sublabel="", color="#cfe2ff", role=""):
    """Draw a labeled box representing a field. `role` is the small caption above ('context', 'target', etc)."""
    if role:
        ax.text(x + w / 2, y + h + 0.015, role, ha="center", va="bottom",
                fontsize=8, color="#52647a")
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.5, edgecolor="#33415c", facecolor=color,
    )
    ax.add_patch(box)
    if sublabel:
        ax.text(x + w / 2, y + h / 2 + 0.025, label,
                ha="center", va="center", fontsize=13, fontweight="bold")
        ax.text(x + w / 2, y + h / 2 - 0.030, sublabel,
                ha="center", va="center", fontsize=8, style="italic", color="#52647a")
    else:
        ax.text(x + w / 2, y + h / 2, label,
                ha="center", va="center", fontsize=13, fontweight="bold")


def encoder_arrow(ax, x0, x1, y, label, label_color="#1f2937"):
    arrow = FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
        linewidth=1.8, color="#33415c",
    )
    ax.add_patch(arrow)
    ax.text((x0 + x1) / 2, y + 0.022, label,
            ha="center", va="bottom", fontsize=8.5, color=label_color)


def latent_disk(ax, cx, cy, label, sublabel, color):
    circle = plt.Circle((cx, cy), 0.085, color=color, ec="#33415c", lw=1.5, zorder=3)
    ax.add_patch(circle)
    ax.text(cx, cy + 0.018, label, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    ax.text(cx, cy - 0.030, sublabel, ha="center", va="center",
            fontsize=8, style="italic", zorder=4, color="#1a1a1a")


def panel(ax, title, ctx_label, ctx_sub, tgt_label, tgt_sub,
          encoder_caption, latent_label, latent_sub, latent_dim,
          punchline, latent_color):
    """Layout (axes coords):

         y=0.93  ─── title
         y=0.78  ─── [ctx] ─encoder caption→ [tgt]
                       │                       ↕
                     encoder              JEPA loss
                       ↓                       ↕
         y=0.50  ──── (z) ─predictor→  [tgt*]
         y=0.34  ─── "what the latent looks like" line
         y=0.18  ─── ┌────punchline box────┐
    """
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(0.5, 0.96, title, ha="center", va="top", fontsize=14, fontweight="bold")

    # ---- top row: ctx, mapping, tgt at y=0.74-0.83 ----
    field_box(ax, 0.04, 0.74, 0.28, 0.10, ctx_label, ctx_sub, color="#cfe2ff", role="context")
    encoder_arrow(ax, 0.34, 0.62, 0.79, encoder_caption, label_color="#7a3a3a")
    field_box(ax, 0.64, 0.74, 0.28, 0.10, tgt_label, tgt_sub, color="#fce5cf", role="target")

    # ---- bottom row: z, predictor, predicted-tgt at y=0.46-0.56 ----
    # Down arrow ctx -> z
    ax.annotate("", xy=(0.18, 0.55), xytext=(0.18, 0.72),
                arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.5))
    ax.text(0.21, 0.635, "encoder", ha="left", va="center", fontsize=8.5, color="#1a4480")

    # latent disk
    latent_disk(ax, 0.18, 0.50, latent_label, latent_sub, latent_color)

    # z -> predicted target
    ax.annotate("", xy=(0.62, 0.50), xytext=(0.28, 0.50),
                arrowprops=dict(arrowstyle="-|>", color="#33415c", lw=1.5))
    ax.text(0.45, 0.525, "predictor", ha="center", va="bottom", fontsize=8.5, color="#1a4480")

    field_box(ax, 0.64, 0.45, 0.28, 0.10, tgt_label + "*", "", color="#ffe9b3", role="predictor output")

    # JEPA loss double-arrow between tgt (top) and tgt* (bottom)
    ax.annotate("", xy=(0.78, 0.73), xytext=(0.78, 0.56),
                arrowprops=dict(arrowstyle="<->", color="#7a3a3a", lw=1.3, linestyle="dashed"))
    ax.text(0.96, 0.645, "JEPA\nloss", ha="right", va="center", fontsize=8, color="#7a3a3a")

    # ---- "latent dim" annotation centered between encoder pipeline and punchline ----
    ax.text(0.5, 0.355, latent_dim, ha="center", va="center",
            fontsize=8.5, color="#7a3a3a", style="italic")

    # ---- punchline ----
    ax.text(0.5, 0.135, punchline, ha="center", va="center",
            fontsize=9, color="#1a1a1a",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff8d8", edgecolor="#7a3a3a", lw=1))


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    panel(
        axes[0],
        title="baseline:  ctx ≈ tgt",
        ctx_label="all 11ch", ctx_sub="t = 0..7",
        tgt_label="all 11ch", tgt_sub="t = 8..15",
        encoder_caption="≈ identity over time",
        latent_label="z", latent_sub="rich",
        latent_dim="z carries full state.\nVariance concentrated\non α, ζ, field modes.",
        punchline="Encoder must preserve the state.\nα and ζ both surface in the top PCs.",
        latent_color="#a8d8ea",
    )

    panel(
        axes[1],
        title="exp_a:  D → u",
        ctx_label="D", ctx_sub="dipole orient.",
        tgt_label="u", tgt_sub="velocity",
        encoder_caption="invert Stokes:  u = α·K∗D",
        latent_label="z", latent_sub="α × field",
        latent_dim="z must encode field +\nthe scalar α that scales it.",
        punchline="α is multiplicatively coupled\nto the field — encoder forced to encode both.",
        latent_color="#bce8b6",
    )

    panel(
        axes[2],
        title="exp_b:  ∇·D → Δu",
        ctx_label="∇·D", ctx_sub="div. of dipole",
        tgt_label="Δu", tgt_sub="lap. of velocity",
        encoder_caption="local linear:  Δu ≈ −α · ∇·D",
        latent_label="z", latent_sub="small magnitude",
        latent_dim="On the chosen cell:\nσ_i ~10³× smaller than baseline.\nα and ζ probes both fail.",
        punchline="Empirically the worst routing on α-kNN\nin every cell tested\n(min 0.41 across 15 exp_b cells\nvs baseline best 0.013).",
        latent_color="#f3a6a6",
    )

    fig.suptitle(
        "What each channel routing asks the encoder to learn",
        fontsize=15, fontweight="bold", y=1.00,
    )
    fig.text(
        0.5, 0.005,
        "The active-matter Stokes equation gives Δu ≈ −α·∇·D up to geometric factors. "
        "Per-routing α-kNN ranges across our 34 trained cells: baseline [0.013, 0.78]; exp_a [0.16, 0.39]; exp_b [0.41, 0.82]. "
        "See results/RUN_INVENTORY.md for the full table.",
        ha="center", va="bottom", fontsize=8, style="italic", color="#52647a", wrap=True,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path = OUT / "routing_concepts.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
