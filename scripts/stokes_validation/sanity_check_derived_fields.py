"""Sanity check for ∇·D and Δu.

Run:
    python scripts/stokes_validation/sanity_check_derived_fields.py

What it does:
  1. Unit tests against analytical derivatives of sinusoids. This is the
     primary pass/fail check — it verifies the conv stencils are correct.
  2. Visualizes -div(D) vs Delta u on a real sample and reports the
     Pearson correlation. This is informational, not pass/fail: the Stokes
     relation Delta u = -alpha * div D + grad Pi has a pressure-gradient
     residual that breaks the pointwise linear correlation once the flow
     develops. We check after taking the curl (which kills grad Pi) as a
     cleaner physical test.

Outputs `outputs/stokes_diagnostic_alpha_-3.0.png` next to this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/stokes_validation/<this file>  ->  parents[2] = REFACTORED_CODEBASE
REFACTORED_ROOT = Path(__file__).resolve().parents[2]
# parents[3] = the project root (one level above REFACTORED_CODEBASE) where data/ lives
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REFACTORED_ROOT))

import math
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.well_dataset import WellDatasetForJEPA
from src.data.derived_fields import (
    divergence_D, laplacian_u,
    extract_D, extract_u,
    _d_dx, _d_dy,
)
from src.data.channel_map import ALPHA_IDX


DATA_DIR = PROJECT_ROOT / "data" / "active_matter"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 1. Analytical derivative tests
# ------------------------------------------------------------------

def test_laplacian_against_analytic():
    """Delta sin(kx+ly) = -(k^2 + l^2) sin(kx+ly). Check within central-diff
    truncation error on a dense grid."""
    H = W = 256
    L = 2 * math.pi                        # period
    hx = hy = L / H
    x = torch.linspace(0, L, H + 1)[:-1]
    y = torch.linspace(0, L, W + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")   # X varies along axis -2

    k = 3.0
    l = 2.0
    phi = torch.sin(k * X + l * Y)
    analytic = -(k ** 2 + l ** 2) * phi

    # u_channels shape (2, H, W), channel axis on -4? No: this is a 3D tensor
    # (C, H, W). We want (B=1, C=2, T=1, H, W) so laplacian_u can run.
    u = torch.stack([phi, phi], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 2, 1, H, W)
    out = laplacian_u(u, spacing=(hx, hy))        # (1, 2, 1, H, W)
    got = out[0, 0, 0]

    err = (got - analytic).abs().max().item()
    rel = err / analytic.abs().max().item()
    print(f"  laplacian_u: max abs err={err:.4e}, relative={rel:.4e}")
    assert rel < 1e-2, f"laplacian_u relative error {rel} too large"


def test_divergence_against_analytic():
    """Build D(x,y) such that div D has a known analytic form.

    Take D_11 = cos(kx), D_12 = 0, D_21 = 0, D_22 = sin(ly).
    Then (div D)_1 = d(D_11)/dx + d(D_12)/dy = -k sin(kx)
         (div D)_2 = d(D_21)/dx + d(D_22)/dy =  l cos(ly)
    """
    H = W = 256
    L = 2 * math.pi
    hx = hy = L / H
    x = torch.linspace(0, L, H + 1)[:-1]
    y = torch.linspace(0, L, W + 1)[:-1]
    X, Y = torch.meshgrid(x, y, indexing="ij")   # X varies along axis -2

    k = 4.0
    l = 3.0
    D11 = torch.cos(k * X)
    D12 = torch.zeros_like(D11)
    D21 = torch.zeros_like(D11)
    D22 = torch.sin(l * Y)
    D = torch.stack([D11, D12, D21, D22], dim=0).unsqueeze(0).unsqueeze(2)  # (1, 4, 1, H, W)

    analytic_1 = -k * torch.sin(k * X)
    analytic_2 = l * torch.cos(l * Y)

    out = divergence_D(D, spacing=(hx, hy))        # (1, 2, 1, H, W)
    got_1 = out[0, 0, 0]
    got_2 = out[0, 1, 0]

    err_1 = (got_1 - analytic_1).abs().max().item() / analytic_1.abs().max().item()
    err_2 = (got_2 - analytic_2).abs().max().item() / analytic_2.abs().max().item()
    print(f"  divergence_D: rel err (i=1)={err_1:.4e}, (i=2)={err_2:.4e}")
    assert err_1 < 1e-2 and err_2 < 1e-2, "divergence_D analytic check failed"


def run_unit_tests():
    print("\n[1] Analytical derivative unit tests")
    test_laplacian_against_analytic()
    test_divergence_against_analytic()
    print("  PASSED")


# ------------------------------------------------------------------
# 2. Stokes correlation on real data (informational)
# ------------------------------------------------------------------

L_PHYS = 10.0
N = 256
H_PHYS = L_PHYS / N


def find_indices_by_alpha(ds, alphas_wanted, prefer_late=False):
    picks = {}
    for idx, (fname, obj_id, t0) in enumerate(ds.index):
        pp = ds.physical_params_idx[fname]
        a = float(pp[ALPHA_IDX])
        if a not in alphas_wanted:
            continue
        if prefer_late:
            if a not in picks or t0 > picks[a][1]:
                picks[a] = (idx, t0)
        else:
            if a not in picks or t0 < picks[a][1]:
                picks[a] = (idx, t0)
    return {a: i for a, (i, _) in picks.items()}


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1) - a.mean()
    bf = b.reshape(-1) - b.mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-20
    return float(np.dot(af, bf) / denom)


def stokes_correlation_check():
    print("\n[2] Stokes correlation diagnostic (informational)")
    ds = WellDatasetForJEPA(data_dir=str(DATA_DIR), num_frames=16, split="train")

    alphas = [-1.0, -2.0, -3.0, -4.0, -5.0]
    picks = find_indices_by_alpha(ds, set(alphas), prefer_late=False)
    assert len(picks) == len(alphas)

    print(f"\n  {'alpha':>6}{'r(pointwise)':>16}{'r(curl)':>12}")
    print("  " + "-" * 36)

    for alpha in alphas:
        i = picks[alpha]
        sample = ds[i]
        ctx = sample["context"]

        div_D = divergence_D(extract_D(ctx), spacing=(H_PHYS, H_PHYS))   # (2, T, H, W)
        lap_u = laplacian_u(extract_u(ctx), spacing=(H_PHYS, H_PHYS))    # (2, T, H, W)

        # Raw pointwise correlation (polluted by pressure gradient)
        r_point = pearson(lap_u.numpy(), (-div_D).numpy())

        # Curl-based: take scalar curl of each vector field.
        # curl(F)_z = dF2/dx - dF1/dy
        curl_lap = _d_dx(lap_u.select(-4, 1), H_PHYS) - _d_dy(lap_u.select(-4, 0), H_PHYS)
        curl_divD = _d_dx(div_D.select(-4, 1), H_PHYS) - _d_dy(div_D.select(-4, 0), H_PHYS)
        # Stokes after curl: curl(Delta u) = -alpha * curl(div D)  (grad Pi killed)
        r_curl = pearson(curl_lap.numpy(), (-curl_divD).numpy())

        print(f"  {alpha:>6.1f}{r_point:>16.4f}{r_curl:>12.4f}")

    # ---- Visualize one sample ----
    alpha_plot = -3.0
    idx = picks[alpha_plot]
    sample = ds[idx]
    ctx = sample["context"]
    div_D = divergence_D(extract_D(ctx), spacing=(H_PHYS, H_PHYS))
    lap_u = laplacian_u(extract_u(ctx), spacing=(H_PHYS, H_PHYS))

    t = ctx.shape[1] // 2
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5))
    for row, name, ddi, lui in [
        (0, "i=1 (x)", (-div_D)[0, t].numpy(), lap_u[0, t].numpy()),
        (1, "i=2 (y)", (-div_D)[1, t].numpy(), lap_u[1, t].numpy()),
    ]:
        vmax = max(np.abs(ddi).max(), np.abs(lui).max())
        axes[row, 0].imshow(ddi, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 0].set_title(f"-div D  {name}", fontsize=9)
        axes[row, 1].imshow(lui, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row, 1].set_title(f"Delta u  {name}", fontsize=9)
        axes[row, 2].scatter(ddi.reshape(-1), lui.reshape(-1), s=0.4, alpha=0.25)
        xr = np.linspace(ddi.min(), ddi.max(), 50)
        axes[row, 2].plot(xr, alpha_plot * xr, "r-", lw=1, label=f"y={alpha_plot:.1f}x")
        axes[row, 2].set_xlabel("-div D"); axes[row, 2].set_ylabel("Delta u")
        axes[row, 2].legend(fontsize=8)
        for j in (0, 1):
            axes[row, j].set_xticks([]); axes[row, j].set_yticks([])

    fig.suptitle(f"Stokes correlation diagnostic  |  alpha={alpha_plot:.1f}", fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"stokes_diagnostic_alpha_{alpha_plot:.1f}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\n  saved: {out}")


def run():
    torch.manual_seed(0)
    run_unit_tests()
    stokes_correlation_check()
    print("\nDONE")


if __name__ == "__main__":
    run()
