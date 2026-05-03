"""Multi-metric Stokes-relation correlation diagnostic.

For each target alpha, computes and reports:
  (1) Per-component Pearson r between Delta u_i and (-div D)_i, i in {1,2}
  (2) Per-timestep Pearson r averaged over the 16 frames, plus std
  (3) Spearman rank correlation (robust to outliers and non-linear monotonic
      relationships) pooled over all pixels
  (4) Sample-level OLS slope (no intercept) + R^2, per component
  (5) Patch-averaged Pearson r at multiple scales k in {1,2,4,8,16,32,64} —
      reveals whether the Stokes relation holds better at coarse vs fine
      spatial scales.

Run:
    python scripts/stokes_validation/stokes_correlation_analysis.py

Outputs `outputs/patch_scale_sweep.png` next to this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# scripts/stokes_validation/<this file>  ->  parents[2] = REFACTORED_CODEBASE
REFACTORED_ROOT = Path(__file__).resolve().parents[2]
# parents[3] = the project root (one level above REFACTORED_CODEBASE) where data/ lives
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REFACTORED_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.well_dataset import WellDatasetForJEPA
from src.data.derived_fields import divergence_D, laplacian_u, extract_D, extract_u
from src.data.channel_map import ALPHA_IDX


DATA_DIR = PROJECT_ROOT / "data" / "active_matter"
OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

L_PHYS = 10.0
N = 256
H_PHYS = L_PHYS / N


# ------------------------------------------------------------------
# Metric primitives
# ------------------------------------------------------------------

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    af = a.reshape(-1) - a.mean()
    bf = b.reshape(-1) - b.mean()
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-20
    return float(np.dot(af, bf) / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-based Pearson — i.e., Spearman's rho."""
    af = a.reshape(-1)
    bf = b.reshape(-1)
    ra = af.argsort().argsort().astype(np.float64)
    rb = bf.argsort().argsort().astype(np.float64)
    return pearson(ra, rb)


def ols_slope_r2(y: np.ndarray, x: np.ndarray):
    """Fit y = s * x (no intercept). Returns (slope, R^2)."""
    xf = x.reshape(-1).astype(np.float64)
    yf = y.reshape(-1).astype(np.float64)
    s = float(np.dot(xf, yf) / (np.dot(xf, xf) + 1e-20))
    ss_res = float(np.sum((yf - s * xf) ** 2))
    ss_tot = float(np.sum((yf - yf.mean()) ** 2))
    return s, 1.0 - ss_res / (ss_tot + 1e-20)


def patch_average(x: torch.Tensor, k: int) -> torch.Tensor:
    """Block-reduce the last two (H, W) axes by mean over k x k tiles.

    Input: (..., H, W) with H,W divisible by k.
    Output: (..., H//k, W//k).
    """
    if k == 1:
        return x
    H, W = x.shape[-2:]
    assert H % k == 0 and W % k == 0, f"H={H},W={W} not divisible by k={k}"
    leading = x.shape[:-2]
    x2 = x.reshape(*leading, H // k, k, W // k, k)
    return x2.mean(dim=(-3, -1))


# ------------------------------------------------------------------
# Picking samples
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------

ALPHAS = [-1.0, -2.0, -3.0, -4.0, -5.0]
PATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]


def per_component_pearson(lap_u, neg_div_D):
    # lap_u, neg_div_D: (2, T, H, W)
    r1 = pearson(lap_u[0].numpy(), neg_div_D[0].numpy())
    r2 = pearson(lap_u[1].numpy(), neg_div_D[1].numpy())
    return r1, r2


def per_timestep_pearson(lap_u, neg_div_D):
    T = lap_u.shape[1]
    rs = [pearson(lap_u[:, t].numpy(), neg_div_D[:, t].numpy()) for t in range(T)]
    arr = np.array(rs)
    return float(arr.mean()), float(arr.std())


def spearman_pooled(lap_u, neg_div_D):
    return spearman(lap_u.numpy(), neg_div_D.numpy())


def ols_per_component(lap_u, neg_div_D):
    s1, r2_1 = ols_slope_r2(lap_u[0].numpy(), neg_div_D[0].numpy())
    s2, r2_2 = ols_slope_r2(lap_u[1].numpy(), neg_div_D[1].numpy())
    return s1, r2_1, s2, r2_2


def patch_pearson_sweep(lap_u, neg_div_D, sizes):
    out = {}
    for k in sizes:
        a = patch_average(lap_u, k)
        b = patch_average(neg_div_D, k)
        out[k] = pearson(a.numpy(), b.numpy())
    return out


def analyze_one_sample(sample, alpha):
    ctx = sample["context"]
    div_D = divergence_D(extract_D(ctx), spacing=(H_PHYS, H_PHYS))   # (2, T, H, W)
    lap_u = laplacian_u(extract_u(ctx), spacing=(H_PHYS, H_PHYS))
    neg_div_D = -div_D

    r_c1, r_c2 = per_component_pearson(lap_u, neg_div_D)
    r_t_mean, r_t_std = per_timestep_pearson(lap_u, neg_div_D)
    r_sp = spearman_pooled(lap_u, neg_div_D)
    s1, r2_1, s2, r2_2 = ols_per_component(lap_u, neg_div_D)
    patch = patch_pearson_sweep(lap_u, neg_div_D, PATCH_SIZES)

    return {
        "alpha": alpha,
        "pearson_c1": r_c1,
        "pearson_c2": r_c2,
        "pearson_per_t_mean": r_t_mean,
        "pearson_per_t_std": r_t_std,
        "spearman": r_sp,
        "ols_slope_c1": s1,
        "ols_r2_c1": r2_1,
        "ols_slope_c2": s2,
        "ols_r2_c2": r2_2,
        "patch": patch,
    }


def print_table(rows):
    print("\n" + "=" * 96)
    print("Stokes-relation correlation metrics   (pearson r between Delta u and -div D)")
    print("=" * 96)

    # 1. Per-component Pearson
    print("\n[1] Per-component Pearson r (pooled across t, H, W)")
    print(f"  {'alpha':>6}{'r (i=1)':>12}{'r (i=2)':>12}")
    for r in rows:
        print(f"  {r['alpha']:>6.1f}{r['pearson_c1']:>12.4f}{r['pearson_c2']:>12.4f}")

    # 2. Per-timestep Pearson averaged
    print("\n[2] Per-timestep Pearson r, averaged across 16 frames")
    print(f"  {'alpha':>6}{'mean':>12}{'std':>12}")
    for r in rows:
        print(f"  {r['alpha']:>6.1f}{r['pearson_per_t_mean']:>12.4f}{r['pearson_per_t_std']:>12.4f}")

    # 3. Spearman
    print("\n[3] Spearman rank correlation (pooled)")
    print(f"  {'alpha':>6}{'rho':>12}")
    for r in rows:
        print(f"  {r['alpha']:>6.1f}{r['spearman']:>12.4f}")

    # 4. OLS
    print("\n[4] OLS slope (no intercept) + R^2, per component")
    print("    Stokes expects slope = alpha (in appropriate units)")
    print(f"  {'alpha':>6}{'slope i=1':>12}{'R^2 i=1':>12}{'slope i=2':>12}{'R^2 i=2':>12}")
    for r in rows:
        print(f"  {r['alpha']:>6.1f}{r['ols_slope_c1']:>12.4f}{r['ols_r2_c1']:>12.4f}"
              f"{r['ols_slope_c2']:>12.4f}{r['ols_r2_c2']:>12.4f}")

    # 5. Patch-scale sweep
    print("\n[5] Patch-averaged Pearson r vs patch size k (pixels)")
    header = f"  {'alpha':>6}"
    for k in PATCH_SIZES:
        header += f"{'k='+str(k):>10}"
    print(header)
    for r in rows:
        row = f"  {r['alpha']:>6.1f}"
        for k in PATCH_SIZES:
            row += f"{r['patch'][k]:>10.4f}"
        print(row)


def plot_patch_sweep(rows, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r in rows:
        ks = np.array(PATCH_SIZES)
        rs = np.array([r["patch"][k] for k in PATCH_SIZES])
        ax.plot(ks, rs, marker="o", label=f"alpha={r['alpha']:.1f}")
    ax.set_xscale("log", base=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("patch size k (pixels, k x k average)")
    ax.set_ylabel("Pearson r (Delta u vs -div D)")
    ax.set_title("Stokes correlation vs spatial averaging scale")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ds = WellDatasetForJEPA(data_dir=str(DATA_DIR), num_frames=16, split="train")
    picks = find_indices_by_alpha(ds, set(ALPHAS), prefer_late=False)
    assert len(picks) == len(ALPHAS), f"missing alphas: {picks}"

    rows = [analyze_one_sample(ds[picks[a]], a) for a in ALPHAS]
    print_table(rows)

    out = OUT_DIR / "patch_scale_sweep.png"
    plot_patch_sweep(rows, out)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
