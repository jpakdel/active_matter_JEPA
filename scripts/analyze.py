"""Representation analysis for §6.3 / §7 step 13.

For each run's cached features (runs/<run>/features/{train,val,test}.pt), compute:

  1. PCA — top-2 PC scatter on test split, colored by α and by ζ. Saved as
     `pca_<run>.png`. Lets us eyeball whether any single representation has a
     direction that varies smoothly with a physical parameter.

  2. PC correlation — for the top-K PCs, correlate each PC coordinate against
     α and against ζ (Pearson |r|). Saved as `pc_corr_<run>.png` (bar chart)
     and printed. This is the quantitative version of the PCA scatter: one PC
     carrying α structure → one big bar. Diffuse correlations → no single
     linear α axis.

  3. Cross-prediction residuals — fit the best-val ridge on α, take test
     residuals, correlate with ζ. Then flip (fit on ζ, correlate residuals
     with α). Tests disentanglement: if α and ζ live in independent subspaces,
     α-probe residuals should carry no ζ information. Saved to summary.json
     and printed.

Writes a summary.json with all numbers + per-run bar chart comparison PNGs
(so the runs can be compared directly).

Run:
    python scripts/active_matter/analyze_representations.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


# -- config: which runs to analyze -----------------------------------------

RUNS = [
    ("baseline",       "baseline_v0_20260421_152635"),
    ("exp_a",          "exp_a_v0_20260421_185035"),
    ("exp_b_lam0.01",  "exp_b_lam001_v0_20260422_113047"),
    ("exp_b_lam0.1",   "exp_b_v0_20260421_200801"),
    ("exp_b_lam1.0",   "exp_b_lam1_v0_20260422_121613"),
]

TOP_K = 10                # bar chart covers top-K PCs by singular value
OUT_DIR = PROJECT_ROOT / "outputs" / "analysis"

# Where to look for cached features. Defaults to the parent repo's runs/
# (so the refactored codebase can analyze the previously-trained checkpoints
# without copying them in). Override via RUNS_ROOT env var or by editing.
import os as _os
RUNS_ROOT = Path(_os.environ.get("DJEPA_RUNS_ROOT",
                                 str(PROJECT_ROOT.parent / "runs"))).resolve()


def _load_split(run_dir: Path, split: str):
    """Return (features (N, D), labels (N, 2) = [alpha, zeta])."""
    d = torch.load(run_dir / "features" / f"{split}.pt", map_location="cpu", weights_only=False)
    return d["features"].numpy(), d["labels"].numpy()


def _zscore_train(X_tr, X_va, X_te):
    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-8
    return (X_tr - mu) / sd, (X_va - mu) / sd, (X_te - mu) / sd


def _ridge_fit(X, y, lam):
    """Closed-form ridge. X has bias column appended. Returns w."""
    XTX = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(XTX, X.T @ y)


def _ridge_eval(X_tr, y_tr, X_va, y_va, X_te, y_te, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Ridge with lam selected on val. Returns (best_lam, test_pred, test_resid)."""
    Xb_tr = np.concatenate([X_tr, np.ones((X_tr.shape[0], 1))], axis=1)
    Xb_va = np.concatenate([X_va, np.ones((X_va.shape[0], 1))], axis=1)
    Xb_te = np.concatenate([X_te, np.ones((X_te.shape[0], 1))], axis=1)
    best = (None, np.inf, None)
    for lam in lams:
        w = _ridge_fit(Xb_tr, y_tr, lam)
        pred_va = Xb_va @ w
        mse_va = float(((pred_va - y_va) ** 2).mean())
        if mse_va < best[1]:
            best = (lam, mse_va, w)
    lam_best, mse_va_best, w = best
    pred_te = Xb_te @ w
    resid_te = y_te - pred_te
    return lam_best, pred_te, resid_te


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float((a * b).sum() / denom)


@dataclass
class RunAnalysis:
    tag: str
    run_id: str
    pc_r_alpha: list   # len TOP_K, Pearson r PC_k vs α (test split)
    pc_r_zeta: list    # len TOP_K, PC_k vs ζ
    singular_frac: list  # explained-variance fraction, top-K
    alpha_resid_r_with_zeta: float
    zeta_resid_r_with_alpha: float
    alpha_lam: float
    zeta_lam: float
    alpha_test_mse: float
    zeta_test_mse: float
    alpha_resid_std: float
    zeta_resid_std: float


def analyze(tag: str, run_id: str) -> RunAnalysis:
    run_dir = RUNS_ROOT / run_id
    X_tr, y_tr = _load_split(run_dir, "train")
    X_va, y_va = _load_split(run_dir, "val")
    X_te, y_te = _load_split(run_dir, "test")

    # z-score on train stats (train-distribution-relative geometry matters here)
    X_tr_z, X_va_z, X_te_z = _zscore_train(X_tr, X_va, X_te)

    # ---- 1/2: PCA on train, project test ----
    # SVD on z-scored train; project test onto PCs.
    U, S, Vt = np.linalg.svd(X_tr_z, full_matrices=False)
    sing_frac = (S ** 2 / (S ** 2).sum())[:TOP_K].tolist()
    PC_te = X_te_z @ Vt[:TOP_K].T   # (N_test, K)

    # z-score labels on train stats so Pearson is scale-invariant anyway
    # (we compute Pearson below, so z-scoring is not strictly needed)
    alpha_te = y_te[:, 0]
    zeta_te  = y_te[:, 1]

    pc_r_alpha = [_pearson(PC_te[:, k], alpha_te) for k in range(TOP_K)]
    pc_r_zeta  = [_pearson(PC_te[:, k], zeta_te)  for k in range(TOP_K)]

    # ---- 3: cross-prediction residuals (α probe residuals vs ζ, and flip) ----
    # Use *raw* (not feature-z-scored) features + z-scored targets to match
    # src/eval/linear_probe.py's convention so MSE numbers are directly
    # comparable to runs/<run>/eval_results.json.
    y_tr_z = (y_tr - y_tr.mean(0)) / (y_tr.std(0) + 1e-8)
    y_va_z = (y_va - y_tr.mean(0)) / (y_tr.std(0) + 1e-8)
    y_te_z = (y_te - y_tr.mean(0)) / (y_tr.std(0) + 1e-8)

    # α probe
    alpha_lam, alpha_pred, alpha_resid = _ridge_eval(
        X_tr, y_tr_z[:, 0], X_va, y_va_z[:, 0], X_te, y_te_z[:, 0]
    )
    alpha_test_mse = float(((alpha_pred - y_te_z[:, 0]) ** 2).mean())
    # ζ probe
    zeta_lam, zeta_pred, zeta_resid = _ridge_eval(
        X_tr, y_tr_z[:, 1], X_va, y_va_z[:, 1], X_te, y_te_z[:, 1]
    )
    zeta_test_mse = float(((zeta_pred - y_te_z[:, 1]) ** 2).mean())

    # disentanglement: |r(α-resid, ζ)| and |r(ζ-resid, α)|
    alpha_resid_r_with_zeta = _pearson(alpha_resid, y_te_z[:, 1])
    zeta_resid_r_with_alpha = _pearson(zeta_resid,  y_te_z[:, 0])

    # ---- 2D PCA scatter plot ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    sc0 = axes[0].scatter(PC_te[:, 0], PC_te[:, 1], c=alpha_te, cmap="viridis", s=22, alpha=0.85)
    axes[0].set_title(f"{tag}  test PC1 vs PC2 (color = α)")
    axes[0].set_xlabel(f"PC1  (r(α)={pc_r_alpha[0]:+.2f})")
    axes[0].set_ylabel(f"PC2  (r(α)={pc_r_alpha[1]:+.2f})")
    plt.colorbar(sc0, ax=axes[0], label="α")
    sc1 = axes[1].scatter(PC_te[:, 0], PC_te[:, 1], c=zeta_te, cmap="plasma", s=22, alpha=0.85)
    axes[1].set_title(f"{tag}  test PC1 vs PC2 (color = ζ)")
    axes[1].set_xlabel(f"PC1  (r(ζ)={pc_r_zeta[0]:+.2f})")
    axes[1].set_ylabel(f"PC2  (r(ζ)={pc_r_zeta[1]:+.2f})")
    plt.colorbar(sc1, ax=axes[1], label="ζ")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"pca_{tag}.png", dpi=110)
    plt.close(fig)

    # ---- PC correlation bar chart ----
    fig, ax = plt.subplots(figsize=(9, 3.4))
    x = np.arange(TOP_K)
    ax.bar(x - 0.2, [abs(r) for r in pc_r_alpha], width=0.4, label="|r(PC, α)|", color="#4c72b0")
    ax.bar(x + 0.2, [abs(r) for r in pc_r_zeta],  width=0.4, label="|r(PC, ζ)|", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{k+1}" for k in range(TOP_K)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("|Pearson r|")
    ax.set_title(f"{tag}  PC↔physical-parameter correlation on test split")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"pc_corr_{tag}.png", dpi=110)
    plt.close(fig)

    return RunAnalysis(
        tag=tag,
        run_id=run_id,
        pc_r_alpha=pc_r_alpha,
        pc_r_zeta=pc_r_zeta,
        singular_frac=sing_frac,
        alpha_resid_r_with_zeta=alpha_resid_r_with_zeta,
        zeta_resid_r_with_alpha=zeta_resid_r_with_alpha,
        alpha_lam=alpha_lam,
        zeta_lam=zeta_lam,
        alpha_test_mse=alpha_test_mse,
        zeta_test_mse=zeta_test_mse,
        alpha_resid_std=float(alpha_resid.std()),
        zeta_resid_std=float(zeta_resid.std()),
    )


def _cross_run_summary_plot(results, out_path: Path):
    """Side-by-side bar chart: for each run, max |r(PC_k, α)| and max |r(PC_k, ζ)|.
    One bar per run per parameter — shows how concentrated the α/ζ signal is in
    the top PCs."""
    tags = [r.tag for r in results]
    max_r_alpha = [max(abs(x) for x in r.pc_r_alpha) for r in results]
    max_r_zeta  = [max(abs(x) for x in r.pc_r_zeta)  for r in results]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(tags))
    ax.bar(x - 0.2, max_r_alpha, width=0.4, label="max |r(PC, α)|", color="#4c72b0")
    ax.bar(x + 0.2, max_r_zeta,  width=0.4, label="max |r(PC, ζ)|", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("max |r| over top-10 PCs")
    ax.set_title("Strength of α vs ζ signal in top-10 PCs by experiment")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def _residual_disentangle_plot(results, out_path: Path):
    """For each run, |r(α-resid, ζ)| and |r(ζ-resid, α)|. Large values = the
    residuals after probing one parameter still carry the other, so α and ζ
    are not cleanly separable by a linear readout."""
    tags = [r.tag for r in results]
    r_ares_zeta = [abs(r.alpha_resid_r_with_zeta) for r in results]
    r_zres_alpha = [abs(r.zeta_resid_r_with_alpha) for r in results]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(tags))
    ax.bar(x - 0.2, r_ares_zeta,  width=0.4, label="|r(α-probe residuals, ζ)|", color="#4c72b0")
    ax.bar(x + 0.2, r_zres_alpha, width=0.4, label="|r(ζ-probe residuals, α)|", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylim(0, 1.0); ax.set_ylabel("|Pearson r|")
    ax.set_title("Residual leakage (lower = α, ζ better disentangled)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for tag, run_id in RUNS:
        print(f"[analyze] {tag:<15}  {run_id}")
        r = analyze(tag, run_id)
        results.append(r)
        print(f"  top-5 |r(PC,alpha)|: {[round(abs(x),2) for x in r.pc_r_alpha[:5]]}")
        print(f"  top-5 |r(PC,zeta)|:  {[round(abs(x),2) for x in r.pc_r_zeta[:5]]}")
        print(f"  alpha test MSE={r.alpha_test_mse:.3f}   zeta test MSE={r.zeta_test_mse:.3f}")
        print(f"  |r(alpha-resid, zeta)|={abs(r.alpha_resid_r_with_zeta):.3f}   "
              f"|r(zeta-resid, alpha)|={abs(r.zeta_resid_r_with_alpha):.3f}")
        print()

    _cross_run_summary_plot(results, OUT_DIR / "cross_run_max_pc_corr.png")
    _residual_disentangle_plot(results, OUT_DIR / "cross_run_residual_leakage.png")

    # JSON summary
    summary = {
        "runs": [
            {
                "tag": r.tag,
                "run_id": r.run_id,
                "alpha_lam": r.alpha_lam,
                "zeta_lam": r.zeta_lam,
                "alpha_test_mse_zscored": r.alpha_test_mse,
                "zeta_test_mse_zscored": r.zeta_test_mse,
                "pc_r_alpha": r.pc_r_alpha,
                "pc_r_zeta": r.pc_r_zeta,
                "singular_frac_top_k": r.singular_frac,
                "max_abs_r_alpha": max(abs(x) for x in r.pc_r_alpha),
                "max_abs_r_zeta":  max(abs(x) for x in r.pc_r_zeta),
                "alpha_resid_r_with_zeta": r.alpha_resid_r_with_zeta,
                "zeta_resid_r_with_alpha": r.zeta_resid_r_with_alpha,
                "alpha_resid_std": r.alpha_resid_std,
                "zeta_resid_std":  r.zeta_resid_std,
            }
            for r in results
        ],
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analyze] wrote {OUT_DIR/'summary.json'}")
    print(f"[analyze] figures under {OUT_DIR}/")


if __name__ == "__main__":
    main()
