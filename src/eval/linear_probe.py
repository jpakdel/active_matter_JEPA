"""Linear probe on frozen features: single Linear layer, MSE on z-scored targets.

Per project_plan §4: one linear layer. No MLP heads. Trained with MSE. We
report alpha and zeta separately (not a combined scalar).

Why "one linear layer" and not closed-form ridge? Closed form is equivalent
and faster, and we use it: `fit_linear_probe_closed_form` solves the ridge
regression normal equations. We also keep an SGD-based variant for cases
where N exceeds memory and we want mini-batch SGD; not currently used but
kept for future fine-tuning-sweep experiments.

Output convention: `{target_name: val_mse, test_mse}` where `val_mse` is on
z-scored targets (mean=0, std=1 on train), so a constant-mean predictor
gets exactly 1.0. Lower is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from src.eval.normalize_labels import LabelStats, apply_label_stats, fit_label_stats


# ---- closed-form ridge -------------------------------------------------------

def _ridge_closed_form(
    X: torch.Tensor,           # (N, D) train features
    Y: torch.Tensor,           # (N, K) train targets (already z-scored)
    alpha: float,              # L2 penalty on weights
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (W, b) solving  min_W,b ||Y - (X W + b)||^2 + alpha * ||W||^2.

    Closed form: append a column of ones to X and solve the normal equations
    on the augmented design matrix, then strip out the bias. We *don't*
    penalize the bias (standard choice). With D=384 and ridge, this is a
    ~40ms solve even on CPU.
    """
    N, D = X.shape
    K = Y.shape[1]
    # Center so the bias comes out cleanly and we can L2-regularize only W.
    xm = X.mean(dim=0, keepdim=True)
    ym = Y.mean(dim=0, keepdim=True)
    Xc = X - xm
    Yc = Y - ym
    # (D, D) + alpha I
    XtX = Xc.T @ Xc
    XtX.diagonal().add_(alpha)
    XtY = Xc.T @ Yc                           # (D, K)
    # Solve XtX W = XtY
    W = torch.linalg.solve(XtX, XtY)          # (D, K)
    b = ym.squeeze(0) - xm.squeeze(0) @ W     # (K,)
    return W, b


def _predict(X: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return X @ W + b


def _per_target_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """(N, K) -> (K,)"""
    return ((pred - target) ** 2).mean(dim=0)


# ---- public API --------------------------------------------------------------

@dataclass
class LinearProbeResult:
    target_names: tuple[str, ...]
    val_mse: list[float]          # per-target, z-scored
    test_mse: Optional[list[float]]   # per-target, z-scored, None if no test split
    best_alpha: float
    alpha_grid: tuple[float, ...]
    # For reproducibility:
    stats: LabelStats

    def to_dict(self) -> dict:
        return {
            "target_names": list(self.target_names),
            "val_mse": self.val_mse,
            "test_mse": self.test_mse,
            "best_alpha": self.best_alpha,
            "alpha_grid": list(self.alpha_grid),
            "stats": self.stats.to_dict(),
        }


def fit_linear_probe(
    train_features: torch.Tensor,       # (N_tr, D)
    train_labels: torch.Tensor,         # (N_tr, K)
    val_features: torch.Tensor,         # (N_va, D)
    val_labels: torch.Tensor,           # (N_va, K)
    test_features: Optional[torch.Tensor] = None,    # (N_te, D) or None
    test_labels: Optional[torch.Tensor] = None,
    *,
    target_names: Sequence[str] = ("alpha", "zeta"),
    alpha_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    stats: Optional[LabelStats] = None,
    verbose: bool = False,
) -> LinearProbeResult:
    """Ridge-regression probe with L2 sweep tuned on val.

    Steps:
      1. Z-score targets using train-split stats (reuse `stats` if given so the
         same normalization applies to kNN too).
      2. For each alpha in `alpha_grid`, fit on train, score on val (per-target
         MSE, averaged across targets for model selection).
      3. Pick best alpha by mean val MSE; refit (trivially the same W), report
         per-target val/test MSE.
    """
    if stats is None:
        stats = fit_label_stats(train_labels)
    y_tr = apply_label_stats(train_labels, stats)
    y_va = apply_label_stats(val_labels, stats)
    y_te = apply_label_stats(test_labels, stats) if test_labels is not None else None

    X_tr = train_features.float()
    X_va = val_features.float()
    X_te = test_features.float() if test_features is not None else None

    best = None
    for a in alpha_grid:
        W, b = _ridge_closed_form(X_tr, y_tr, alpha=a)
        val_mse = _per_target_mse(_predict(X_va, W, b), y_va)    # (K,)
        mean_val = float(val_mse.mean().item())
        if verbose:
            tag = " ".join(f"{n}={v:.4f}" for n, v in zip(target_names, val_mse.tolist()))
            print(f"  [probe] alpha={a:<7g} val: {tag}  mean={mean_val:.4f}", flush=True)
        if best is None or mean_val < best["mean_val"]:
            best = {"alpha": a, "mean_val": mean_val, "W": W, "b": b, "val_mse": val_mse}

    assert best is not None
    test_mse = None
    if X_te is not None and y_te is not None:
        test_mse = _per_target_mse(_predict(X_te, best["W"], best["b"]), y_te)

    return LinearProbeResult(
        target_names=tuple(target_names),
        val_mse=[float(v) for v in best["val_mse"].tolist()],
        test_mse=[float(v) for v in test_mse.tolist()] if test_mse is not None else None,
        best_alpha=float(best["alpha"]),
        alpha_grid=tuple(alpha_grid),
        stats=stats,
    )
