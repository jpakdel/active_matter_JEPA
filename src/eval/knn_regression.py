"""kNN regression on frozen features, tuning k on validation only.

Per project_plan §4: kNN regression with k selected on val, MSE reported per
target on z-scored labels so val/test numbers are comparable to the linear
probe (a constant-mean predictor scores 1.0).

Distance metric: L2 on raw features. For higher-dimensional ViT embeddings
cosine can outperform L2, so we try both and pick per-metric+k via val.

Implementation note: for our dataset sizes (train ≤ ~10k, D=384) the exact
pairwise distance matrix fits comfortably in GPU memory. We use chunked torch
cdist so this also works if N_val grows, without bringing in sklearn.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch

from src.eval.normalize_labels import LabelStats, apply_label_stats, fit_label_stats


# ---- distances ---------------------------------------------------------------

def _pairwise_dist(
    queries: torch.Tensor,    # (M, D)
    bank: torch.Tensor,       # (N, D)
    metric: str = "l2",
    chunk: int = 1024,
) -> torch.Tensor:
    """Returns (M, N). Chunked over M so memory stays bounded."""
    if metric == "l2":
        out = torch.empty((queries.shape[0], bank.shape[0]), dtype=torch.float32,
                          device=queries.device)
        for i in range(0, queries.shape[0], chunk):
            q = queries[i:i+chunk]
            out[i:i+chunk] = torch.cdist(q, bank, p=2)
        return out
    if metric == "cosine":
        qn = torch.nn.functional.normalize(queries, dim=1)
        bn = torch.nn.functional.normalize(bank, dim=1)
        # 1 - cos_sim, so smaller = more similar (same convention as l2).
        return 1.0 - qn @ bn.T
    raise ValueError(f"unknown metric {metric!r}")


# ---- kNN predict -------------------------------------------------------------

def _knn_predict(
    dists: torch.Tensor,      # (M, N) query-bank distances
    bank_labels: torch.Tensor,  # (N, K)
    k: int,
) -> torch.Tensor:
    """(M, K) uniform-weighted mean of the k nearest train labels per query."""
    _, idx = torch.topk(dists, k=k, largest=False, dim=1)    # (M, k)
    gathered = bank_labels[idx]                              # (M, k, K)
    return gathered.mean(dim=1)


def _per_target_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=0)


# ---- public API --------------------------------------------------------------

@dataclass
class KNNResult:
    target_names: tuple[str, ...]
    val_mse: list[float]               # per-target, z-scored (at best (k, metric))
    test_mse: Optional[list[float]]    # per-target, z-scored; None if no test
    best_k: int
    best_metric: str
    k_grid: tuple[int, ...]
    metric_grid: tuple[str, ...]
    stats: LabelStats

    def to_dict(self) -> dict:
        return {
            "target_names": list(self.target_names),
            "val_mse": self.val_mse,
            "test_mse": self.test_mse,
            "best_k": int(self.best_k),
            "best_metric": self.best_metric,
            "k_grid": list(self.k_grid),
            "metric_grid": list(self.metric_grid),
            "stats": self.stats.to_dict(),
        }


def fit_knn(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    test_features: Optional[torch.Tensor] = None,
    test_labels: Optional[torch.Tensor] = None,
    *,
    target_names: Sequence[str] = ("alpha", "zeta"),
    k_grid: Sequence[int] = (1, 3, 5, 10, 20, 50, 100),
    metric_grid: Sequence[str] = ("l2", "cosine"),
    stats: Optional[LabelStats] = None,
    device: Optional[str] = None,
    verbose: bool = False,
) -> KNNResult:
    """kNN regression. Returns per-target val/test MSE on z-scored targets.

    We pick (k, metric) by lowest mean-over-targets val MSE, then evaluate on
    test with the same (k, metric).
    """
    if stats is None:
        stats = fit_label_stats(train_labels)
    y_tr = apply_label_stats(train_labels, stats)
    y_va = apply_label_stats(val_labels, stats)
    y_te = apply_label_stats(test_labels, stats) if test_labels is not None else None

    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    X_tr = train_features.float().to(dev)
    X_va = val_features.float().to(dev)
    y_tr_d = y_tr.float().to(dev)
    y_va_d = y_va.float().to(dev)

    # Filter k to fit: can't take more neighbors than we have train points.
    k_grid = tuple(k for k in k_grid if k <= X_tr.shape[0])

    best = None
    for metric in metric_grid:
        dists_va = _pairwise_dist(X_va, X_tr, metric=metric)
        for k in k_grid:
            pred = _knn_predict(dists_va, y_tr_d, k)
            val_mse = _per_target_mse(pred, y_va_d)            # (K,)
            mean_val = float(val_mse.mean().item())
            if verbose:
                tag = " ".join(f"{n}={v:.4f}" for n, v in zip(target_names, val_mse.tolist()))
                print(f"  [knn] metric={metric:<6} k={k:>3d}  val: {tag}  mean={mean_val:.4f}",
                      flush=True)
            if best is None or mean_val < best["mean_val"]:
                best = {
                    "metric": metric, "k": k, "mean_val": mean_val,
                    "val_mse": val_mse.detach().cpu(),
                }

    assert best is not None
    test_mse = None
    if test_features is not None and y_te is not None:
        X_te = test_features.float().to(dev)
        y_te_d = y_te.float().to(dev)
        dists_te = _pairwise_dist(X_te, X_tr, metric=best["metric"])
        pred = _knn_predict(dists_te, y_tr_d, best["k"])
        test_mse = _per_target_mse(pred, y_te_d).detach().cpu()

    return KNNResult(
        target_names=tuple(target_names),
        val_mse=[float(v) for v in best["val_mse"].tolist()],
        test_mse=[float(v) for v in test_mse.tolist()] if test_mse is not None else None,
        best_k=int(best["k"]),
        best_metric=best["metric"],
        k_grid=tuple(k_grid),
        metric_grid=tuple(metric_grid),
        stats=stats,
    )
