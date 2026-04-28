"""Label normalization for eval.

Per project_plan §4: z-score alpha and zeta using train-split statistics only,
then apply the same transform to val and test. The frozen-encoder MSE numbers
we report are on z-scored targets, so a model that always predicts the mean
scores exactly 1.0 per-parameter — this gives an intuitive 0→1 scale where
lower-than-1 means "better than the constant-mean baseline".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import torch


@dataclass(frozen=True)
class LabelStats:
    means: tuple[float, ...]   # per-target mean (computed on train only)
    stds: tuple[float, ...]    # per-target std  (computed on train only)

    def as_tensors(self, device=None, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        m = torch.tensor(self.means, device=device, dtype=dtype)
        s = torch.tensor(self.stds, device=device, dtype=dtype)
        return m, s

    def to_dict(self) -> dict:
        return {"means": list(self.means), "stds": list(self.stds)}

    @classmethod
    def from_dict(cls, d: dict) -> "LabelStats":
        return cls(means=tuple(d["means"]), stds=tuple(d["stds"]))


def fit_label_stats(train_labels: torch.Tensor, eps: float = 1e-8) -> LabelStats:
    """Fit per-column mean/std on training labels.

    `train_labels`: (N, K). We z-score each of the K targets independently
    (K=2 for active_matter: alpha, zeta).
    """
    if train_labels.ndim != 2:
        raise ValueError(f"expected (N, K), got {tuple(train_labels.shape)}")
    mu = train_labels.mean(dim=0).tolist()
    sd = train_labels.std(dim=0, unbiased=False).clamp_min(eps).tolist()
    return LabelStats(means=tuple(mu), stds=tuple(sd))


def apply_label_stats(labels: torch.Tensor, stats: LabelStats) -> torch.Tensor:
    """Apply z-score transform using given stats. Out-of-place."""
    m, s = stats.as_tensors(device=labels.device, dtype=labels.dtype)
    return (labels - m) / s
