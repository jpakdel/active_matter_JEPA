"""VICReg: Variance-Invariance-Covariance Regularization.

Reference:
    Bardes, Ponce, LeCun. "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning." ICLR 2022. arXiv:2105.04906.

Role in this project
--------------------

SIGReg is our default collapse-prevention term (see src/losses/sigreg.py).
VICReg is an alternative with a *weaker* distributional prior — it only
constrains first and second moments of the batch embedding distribution, not
the full shape. This matters because the physical scalar alpha we want to
linearly probe most naturally lives on a magnitude/scale axis, which SIGReg's
isotropic-Gaussian matching aggressively flattens. VICReg is the natural
counterfactual: same collapse-prevention role, softer prior.

Formulation
-----------

Given Z of shape (N, D) — N embeddings of dimension D — VICReg's regularizer
part is:

    V(Z) = (1/D) * sum_d  max(0, gamma - sqrt(Var(z_d) + eps))     (variance)
    C(Z) = (1/D) * sum_{i != j}  Cov(Z)_{i,j}^2                    (covariance)

    reg(Z) = var_weight * V(Z) + cov_weight * C(Z)

V is a hinge penalty that pushes each feature dim's std toward gamma=1; it
does nothing once std >= gamma, so it's a collapse floor, not a whitening
constraint. C pushes the off-diagonal cross-feature covariances toward zero
(decorrelation). Neither term touches the shape of the distribution beyond
second moments — a magnitude axis with std >> 1 is left alone.

The standard VICReg loss also has an invariance term (MSE between two views);
in D-JEPA that role is already played by the predictor MSE, so we return only
V + C here. Outer DJepaLoss.lam scales the full regularizer, analogous to
lambda_sigreg.

Defaults (var_weight=25, cov_weight=1, gamma=1.0, eps=1e-4) match the
original paper. The paper's invariance weight of 25 is not our concern here
since pred_mse plays that role at its own natural scale.
"""

from __future__ import annotations

import torch
from torch import nn

from src.losses._ddp import (
    is_ddp_active as _is_ddp_active,
    all_reduce_avg as _all_reduce_avg,
)


class VICReg(nn.Module):
    """VICReg regularizer (variance + covariance terms only).

    Args:
        embed_dim: D. Kept for symmetry with SIGReg's interface; used only
            for a shape check at forward time.
        var_weight: multiplier on the variance hinge term (paper: mu = 25).
        cov_weight: multiplier on the covariance term (paper: nu = 1).
        gamma: variance hinge target (paper: 1.0). Per-dim std is pushed up
            to at least this value.
        eps: numerical floor inside the sqrt when computing per-dim std.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        var_weight: float = 25.0,
        cov_weight: float = 1.0,
        gamma: float = 1.0,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.var_weight = float(var_weight)
        self.cov_weight = float(cov_weight)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        # Center per-dim, compute std with eps floor, hinge at gamma.
        # Unbiased var (N-1) so N=1 would error; we guard against that.
        if z.size(0) < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        std = torch.sqrt(z.var(dim=0, unbiased=True) + self.eps)   # (D,)
        hinge = torch.relu(self.gamma - std)                       # (D,)
        return hinge.mean()

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        # Empirical covariance with N-1 denominator, then sum squared off-diag
        # entries, normalized by D (not D*(D-1)) per the VICReg paper.
        N = z.size(0)
        if N < 2:
            return torch.zeros((), device=z.device, dtype=z.dtype)
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (N - 1)                                # (D, D)
        D = z.size(1)
        off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
        return off_diag_sq / D

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: shape (N, D) — batch of N embeddings of dimension D.
        Returns:
            scalar loss = var_weight * V(Z) + cov_weight * C(Z).
        """
        if z.ndim != 2:
            raise ValueError(f"VICReg expects (N, D), got shape {tuple(z.shape)}")
        if z.size(-1) != self.embed_dim:
            raise ValueError(
                f"VICReg built for D={self.embed_dim}, got D={z.size(-1)}"
            )
        # Under DDP we average variance and covariance estimates across ranks
        # so each rank sees the same effective batch statistics. This keeps
        # gradients consistent across ranks without any all_gather of z itself.
        v = _all_reduce_avg(self.variance_loss(z))
        c = _all_reduce_avg(self.covariance_loss(z))
        return self.var_weight * v + self.cov_weight * c
