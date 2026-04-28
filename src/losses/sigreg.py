"""SIGReg: Sketched Isotropic Gaussian Regularization.

Port of the reference implementation in LeJEPA (Balestriero & LeCun,
arXiv 2511.08544). Canonical source:
    https://github.com/rbalestr-lab/lejepa
    commit c293d291ca87cd4fddee9d3fffe4e914c7272052

Replaces the physics repo's VICReg. Replaces V-JEPA's EMA target encoder as
the collapse-prevention mechanism.

---

What SIGReg does
----------------

Given a batch of embeddings Z of shape (N, D), SIGReg pushes the empirical
distribution of Z toward N(0, I_D) by:

    1. Drawing K random unit-norm projection directions a_k in R^D.
    2. Projecting Z onto each direction: z_k = Z @ a_k in R^N.
    3. For each 1D sample z_k, computing the Epps-Pulley statistic — a
       weighted L^2 distance between the empirical characteristic function
       and that of a standard normal.
    4. Averaging across the K slices.

The Cramér-Wold theorem guarantees that if *all* 1D marginals match N(0, 1),
the joint matches N(0, I). In practice we approximate this by random slicing
and integrate |ECF - phi|^2 with a small trapezoidal quadrature.

Epps-Pulley test statistic for one 1D sample z of size N against N(0, 1):

    T(z) = N * integral_{-inf}^{+inf} |phi_emp(t) - exp(-t^2/2)|^2 w(t) dt

where phi_emp(t) = (1/N) sum_j exp(i t z_j) and w(t) = exp(-t^2/2).
Using |phi_emp(t) - exp(-t^2/2)|^2 = (cos_mean - phi)^2 + sin_mean^2 and
exploiting even-symmetry, we integrate on [0, t_max] with doubled interior
trapezoidal weights; the weight array is pre-multiplied by phi(t).
"""

from __future__ import annotations

import torch
from torch import nn

from src.losses._ddp import (
    is_ddp_active as _is_ddp_active,
    all_reduce_avg as _all_reduce_avg,
    world_size as _world_size,
)


class EppsPulley(nn.Module):
    """1D Epps-Pulley normality test statistic vs N(0, 1).

    Accepts shape (..., N, K) — N samples along axis -2, K slices along -1.
    Returns shape (..., K) — one test statistic per slice.
    """

    def __init__(self, t_max: float = 3.0, n_points: int = 17):
        super().__init__()
        assert n_points % 2 == 1, "n_points must be odd for trapezoidal rule"
        self.n_points = n_points
        self.t_max = t_max

        t = torch.linspace(0.0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)

        # Doubled interior weights (symmetry fold); half-weight at endpoints.
        weights = torch.full((n_points,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        phi = torch.exp(-0.5 * t * t)
        # Pre-multiply weights by phi(t) so the forward reduces to a matmul.
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(-2)
        # (*, N, K, n_points)
        xt = x.unsqueeze(-1) * self.t
        cos_mean = torch.cos(xt).mean(dim=-3)    # (*, K, n_points)
        sin_mean = torch.sin(xt).mean(dim=-3)
        cos_mean = _all_reduce_avg(cos_mean)
        sin_mean = _all_reduce_avg(sin_mean)
        err = (cos_mean - self.phi).square() + sin_mean.square()
        # Scale by total (global) sample count; matches classical T = N * integral
        return (err @ self.weights) * N * _world_size()


class SIGReg(nn.Module):
    """SIGReg = mean over random 1D slices of the Epps-Pulley statistic.

    Args:
        embed_dim: D. Used to build random projection shapes.
        num_slices: K. Defaults to D (paper sweeps; D..4D works in practice).
        t_max, n_points: Epps-Pulley quadrature settings.
        reduction: 'mean' (default) or 'sum' over the K slices.

    Projection directions are re-drawn every forward. We keep a step counter
    (a registered buffer) and use it as the torch.Generator seed, so under
    DDP every rank draws the same directions. The step increments each call.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_slices: int | None = None,
        t_max: float = 3.0,
        n_points: int = 17,
        reduction: str = "mean",
    ):
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.embed_dim = embed_dim
        self.num_slices = num_slices if num_slices is not None else embed_dim
        self.reduction = reduction
        self.epps_pulley = EppsPulley(t_max=t_max, n_points=n_points)
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))

        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def _sample_directions(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        with torch.no_grad():
            # Sync step across ranks so every rank draws identical directions.
            step = self.global_step.clone()
            if _is_ddp_active():
                dist.all_reduce(step, op=dist.ReduceOp.MAX)
            g = self._get_generator(device, int(step.item()))
            A = torch.randn(
                (self.embed_dim, self.num_slices),
                device=device,
                dtype=dtype,
                generator=g,
            )
            A = A / A.norm(p=2, dim=0, keepdim=True)
            self.global_step.add_(1)
        return A

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: shape (N, D) — batch of N embeddings of dimension D.
        Returns:
            scalar loss.
        """
        if z.ndim != 2:
            raise ValueError(f"SIGReg expects (N, D), got shape {tuple(z.shape)}")
        if z.size(-1) != self.embed_dim:
            raise ValueError(
                f"SIGReg built for D={self.embed_dim}, got D={z.size(-1)}"
            )
        A = self._sample_directions(z.device, z.dtype)
        sliced = z @ A                               # (N, K)
        stats = self.epps_pulley(sliced)             # (K,)
        if self.reduction == "mean":
            return stats.mean()
        return stats.sum()
