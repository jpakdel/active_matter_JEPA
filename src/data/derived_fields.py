"""Derived spatial-differential fields for active_matter Experiment B.

We compute two quantities per sample, using periodic-boundary central-difference
stencils on the last two (spatial) axes:

    div_D[i] = sum_j d(D_ij)/dx_j              shape: (B, 2, T, H, W)
    laplacian_u[i] = sum_j d^2(u_i)/dx_j^2     shape: (B, 2, T, H, W)

The Stokes equation for this dataset is, schematically:

    Delta u  ~  -alpha * (div D)  +  grad Pi

so Laplacian(u) and -alpha * div(D) should be approximately linearly related
pointwise, up to a pressure-gradient residual.

Channel ordering follows `src.data.channel_map`:
    D input is (B, 4, T, H, W) with channels [D11, D12, D21, D22]
    u input is (B, 2, T, H, W) with channels [u1, u2]
    Output channels are ordered by vector index i = 1..2.

Implementation notes:
    * We use central differences (2nd-order accurate):
          df/dx  ~  (f(x+h) - f(x-h)) / (2h)
          d2f/dx2 ~ (f(x+h) - 2f(x) + f(x-h)) / h^2
    * Periodic padding via torch.roll (equivalent to circular conv).
    * Default spacing h=1 (lattice units). Pass `spacing=(hx, hy)` to use
      physical units; the linear Stokes relation Delta u = -alpha * div D still
      holds after a constant rescale.
    * Works on any dtype/device torch accepts; keeps input dtype and device.

Convention for the trailing spatial axes: (..., H, W) = (..., x, y).
    axis -2 (H) is x, axis -1 (W) is y.
    This matches The Well's HDF5 layout (spatial_dims=['x','y']) as delivered
    by `WellDatasetForJEPA`. Note this is *opposite* the usual image convention
    (H=rows=y, W=cols=x); the unit tests in scripts/ follow the same (x,y) order.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor

from src.data.channel_map import D, U


def _d_dx(f: Tensor, h: float) -> Tensor:
    """Central difference along x (axis -2) with periodic BC."""
    return (torch.roll(f, shifts=-1, dims=-2) - torch.roll(f, shifts=1, dims=-2)) / (2.0 * h)


def _d_dy(f: Tensor, h: float) -> Tensor:
    """Central difference along y (axis -1) with periodic BC."""
    return (torch.roll(f, shifts=-1, dims=-1) - torch.roll(f, shifts=1, dims=-1)) / (2.0 * h)


def _laplacian_2d(f: Tensor, hx: float, hy: float) -> Tensor:
    """5-point-stencil Laplacian on the last two axes with periodic BC."""
    d2_dx2 = (torch.roll(f, shifts=-1, dims=-2)
              + torch.roll(f, shifts=1, dims=-2)
              - 2.0 * f) / (hx * hx)
    d2_dy2 = (torch.roll(f, shifts=-1, dims=-1)
              + torch.roll(f, shifts=1, dims=-1)
              - 2.0 * f) / (hy * hy)
    return d2_dx2 + d2_dy2


def divergence_D(
    D_channels: Tensor,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Tensor:
    """Compute (div D)_i = d(D_i1)/dx_1 + d(D_i2)/dx_2.

    Args:
        D_channels: tensor of shape (..., 4, H, W) with channel order [D11, D12, D21, D22].
        spacing: (hx, hy). hx is spacing along axis -2 (x), hy along axis -1 (y).

    Returns:
        tensor of shape (..., 2, H, W) with channel order [(div D)_1, (div D)_2].

    Convention:
        x_1 = x (axis -2, height), x_2 = y (axis -1, width).
        (div D)_1 = dD11/dx + dD12/dy
        (div D)_2 = dD21/dx + dD22/dy
    """
    assert D_channels.shape[-4] == 4, f"expected 4 D channels on axis -4, got shape {tuple(D_channels.shape)}"
    hx, hy = spacing
    D11 = D_channels.select(-4, 0)
    D12 = D_channels.select(-4, 1)
    D21 = D_channels.select(-4, 2)
    D22 = D_channels.select(-4, 3)
    div1 = _d_dx(D11, hx) + _d_dy(D12, hy)
    div2 = _d_dx(D21, hx) + _d_dy(D22, hy)
    return torch.stack([div1, div2], dim=-4)


def laplacian_u(
    u_channels: Tensor,
    spacing: Tuple[float, float] = (1.0, 1.0),
) -> Tensor:
    """Compute Delta u_i = d2(u_i)/dx_1^2 + d2(u_i)/dx_2^2.

    Args:
        u_channels: tensor of shape (..., 2, H, W) with channel order [u1, u2].
        spacing: (hx, hy).

    Returns:
        tensor of shape (..., 2, H, W) with channel order [Delta u_1, Delta u_2].
    """
    assert u_channels.shape[-4] == 2, f"expected 2 u channels on axis -4, got shape {tuple(u_channels.shape)}"
    hx, hy = spacing
    u1 = u_channels.select(-4, 0)
    u2 = u_channels.select(-4, 1)
    lap1 = _laplacian_2d(u1, hx, hy)
    lap2 = _laplacian_2d(u2, hx, hy)
    return torch.stack([lap1, lap2], dim=-4)


def extract_D(x: Tensor) -> Tensor:
    """Slice the D channels from an 11-channel active_matter tensor.

    Accepts shape (C, T, H, W) or (B, C, T, H, W). Channel axis is assumed to
    be the 4th-from-last axis in either case.
    """
    if x.ndim == 5:
        return x[:, D]
    if x.ndim == 4:
        return x[D]
    raise ValueError(f"extract_D expects 4D or 5D tensor, got ndim={x.ndim}")


def extract_u(x: Tensor) -> Tensor:
    """Slice the u channels from an 11-channel active_matter tensor."""
    if x.ndim == 5:
        return x[:, U]
    if x.ndim == 4:
        return x[U]
    raise ValueError(f"extract_u expects 4D or 5D tensor, got ndim={x.ndim}")
