"""D-JEPA total loss = predictor MSE + lambda * regularizer.

Per the project plan (§1):

    Loss = ||z_hat - sg(z_tgt)||^2  +  lambda * Reg(z)

where:
  - z_hat  is the predictor output given the context encoding.
  - z_tgt  is the target-branch encoding (treated as a stop-gradient constant).
  - z      is the embedding passed to the regularizer. By default we regularize
           only the context-branch embeddings; caller can pass a larger pool
           (e.g. concat of context + target) if they want.
  - Reg    is either SIGReg (default; isotropic-Gaussian sliced test) or
           VICReg (variance hinge + off-diagonal covariance decorrelation).
           Selected by `reg_type`.

The `DJepaLossOutput.sigreg` field is named historically but holds whichever
regularizer value was computed — `reg_type` on the dataclass disambiguates.
Keeping the field name means metrics.jsonl / manifest.tsv schemas don't move.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import nn

from src.losses.sigreg import SIGReg
from src.losses.vicreg import VICReg


@dataclass
class DJepaLossOutput:
    total: torch.Tensor       # scalar, gradient target
    pred_mse: torch.Tensor    # scalar, predictor term (per-element MSE)
    sigreg: torch.Tensor      # scalar, regularizer value (pre-weighting). Name
                              #   kept for backward compat; see `reg_type`.
    lam: float                # the lambda used, for logging
    reg_type: str             # "sigreg" | "vicreg"


class DJepaLoss(nn.Module):
    """Predictor MSE + lambda * regularizer, with pluggable regularizer.

    Args:
        embed_dim: D, the encoder embedding dim the regularizer acts on.
        lam: outer weight on the regularizer. Plays the same role it did for
            SIGReg-only loss (`lambda_sigreg` in config).
        reg_type: "sigreg" (default) or "vicreg".
        reg_kwargs: optional dict of extra kwargs forwarded to the chosen
            regularizer's constructor. For SIGReg this can include
            `num_slices`, `t_max`, `n_points`. For VICReg: `var_weight`,
            `cov_weight`, `gamma`, `eps`.

    The legacy `num_slices` kwarg is still accepted and forwarded to SIGReg
    for backward compatibility with existing call sites.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        lam: float = 1.0,
        reg_type: str = "sigreg",
        reg_kwargs: Optional[dict[str, Any]] = None,
        num_slices: Optional[int] = None,   # legacy: SIGReg only
    ):
        super().__init__()
        self.lam = lam
        self.reg_type = reg_type
        reg_kwargs = dict(reg_kwargs or {})

        if reg_type == "sigreg":
            if num_slices is not None and "num_slices" not in reg_kwargs:
                reg_kwargs["num_slices"] = num_slices
            self.regularizer: nn.Module = SIGReg(embed_dim=embed_dim, **reg_kwargs)
        elif reg_type == "vicreg":
            if num_slices is not None:
                raise ValueError(
                    "num_slices is a SIGReg-only kwarg; do not pass it with reg_type='vicreg'"
                )
            self.regularizer = VICReg(embed_dim=embed_dim, **reg_kwargs)
        else:
            raise ValueError(
                f"reg_type must be 'sigreg' or 'vicreg', got {reg_type!r}"
            )

        # Alias `self.sigreg` kept so older code / checkpoints that refer to
        # the submodule by name still resolve when reg_type=='sigreg'. Under
        # reg_type=='vicreg' this alias does not exist; callers that poke at
        # internals should use `self.regularizer`.
        if reg_type == "sigreg":
            self.sigreg = self.regularizer

    def forward(
        self,
        z_hat: torch.Tensor,
        z_tgt: torch.Tensor,
        z_for_reg: Optional[torch.Tensor] = None,
    ) -> DJepaLossOutput:
        """
        Args:
            z_hat: predictor output, shape (B, N, D).
            z_tgt: target encoding, shape (B, N, D). Must already be detached
                by the caller (we do not detach here so the caller stays in
                control of what gets stop-gradient'd).
            z_for_reg: tokens to regularize, shape (B', N', D) or (M, D). If
                None, we regularize `z_hat` flattened to (B*N, D).

        Returns:
            DJepaLossOutput with scalar `.total` to call .backward() on.
        """
        if z_hat.shape != z_tgt.shape:
            raise ValueError(
                f"z_hat {tuple(z_hat.shape)} and z_tgt {tuple(z_tgt.shape)} must match"
            )

        pred_mse = torch.mean((z_hat - z_tgt) ** 2)

        reg_input = z_for_reg if z_for_reg is not None else z_hat
        if reg_input.ndim == 3:
            reg_input = reg_input.reshape(-1, reg_input.size(-1))
        reg_val = self.regularizer(reg_input)

        total = pred_mse + self.lam * reg_val
        return DJepaLossOutput(
            total=total,
            pred_mse=pred_mse.detach(),
            sigreg=reg_val.detach(),
            lam=self.lam,
            reg_type=self.reg_type,
        )
