"""Builders shared by training and eval.

Channel routing, encoder/predictor/loss construction, dataloader construction,
and YAML loading. Both ``scripts/train.py`` and ``src/eval/extract_features.py``
import from here so the two paths stay in sync.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.well_dataset import WellDatasetForJEPA
from src.losses.djepa_loss import DJepaLoss
from src.models.cnn_encoder import ConvEncoder, DualConvEncoder
from src.models.dual_patch_encoder import DualPatchEncoder
from src.models.encoder import build_encoder, embed_dim_for
from src.models.simple_predictor import build_predictor


# ---- channel selection -------------------------------------------------------

def select_channels(x: torch.Tensor, spec: str) -> torch.Tensor:
    """Slice the 11-channel stack down to a named channel group.

    `spec` is one of the strings documented in ``configs/active_matter/default.yaml``.
    Plus-separated specs (e.g. ``"D+phi"``) concatenate groups along the channel axis.
    """
    from src.data.channel_map import PHI, U, D as D_SLICE, E
    if spec == "all":
        return x
    parts = spec.split("+")
    chunks = []
    for p in parts:
        p = p.strip()
        if p == "phi":
            chunks.append(x[:, PHI:PHI+1])
        elif p == "u":
            chunks.append(x[:, U])
        elif p == "D":
            chunks.append(x[:, D_SLICE])
        elif p == "E":
            chunks.append(x[:, E])
        elif p == "divD":
            from src.data.derived_fields import divergence_D
            d = x[:, D_SLICE]
            chunks.append(divergence_D(d))
        elif p == "lapU":
            from src.data.derived_fields import laplacian_u
            u = x[:, U]
            chunks.append(laplacian_u(u))
        else:
            raise ValueError(f"unknown channel spec part: {p!r}")
    return torch.cat(chunks, dim=1)


def encoder_forward(
    encoder: torch.nn.Module, x: torch.Tensor, branch: str
) -> torch.Tensor:
    """Dispatch to ``Dual{Patch,Conv}Encoder(x, branch=...)`` or a plain encoder.

    Lets callers stay agnostic about whether the config wired up a single
    encoder (VisionTransformer / ConvEncoder) or a dual encoder; only duals
    consume the branch arg. Both ViT-side ``DualPatchEncoder`` and CNN-side
    ``DualConvEncoder`` need the routing — failing to dispatch correctly on
    DualConvEncoder silently routes target inputs through ctx_stem and
    causes (a) a shape-mismatch crash when ctx_chans != tgt_chans, or (b) a
    silent collapse to trivial-prediction when the channel counts happen to
    match (e.g. exp_b with 2-ch divD and 2-ch lapU).
    """
    if isinstance(encoder, (DualPatchEncoder, DualConvEncoder)):
        return encoder(x, branch=branch)
    return encoder(x)


def channels_for(spec: str) -> int:
    """Expected channel count after select_channels(spec)."""
    from src.data.channel_map import NUM_CHANNELS
    if spec == "all":
        return NUM_CHANNELS
    total = 0
    for p in spec.split("+"):
        p = p.strip()
        total += {
            "phi": 1, "u": 2, "D": 4, "E": 4, "divD": 2, "lapU": 2,
        }[p]
    return total


# ---- builders ----------------------------------------------------------------

def build_encoder_from_config(cfg: dict, device: torch.device) -> torch.nn.Module:
    """Build only the encoder (used by eval, which doesn't need predictor/loss).

    Dispatches on ``cfg.model.backbone``:

      - ``"vit"`` (default): ``VisionTransformer`` for matched-channel routings,
        ``DualPatchEncoder`` (two PatchEmbed3D + shared trunk) for mismatched.
      - ``"cnn"``: ``ConvEncoder`` for matched, ``DualConvEncoder`` (two stems
        + shared trunk) for mismatched.

    For the EMA target encoder we always build a *second* instance via this
    same function. Both instances therefore have identical architecture and
    EMA across all parameters works trivially regardless of routing.
    """
    m = cfg["model"]
    d = cfg["data"]
    backbone = m.get("backbone", "vit")

    ctx_ch = channels_for(d["context_channels"])
    tgt_ch = channels_for(d["target_channels"])
    same_branch = (d["context_channels"] == d["target_channels"])

    if backbone == "vit":
        if same_branch:
            return build_encoder(
                in_chans=ctx_ch,
                size=m["encoder_size"],
                img_size=m["encoder"]["img_size"],
                patch_size=m["encoder"]["patch_size"],
                num_frames=m["encoder"]["num_frames"],
                tubelet_size=m["encoder"]["tubelet_size"],
                mlp_ratio=m["encoder"]["mlp_ratio"],
                drop_rate=m["encoder"]["drop_rate"],
                attn_drop_rate=m["encoder"]["attn_drop_rate"],
                uniform_power=m["encoder"]["uniform_power"],
            ).to(device)
        return DualPatchEncoder(
            ctx_in_chans=ctx_ch,
            tgt_in_chans=tgt_ch,
            size=m["encoder_size"],
            img_size=m["encoder"]["img_size"],
            patch_size=m["encoder"]["patch_size"],
            num_frames=m["encoder"]["num_frames"],
            tubelet_size=m["encoder"]["tubelet_size"],
            mlp_ratio=m["encoder"]["mlp_ratio"],
            drop_rate=m["encoder"]["drop_rate"],
            attn_drop_rate=m["encoder"]["attn_drop_rate"],
            uniform_power=m["encoder"]["uniform_power"],
        ).to(device)

    if backbone == "cnn":
        if same_branch:
            return ConvEncoder(
                in_chans=ctx_ch,
                embed_dim=m["encoder"]["embed_dim"],
                base_channels=m["encoder"]["base_channels"],
                num_stages=m["encoder"]["num_stages"],
                res_blocks_per_stage=m["encoder"]["res_blocks_per_stage"],
                tubelet_size=m["encoder"]["tubelet_size"],
                dropout=m["encoder"].get("dropout", 0.0),
            ).to(device)
        return DualConvEncoder(
            ctx_in_chans=ctx_ch,
            tgt_in_chans=tgt_ch,
            embed_dim=m["encoder"]["embed_dim"],
            base_channels=m["encoder"]["base_channels"],
            num_stages=m["encoder"]["num_stages"],
            res_blocks_per_stage=m["encoder"]["res_blocks_per_stage"],
            tubelet_size=m["encoder"]["tubelet_size"],
            dropout=m["encoder"].get("dropout", 0.0),
        ).to(device)

    raise ValueError(f"unknown model.backbone: {backbone!r}")


def _embed_dim_from_config(cfg: dict) -> int:
    """Look up the encoder's output dimension regardless of backbone family."""
    m = cfg["model"]
    backbone = m.get("backbone", "vit")
    if backbone == "vit":
        return embed_dim_for(m["encoder_size"])
    if backbone == "cnn":
        return int(m["encoder"]["embed_dim"])
    raise ValueError(f"unknown model.backbone: {backbone!r}")


def build_from_config(cfg: dict, device: torch.device):
    """Build encoder + predictor + loss from a config dict.

    Branch routing
    --------------
    We compare ``context_channels`` and ``target_channels`` *as strings*, not
    just by channel count. This matters for Experiment B, where both branches
    happen to be 2-channel (∇·D and Δu) but they carry very different
    semantics and must not share the patch-embed Conv3d.

    * Equal strings → single ``VisionTransformer`` (shared encoder, as in the
      baseline).
    * Different strings → ``DualPatchEncoder``: two PatchEmbed3D modules
      feeding one shared transformer trunk. The ``encoder(x, branch=...)``
      call signature takes over from ``encoder(x)``.
    """
    m = cfg["model"]
    enc = build_encoder_from_config(cfg, device)

    D_enc = _embed_dim_from_config(cfg)

    # The predictor's positional embedding grid is determined by an effective
    # ``patch_size`` and ``tubelet_size``: for the ViT these come straight from
    # the config; for the CNN, the effective patch size is the total spatial
    # downsampling (2 ** num_stages), since each stride-2 stage halves H/W.
    backbone = m.get("backbone", "vit")
    if backbone == "vit":
        eff_patch = m["encoder"]["patch_size"]
    elif backbone == "cnn":
        eff_patch = 2 ** int(m["encoder"]["num_stages"])
    else:
        raise ValueError(f"unknown model.backbone: {backbone!r}")

    pred = build_predictor(
        embed_dim=D_enc,
        img_size=m["encoder"]["img_size"],
        patch_size=eff_patch,
        num_frames=m["encoder"]["num_frames"],
        tubelet_size=m["encoder"]["tubelet_size"],
        predictor_embed_dim=m["predictor"]["predictor_embed_dim"],
        depth=m["predictor"]["depth"],
        num_heads=m["predictor"]["num_heads"],
        mlp_ratio=m["predictor"]["mlp_ratio"],
    ).to(device)

    reg_type = cfg["loss"].get("reg_type", "sigreg")
    # Build the reg_kwargs dict by pulling the regularizer-specific keys from
    # the config. Unknown keys are ignored so the same YAML file can keep
    # sigreg_* / vicreg_* blocks side by side.
    if reg_type == "sigreg":
        reg_kwargs = {}
        if cfg["loss"].get("sigreg_num_slices") is not None:
            reg_kwargs["num_slices"] = cfg["loss"]["sigreg_num_slices"]
        if "sigreg_t_max" in cfg["loss"]:
            reg_kwargs["t_max"] = cfg["loss"]["sigreg_t_max"]
        if "sigreg_n_points" in cfg["loss"]:
            reg_kwargs["n_points"] = cfg["loss"]["sigreg_n_points"]
    elif reg_type == "vicreg":
        reg_kwargs = {}
        for k_cfg, k_arg in (
            ("vicreg_var_weight", "var_weight"),
            ("vicreg_cov_weight", "cov_weight"),
            ("vicreg_gamma", "gamma"),
            ("vicreg_eps", "eps"),
        ):
            if k_cfg in cfg["loss"]:
                reg_kwargs[k_arg] = cfg["loss"][k_cfg]
    else:
        raise ValueError(f"unknown loss.reg_type: {reg_type!r}")

    loss_fn = DJepaLoss(
        embed_dim=D_enc,
        lam=cfg["loss"]["lambda_sigreg"],
        reg_type=reg_type,
        reg_kwargs=reg_kwargs,
    ).to(device)

    return enc, pred, loss_fn


def build_loader(cfg: dict, *, split: str, shuffle: bool) -> DataLoader:
    d = cfg["data"]
    ds = WellDatasetForJEPA(
        data_dir=d["data_dir"],
        num_frames=d["num_frames"],
        split=split,
        stride=d.get("stride"),
        noise_std=d.get("noise_std", 0.0),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg["optim"]["batch_size"],
        shuffle=shuffle,
        num_workers=cfg["optim"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,          # drop last in train, keep in eval
        persistent_workers=cfg["optim"]["num_workers"] > 0,
    )
    return loader


# ---- config IO ---------------------------------------------------------------

def load_yaml_config(path: str | Path) -> dict:
    """Tiny YAML loader. Keeps dependencies small — uses PyYAML."""
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)
