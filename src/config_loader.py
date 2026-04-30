"""Layered config loader.

Builds a final config dict by deep-merging four axes, in this order:

    final = default.yaml
            <merge> routings/<routing>.yaml          (channel routing: which
                                                      channels feed ctx vs tgt)
            <merge> backbones/<backbone>.yaml        (model architecture: vit | cnn)
            <merge> targets/<target>.yaml            (target encoder type:
                                                      shared_stopgrad | ema)
            <merge> losses/<loss>.yaml               (regularizer family + knobs)
            <merge> --override key=value flags

This replaces the original 30-file ``configs/active_matter/djepa_*_*.yaml``
sweep with 1 default + 3 routings + 2 backbones + 2 targets + N losses.

Usage:
    from src.config_loader import load_layered_config
    cfg = load_layered_config(
        routing="exp_a", backbone="vit", target="ema", loss="vicreg",
        overrides={"optim.batch_size": "4"},
    )
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Mapping, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = REPO_ROOT / "configs" / "active_matter"


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: dict, overlay: Mapping) -> dict:
    """Recursively merge ``overlay`` into a copy of ``base``.

    Lists and scalars from ``overlay`` replace the corresponding entry in
    ``base`` (no list concatenation). Dicts merge recursively.
    """
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, Mapping):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v) if isinstance(v, (dict, list)) else v
    return out


def _set_dotted(d: dict, dotted_key: str, value):
    """Set ``d['a']['b']['c'] = value`` from ``dotted_key='a.b.c'``."""
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def _coerce_scalar(s: str):
    """Coerce CLI override strings into bool/int/float/str."""
    low = s.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("null", "none", "~"):
        return None
    try:
        return int(low)
    except ValueError:
        pass
    try:
        return float(low)
    except ValueError:
        pass
    return s


def load_layered_config(
    *,
    routing: str,
    backbone: str = "vit",
    target: str = "shared",
    loss: str,
    overrides: Optional[Mapping[str, str]] = None,
    config_root: Optional[Path] = None,
) -> dict:
    """Compose the final config dict.

    Args:
        routing: ``"baseline"`` | ``"exp_a"`` | ``"exp_b"``. Sets data routing.
        backbone: file stem under ``configs/active_matter/backbones/``,
            currently ``"vit"`` or ``"cnn"``.
        target: file stem under ``configs/active_matter/targets/``,
            currently ``"shared"`` or ``"ema"``.
        loss: file stem under ``configs/active_matter/losses/``. E.g.
            ``"sigreg"``, ``"vicreg_lam001"``, ``"vicreg_no_cov"``.
        overrides: optional flat dict of dotted-key strings, e.g.
            ``{"optim.batch_size": "4"}``. Values are coerced (bool/int/float).
        config_root: override config dir for testing.
    """
    root = config_root if config_root is not None else CONFIG_ROOT
    cfg = _load_yaml(root / "default.yaml")
    cfg = _deep_merge(cfg, _load_yaml(root / f"{routing}.yaml"))
    cfg = _deep_merge(cfg, _load_yaml(root / "backbones" / f"{backbone}.yaml"))
    cfg = _deep_merge(cfg, _load_yaml(root / "targets" / f"{target}.yaml"))
    cfg = _deep_merge(cfg, _load_yaml(root / "losses" / f"{loss}.yaml"))
    if overrides:
        for k, v in overrides.items():
            _set_dotted(cfg, k, _coerce_scalar(v))
    return cfg
