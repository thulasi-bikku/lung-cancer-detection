"""Tiny YAML config loader with a one-level ``extends`` mechanism."""
from __future__ import annotations

from pathlib import Path

import yaml


def _merge(base: dict, new: dict) -> dict:
    out = dict(base)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.pop("extends", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    return cfg
