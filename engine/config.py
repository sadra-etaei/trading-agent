"""Configuration loading. A config is a nested dict with attribute access."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class Cfg(dict):
    """dict with attribute access for nested config sections."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path = "config.yaml", overrides: dict | None = None) -> Cfg:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if overrides:
        raw = _merge(raw, overrides)
    return _to_cfg(raw)


def _to_cfg(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _to_cfg(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_cfg(v) for v in d]
    return d


def slippage(cfg, symbol: str) -> float:
    """Per-side trading cost beyond the fee: the pair's measured half-spread when known
    (`costs.slippage_by_pair`, from the gap logger), else the flat `costs.slippage_bps`."""
    by = cfg.costs.get("slippage_by_pair") or {}
    return float(by[symbol]) if symbol in by else cfg.costs.slippage_bps / 1e4


def config_hash(cfg: dict) -> str:
    """Stable short hash of a config, stored with every run for reproducibility."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def universe(cfg: Cfg) -> list[str]:
    """Tradeable pairs: configured universe minus the challenge blocklist."""
    from engine.wallex.allowed import is_allowed

    pairs = list(cfg.universe.core)
    if cfg.universe.include_optional:
        pairs += [p for p in cfg.universe.optional if p not in pairs]
    path = cfg.universe.get("blocked_file", "blocked_markets.txt")
    blocked = [p for p in pairs if not is_allowed(p, path)]
    if blocked:
        import logging

        logging.getLogger(__name__).warning("removing blocked markets from universe: %s", blocked)
    return [p for p in pairs if p not in blocked]
