"""Market rules (precision, minimums) from /v1/markets, cached in data/markets.json.

Verified field meaning: `stepSize` = quantity decimals, `tickSize` = price decimals
(e.g. BTCTMN stepSize 8 / minQty 1e-8; XRPTMN stepSize 1 / minQty 0.1; TMN prices are integers).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT = {"min_notional": 50_000.0, "qty_dec": 8, "price_dec": 0, "min_qty": 0.0, "market_orders": False}


def parse(symbols: dict) -> dict[str, dict]:
    out = {}
    for s, v in symbols.items():
        out[s] = {
            "min_notional": float(v.get("minNotional") or DEFAULT["min_notional"]),
            "qty_dec": int(v.get("stepSize", 8)),
            "price_dec": int(v.get("tickSize", 0)),
            "min_qty": float(v.get("minQty") or 0.0),
            "market_orders": bool(v.get("isMarketTypeEnable", False)),
        }
    return out


def load(cfg, client=None, refresh: bool = False) -> dict[str, dict]:
    path = Path(cfg.data.dir) / "markets.json"
    info = {}
    if client is not None and (refresh or not path.exists()):
        try:
            info = parse(client.markets())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(info, indent=1))
        except Exception as e:
            log.warning("could not fetch markets: %s", e)
    if not info and path.exists():
        info = json.loads(path.read_text())
    return apply_account_minimum(cfg, info)


def apply_account_minimum(cfg, info: dict) -> dict:
    """The challenge account's minimum order (execution.min_order_tmn, 100,000 TMN) is higher than
    the market minimum Wallex publishes (50,000 TMN): use the larger one for every TMN market."""
    m = float((cfg.get("execution") or {}).get("min_order_tmn") or 0)
    if m:
        for s, r in info.items():
            if s.endswith("TMN"):
                r["min_notional"] = max(float(r.get("min_notional", 0)), m)
    return info


def rules(markets: dict, symbol: str) -> dict:
    return {**DEFAULT, **markets.get(symbol, {})}
