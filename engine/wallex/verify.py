"""Read-only checks of the private Wallex API once a key is available. Places NO orders.
Prints the raw shapes so the field names used by WallexBroker can be confirmed."""
from __future__ import annotations

import json

from engine.config import universe
from engine.wallex.client import WallexClient, WallexError


def _show(title: str, fn) -> None:
    print(f"\n=== {title}")
    try:
        res = fn()
        txt = json.dumps(res, ensure_ascii=False, indent=1, default=str)
        print(txt[:1500] + (" ..." if len(txt) > 1500 else ""))
    except WallexError as e:
        print(f"FAILED: {e} (status={e.status})")


def verify(cfg) -> None:
    c = WallexClient.from_config(cfg)
    sym = universe(cfg)[0]
    print("API key present:", bool(c.api_key))
    _show("markets (public)", lambda: {k: c.markets()[k] for k in universe(cfg)[:1]})
    _show(f"depth {sym} (public)", lambda: {k: v[:2] for k, v in c.depth(sym).items()})
    _show("balances (private)", lambda: {k: v for k, v in c.balances().items() if float(v.get("value", 0) or 0) > 0})
    _show("open orders (private)", c.open_orders)
    _show(f"account trades {sym} (private)", lambda: c.account_trades(sym)[:3])
