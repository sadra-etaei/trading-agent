"""Smallest possible real round trip to verify order placement, status fields and fills before
going live. Buys ~`tmn` worth of `symbol` with a marketable limit order, prints the raw order
status, then sells the same quantity back. Run only with --yes; costs about 2 x taker fee.
"""
from __future__ import annotations

import json
import time
import uuid

from engine.broker.brokers import _floor, _fmt
from engine.wallex import markets as mk
from engine.wallex.allowed import assert_allowed
from engine.wallex.client import WallexClient


def _show(title: str, obj) -> None:
    print(f"\n=== {title}\n" + json.dumps(obj, ensure_ascii=False, indent=1, default=str)[:2000])


def smoke(cfg, symbol: str = "USDTTMN", tmn: float = 110_000) -> None:
    assert_allowed(symbol)
    c = WallexClient.from_config(cfg)
    r = mk.rules(mk.load(cfg, c, refresh=True), symbol)
    base = symbol.removesuffix("TMN")
    _show("balances before", {k: v for k, v in c.balances().items() if float(v.get("value", 0) or 0) > 0})

    book = c.depth(symbol)
    ask, bid = float(book["ask"][0]["price"]), float(book["bid"][0]["price"])
    px = round(ask * 1.002, r["price_dec"])
    qty = _floor(tmn / px, r["qty_dec"])
    cid = f"smoke-buy-{uuid.uuid4().hex[:10]}"
    print(f"\nBUY {qty} {base} @ {px} (best ask {ask}) = {qty * px:,.0f} TMN, client_id {cid}")
    _show("place BUY response", c.place_order(symbol, "BUY", _fmt(px, r["price_dec"]), _fmt(qty, r["qty_dec"]), cid))
    time.sleep(3)
    _show("BUY order status", c.get_order(cid))

    time.sleep(1)
    bal = c.balances().get(base, {})
    free = float(bal.get("value", 0) or 0) - float(bal.get("locked", 0) or 0)
    sell_qty = _floor(min(qty, free), r["qty_dec"])
    book = c.depth(symbol)
    bid = float(book["bid"][0]["price"])
    spx = round(bid * 0.998, r["price_dec"])
    cid2 = f"smoke-sell-{uuid.uuid4().hex[:10]}"
    print(f"\nSELL {sell_qty} {base} @ {spx} (best bid {bid}); free balance after buy {free}")
    _show("place SELL response", c.place_order(symbol, "SELL", _fmt(spx, r["price_dec"]), _fmt(sell_qty, r["qty_dec"]), cid2))
    time.sleep(3)
    _show("SELL order status", c.get_order(cid2))
    _show("open orders after", c.open_orders())
    _show("balances after", {k: v for k, v in c.balances().items() if float(v.get("value", 0) or 0) > 0})
    print("\nSmoke test done. Share this output so the order-status parsing can be confirmed.")
