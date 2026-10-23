"""Brokers for live data. ShadowBroker simulates fills against the live order book;
WallexBroker places real orders. Both return the same Fill objects, so the live runner does
not know which one it is using.

Wallex has market orders disabled on several pairs (isMarketTypeEnable=false), so all orders are
LIMIT: passive at the best bid (maker) first, then a marketable limit that crosses the spread
with a slippage cap.
"""
from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass

import pandas as pd

from engine.wallex.allowed import assert_allowed
from engine.wallex.client import WallexClient, WallexError
from engine.wallex.markets import rules as market_rules

log = logging.getLogger(__name__)


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float  # average
    fee_tmn: float
    ts: pd.Timestamp
    orders: list[str]
    note: str = ""


def _floor(x: float, dec: int) -> float:
    f = 10 ** dec
    return math.floor(x * f + 1e-9) / f


def _fmt(x: float, dec: int) -> str:
    return f"{x:.{dec}f}"


def walk_book(levels: list[dict], qty: float | None = None, notional: float | None = None) -> tuple[float, float]:
    """Average price and filled quantity when taking liquidity from `levels`."""
    got_q, cost = 0.0, 0.0
    for lv in levels:
        p, q = float(lv["price"]), float(lv["quantity"])
        if notional is not None:
            take = min(q, (notional - cost) / p)
        else:
            take = min(q, qty - got_q)
        if take <= 0:
            break
        got_q += take
        cost += take * p
        if (notional is not None and cost >= notional * 0.9999) or (qty is not None and got_q >= qty * 0.9999):
            break
    return (cost / got_q if got_q else 0.0), got_q


class ShadowBroker:
    """Fills at live order-book prices, crossing the spread (conservative: taker prices + slippage)."""

    def __init__(self, client: WallexClient, cfg, markets: dict):
        self.client = client
        self.fee = cfg.costs.fee_rate
        # fills already walk the real book (the spread is paid there): only a small impact buffer on top
        self.slip = float(cfg.costs.get("shadow_impact_bps", cfg.costs.slippage_bps)) / 1e4
        self.markets = markets

    def buy(self, symbol: str, notional: float) -> Fill | None:
        assert_allowed(symbol)
        book = self.client.depth(symbol)
        avg, q = walk_book(book["ask"], notional=notional)
        if q <= 0:
            return None
        r = market_rules(self.markets, symbol)
        px = avg * (1 + self.slip)
        q = _floor(notional / px, r["qty_dec"])
        return Fill(symbol, "BUY", q, px, q * px * self.fee, pd.Timestamp.now(tz="UTC"), ["shadow"])

    def sell(self, symbol: str, qty: float) -> Fill | None:
        book = self.client.depth(symbol)
        avg, q = walk_book(book["bid"], qty=qty)
        if q <= 0:
            return None
        px = avg * (1 - self.slip)
        return Fill(symbol, "SELL", qty, px, qty * px * self.fee, pd.Timestamp.now(tz="UTC"), ["shadow"])

    def free_balance(self, asset: str) -> float | None:
        return None


class WallexBroker:
    """Real orders. Order-status field names are parsed defensively; verify with `cli.py verify-api`."""

    def __init__(self, client: WallexClient, cfg, markets: dict):
        self.client = client
        self.cfg = cfg
        self.fee = cfg.costs.fee_rate
        self.markets = markets
        self.ex = cfg.execution
        self.poll_s = 3.0  # order status polling interval
        self.taker_wait_s = 30.0

    # ---------------------------------------------------------------- order helpers
    def _place(self, symbol, side, price, qty, r) -> str:
        assert_allowed(symbol)  # last line of defence: never send an order on a blocked market
        cid = f"ai-{symbol[:6]}-{uuid.uuid4().hex[:12]}"
        self.client.place_order(symbol, side, _fmt(price, r["price_dec"]), _fmt(qty, r["qty_dec"]), cid)
        return cid

    @staticmethod
    def _parse(o: dict) -> tuple[str, float, float, float]:
        status = str(o.get("status", "")).upper()
        q = float(o.get("executedQty") or o.get("executed_qty") or 0)
        summ = float(o.get("executedSum") or o.get("executed_sum") or 0)
        # Wallex's executedPrice is the order's LIMIT price (smoke test 2026-10-23: 273,247 vs fills at 272,702);
        # the real average is executedSum / executedQty, then the fills, then executedPrice as a last resort
        fills = o.get("fills") or []
        fq = sum(float(f.get("quantity") or 0) for f in fills)
        price = (summ / q if q and summ else 0) or \
            (sum(float(f.get("price") or 0) * float(f.get("quantity") or 0) for f in fills) / fq if fq else 0) or \
            float(o.get("executedPrice") or o.get("price") or 0)
        fee = float(o.get("fee") or 0)
        return status, q, price, fee

    def _wait(self, cid: str, timeout_s: float) -> tuple[str, float, float, float]:
        t0 = time.time()
        while True:
            try:
                st = self._parse(self.client.get_order(cid))
            except WallexError as e:
                log.warning("order %s status failed: %s", cid, e)
                st = ("UNKNOWN", 0.0, 0.0, 0.0)
            if st[0] in ("FILLED", "DONE", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED"):
                return st
            if time.time() - t0 > timeout_s:
                return st
            time.sleep(self.poll_s)

    def _cancel_and_final(self, cid: str) -> tuple[str, float, float, float]:
        try:
            self.client.cancel_order(cid)
        except WallexError as e:
            log.info("cancel %s: %s", cid, e)
        time.sleep(min(1.5, self.poll_s))
        try:
            return self._parse(self.client.get_order(cid))
        except WallexError:
            return ("UNKNOWN", 0.0, 0.0, 0.0)

    # ---------------------------------------------------------------- public
    def buy(self, symbol: str, notional: float) -> Fill | None:
        r = market_rules(self.markets, symbol)
        fills: list[tuple[float, float, float]] = []
        orders = []
        remaining = notional
        book = self.client.depth(symbol)
        best_bid, best_ask = float(book["bid"][0]["price"]), float(book["ask"][0]["price"])
        ref = best_ask
        if self.ex.entry_style == "maker_then_taker":
            px = best_bid
            q = _floor(remaining / px, r["qty_dec"])
            if q * px >= r["min_notional"]:
                cid = self._place(symbol, "BUY", px, q, r)
                orders.append(cid)
                st = self._wait(cid, self.ex.maker_timeout_s)
                if st[0] not in ("FILLED", "DONE"):
                    st = self._cancel_and_final(cid)
                if st[1] > 0:
                    fills.append((st[1], st[2], st[3]))
                    remaining -= st[1] * st[2]
        if remaining >= r["min_notional"]:
            book = self.client.depth(symbol)
            best_ask = float(book["ask"][0]["price"])
            if best_ask > ref * (1 + self.ex.max_entry_slippage_pct) and fills:
                log.info("%s: price ran away, keeping partial maker fill", symbol)
            elif best_ask <= ref * (1 + 2 * self.ex.max_entry_slippage_pct):
                px = best_ask * (1 + self.ex.max_entry_slippage_pct)
                q = _floor(remaining / px, r["qty_dec"])
                if q * px >= r["min_notional"]:
                    cid = self._place(symbol, "BUY", px, q, r)
                    orders.append(cid)
                    st = self._wait(cid, self.taker_wait_s)
                    if st[0] not in ("FILLED", "DONE"):
                        st = self._cancel_and_final(cid)
                    if st[1] > 0:
                        fills.append((st[1], st[2], st[3]))
        return self._combine(symbol, "BUY", fills, orders)

    def sell(self, symbol: str, qty: float) -> Fill | None:
        r = market_rules(self.markets, symbol)
        fills, orders = [], []
        remaining = _floor(qty, r["qty_dec"])
        for attempt in range(4):
            book = self.client.depth(symbol)
            best_bid = float(book["bid"][0]["price"])
            px = best_bid * (1 - self.ex.max_entry_slippage_pct * (1 + attempt))
            if remaining * px < r["min_notional"] * 0.5 or remaining <= 0:
                break
            cid = self._place(symbol, "SELL", px, remaining, r)
            orders.append(cid)
            st = self._wait(cid, min(20.0, self.taker_wait_s))
            if st[0] not in ("FILLED", "DONE"):
                st = self._cancel_and_final(cid)
            if st[1] > 0:
                fills.append((st[1], st[2], st[3]))
                remaining = _floor(remaining - st[1], r["qty_dec"])
            if remaining * best_bid < r["min_notional"] * 0.5:
                break
        return self._combine(symbol, "SELL", fills, orders)

    def _combine(self, symbol, side, fills, orders) -> Fill | None:
        q = sum(f[0] for f in fills)
        if q <= 0:
            return None
        avg = sum(f[0] * f[1] for f in fills) / q
        fee = sum(f[2] for f in fills)
        fee_tmn = fee if side == "SELL" else fee * avg  # buy fee in the base asset, sell fee in the quote (confirmed by the smoke test)
        if fee_tmn == 0:
            fee_tmn = q * avg * self.fee
        return Fill(symbol, side, q, avg, fee_tmn, pd.Timestamp.now(tz="UTC"), orders)

    def free_balance(self, asset: str) -> float | None:
        try:
            b = self.client.balances().get(asset)
        except WallexError as e:
            log.warning("balances failed: %s", e)
            return None
        if not b:
            return 0.0
        return float(b.get("value", 0)) - float(b.get("locked", 0))
