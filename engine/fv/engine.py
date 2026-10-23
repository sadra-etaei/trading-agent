"""Per-pair fair-value gap engine, shared by the replay backtester (fv/sim.py) and the shadow
trader (fv/shadow.py): identical decisions in both, by construction.

Per pair, one position at a time:
  ENTRY  when gap_bid <= -theta_in (inputs fresh): post a maker BUY at
         P = min(best_bid + tick, F·e^(offset − theta_in)); re-quote when the target moves;
         cancel when gap_bid > -theta_cancel (opportunity gone).
  EXIT   after a fill: maker SELL at Q = max(F·e^(offset + exit_edge), entry·(1 + min_profit), best_bid + tick),
         re-quoted every snapshot; taker exit at the best bid on time stop or stop-loss.

Conservative maker fill rule (orders become active `latency_s` after the decision):
  buy at P fills if a trade prints strictly below P, or the best ask drops to <= P, or
  seller-initiated volume AT P after activation exceeds the queue ahead of us (the quantity
  resting at P when we joined; 0 if we improved the bid; infinite if we sit below the best bid,
  i.e. only a trade-through fills us). Sells mirror this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class FVParams:
    theta_in: float = 0.004
    theta_cancel: float = 0.0015
    exit_edge: float = 0.0
    min_profit: float = 0.0
    hold_max_s: float = 1800
    stop_pct: float = 0.006
    requote_pct: float = 0.0005
    latency_s: float = 2.0
    notional_tmn: float = 600_000
    maker_fee: float = 0.0025
    taker_fee: float = 0.0025
    max_usd_spread: float = 0.003


@dataclass
class FVTrade:
    pair: str
    entry_ts: str
    exit_ts: str
    entry_px: float
    exit_px: float
    qty: float
    exit_kind: str           # maker_target | time_stop | stop_loss | end
    gap_at_signal: float
    fair_move_pct: float     # change of F during the trade (adverse selection check)
    wait_fill_s: float
    hold_s: float
    fees_tmn: float
    pnl_tmn: float
    ret_pct: float


@dataclass
class Quote:
    t: float          # epoch seconds
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    F: float
    offset: float
    gap_bid: float
    fresh: bool


class Order:
    def __init__(self, side, price, placed, active, queue_ahead):
        self.side, self.price, self.placed, self.active, self.queue = side, price, placed, active, queue_ahead
        self.vol_at_px = 0.0
        self.gap = 0.0
        self.F0 = 0.0


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).isoformat()


def round_px(px: float, dec: int, side: str) -> float:
    f = 10 ** dec
    return (math.floor(px * f) if side == "BUY" else math.ceil(px * f)) / f


class PairEngine:
    """Feed it quotes in time order with the public trades that printed since the previous quote.
    `step` returns events (dicts) describing every decision, for the decision log."""

    def __init__(self, pair: str, p: FVParams, price_dec: int = 0, external_fills: bool = False):
        """external_fills=True (live trading): fills are reported by the executor via on_fill /
        on_taker_exit from real order status instead of the simulated fill rule."""
        self.pair, self.p, self.dec = pair, p, price_dec
        self.external = external_fills
        self.exiting = False
        self.tick = 10.0 ** (-price_dec)
        self.order: Order | None = None
        self.pos: dict | None = None
        self.prev_t: float | None = None
        self.trades: list[FVTrade] = []
        self.stats = {"orders": 0, "requotes": 0, "cancels": 0, "fills": 0, "signals": 0}

    # ---------------------------------------------------------------- fills
    def _filled(self, o: Order, t0: float, t1: float, q: Quote, trades) -> bool:
        if t1 < o.active:
            return False
        lo = max(t0, o.active)
        for tt, price, qty, buy in trades:
            if tt <= lo or tt > t1:
                continue
            if o.side == "BUY":
                if price < o.price:
                    return True
                if price == o.price and not buy:
                    o.vol_at_px += qty
            else:
                if price > o.price:
                    return True
                if price == o.price and buy:
                    o.vol_at_px += qty
            if o.vol_at_px > o.queue:
                return True
        return (q.ask <= o.price) if o.side == "BUY" else (q.bid >= o.price)

    def _close(self, px: float, t: float, kind: str, F_now: float, maker: bool) -> FVTrade:
        p, pos = self.p, self.pos
        qty = pos["qty"]
        fees = qty * pos["entry_px"] * p.maker_fee + qty * px * (p.maker_fee if maker else p.taker_fee)
        pnl = qty * (px - pos["entry_px"]) - fees
        tr = FVTrade(self.pair, iso(pos["entry_t"]), iso(t), pos["entry_px"], px, qty, kind, float(pos["gap"]),
                     100 * (F_now / pos["F0"] - 1), float(pos["wait"]), float(t - pos["entry_t"]), fees, pnl,
                     100 * pnl / (qty * pos["entry_px"]))
        self.trades.append(tr)
        self.pos = None
        return tr

    # ---------------------------------------------------------------- main step
    def step(self, q: Quote, trades=()) -> list[dict]:
        p, ev = self.p, []
        t = q.t
        prev_t = q.t if self.prev_t is None else self.prev_t

        # 1) resting order fill check over (prev_t, t]
        o = self.order
        if not self.external and o is not None and self._filled(o, prev_t, t, q, trades):
            if o.side == "BUY":
                self.pos = {"entry_px": o.price, "qty": p.notional_tmn / o.price, "entry_t": t,
                            "wait": t - o.placed, "gap": o.gap, "F0": o.F0}
                self.stats["fills"] += 1
                ev.append({"kind": "fill", "side": "BUY", "price": o.price, "qty": self.pos["qty"], "wait_s": t - o.placed})
            else:
                tr = self._close(o.price, t, "maker_target", q.F, maker=True)
                ev.append({"kind": "exit", "reason": "maker_target", "price": o.price, "pnl_tmn": tr.pnl_tmn, "ret_pct": tr.ret_pct})
            self.order = None

        # 2) position management
        if self.pos is not None:
            stop = q.bid <= self.pos["entry_px"] * (1 - p.stop_pct)
            if stop or t - self.pos["entry_t"] >= p.hold_max_s:
                kind = "stop_loss" if stop else "time_stop"
                if self.external:  # executor cancels the resting sell and sells aggressively
                    if not self.exiting:
                        self.exiting = True
                        self.order = None
                        ev.append({"kind": "exit_taker", "reason": kind, "ref_bid": q.bid})
                    self.prev_t = t
                    return ev
                tr = self._close(q.bid, t, kind, q.F, maker=False)
                self.order = None
                ev.append({"kind": "exit", "reason": kind, "price": q.bid, "pnl_tmn": tr.pnl_tmn, "ret_pct": tr.ret_pct})
            else:
                target = max(q.F * math.exp(q.offset + p.exit_edge), self.pos["entry_px"] * (1 + p.min_profit), q.bid + self.tick)
                target = round_px(target, self.dec, "SELL")
                if self.order is None or abs(target / self.order.price - 1) > p.requote_pct:
                    kind = "requote" if self.order is not None else "place"
                    if self.order is not None:
                        self.stats["requotes"] += 1
                    queue = q.ask_qty if target == q.ask else (0.0 if target < q.ask else float("inf"))
                    self.order = Order("SELL", target, t, t + p.latency_s, queue)
                    ev.append({"kind": kind, "side": "SELL", "price": target, "fair": q.F})
        # 3) entry management (flat)
        else:
            want = q.fresh and q.gap_bid <= -p.theta_in
            if self.order is not None and self.order.side == "BUY" and (not q.fresh or q.gap_bid > -p.theta_cancel):
                ev.append({"kind": "cancel", "side": "BUY", "price": self.order.price, "gap": q.gap_bid})
                self.order = None
                self.stats["cancels"] += 1
            if want:
                target = round_px(min(q.bid + self.tick, q.F * math.exp(q.offset - p.theta_in)), self.dec, "BUY")
                if target < q.ask and (self.order is None or abs(target / self.order.price - 1) > p.requote_pct):
                    if self.order is None:
                        self.stats["signals"] += 1
                        self.stats["orders"] += 1
                        kind = "place"
                    else:
                        self.stats["requotes"] += 1
                        kind = "requote"
                    queue = q.bid_qty if target == q.bid else (0.0 if target > q.bid else float("inf"))
                    self.order = Order("BUY", target, t, t + p.latency_s, queue)
                    self.order.gap, self.order.F0 = q.gap_bid, q.F
                    ev.append({"kind": kind, "side": "BUY", "price": target, "fair": q.F, "gap": q.gap_bid,
                               "offset": q.offset, "best_bid": q.bid, "best_ask": q.ask})
        self.prev_t = t
        return ev

    # ---------------------------------------------------------------- live (external fills)
    def on_fill(self, side: str, price: float, qty: float, t: float, F_now: float) -> FVTrade | None:
        """Real fill of our resting maker order (possibly partial, reported once the order is done)."""
        o = self.order
        self.order = None
        if side == "BUY":
            self.pos = {"entry_px": price, "qty": qty, "entry_t": t, "wait": t - (o.placed if o else t),
                        "gap": o.gap if o else 0.0, "F0": o.F0 if o else F_now}
            self.stats["fills"] += 1
            return None
        if self.pos is None:
            return None
        if qty < self.pos["qty"] * 0.999:  # partial exit: keep the rest
            self.pos["qty"] -= qty
            return None
        return self._close(price, t, "maker_target", F_now, maker=True)

    def on_taker_exit(self, price: float, t: float, reason: str, F_now: float) -> FVTrade | None:
        self.exiting = False
        if self.pos is None:
            return None
        return self._close(price, t, reason, F_now, maker=False)

    def finish(self, q: Quote) -> list[dict]:
        """Close any open position at the end of data (taker)."""
        if self.pos is None:
            return []
        tr = self._close(q.bid, q.t, "end", q.F, maker=False)
        self.order = None
        return [{"kind": "exit", "reason": "end", "price": q.bid, "pnl_tmn": tr.pnl_tmn}]
