"""Fair-value gap LIVE executor (real maker orders). Same PairEngine as replay and shadow; fills come
from real order status (external_fills=True).

Every `loop_s`:
  1. newest books from the recorder (data/live/latest.json); if older than `max_snapshot_age_s` no new
     orders are placed, and all orders are cancelled if it stays stale.
  2. per pair: poll our open order -> report fills to the engine; engine.step(quote) -> events;
     events -> API calls: place / re-quote (cancel + place) / cancel / taker exit.
  3. account risk: max positions, kill switch on drawdown, end-of-challenge flatten to TMN.
Everything is written to the decision log with the client order ids.

python cli.py fv-live --yes
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.broker.brokers import WallexBroker
from engine.decision_log import DecisionLog
from engine.fv.engine import FVParams, PairEngine
from engine.fv.shadow import FairValueState
from engine.wallex import markets as mk
from engine.wallex.allowed import assert_allowed
from engine.wallex.client import WallexClient, WallexError

log = logging.getLogger(__name__)


def _floor(x: float, dec: int) -> float:
    f = 10 ** dec
    return math.floor(x * f + 1e-9) / f


def params_from_cfg(cfg) -> FVParams:
    fv = dict(cfg.get("fv", {}))
    keys = FVParams.__dataclass_fields__
    return FVParams(**{k: v for k, v in fv.items() if k in keys})


class LiveFV:
    def __init__(self, cfg, client: WallexClient, bases: list[str], markets: dict, run_dir: Path, dlog=None):
        self.cfg, self.client, self.bases, self.markets = cfg, client, bases, markets
        fv = cfg.get("fv", {})
        self.p = params_from_cfg(cfg)
        self.max_positions = int(fv.get("max_positions", 4))
        self.kill_dd = float(fv.get("max_drawdown_kill", cfg.risk.max_drawdown_kill))
        self.max_age = float(fv.get("max_snapshot_age_s", 10))
        self.broker = WallexBroker(client, cfg, markets)
        self.engines = {b: PairEngine(f"{b}TMN", self.p, mk.rules(markets, f"{b}TMN")["price_dec"], external_fills=True)
                        for b in bases}
        self.states = {b: FairValueState(b, self.p.max_usd_spread) for b in bases}
        self.orders: dict[str, dict] = {}  # base -> {cid, side, price, qty, t}
        self.run_dir = run_dir
        self.log = dlog or DecisionLog(run_dir, run_dir.name, "fv-live")
        self.end_ts = pd.Timestamp(cfg.challenge.end).timestamp() if cfg.challenge.end else None
        self.no_entry_s = cfg.challenge.no_new_entries_before_end_min * 60
        self.flatten_s = cfg.challenge.flatten_before_end_min * 60
        self.initial = float(cfg.risk.initial_equity_tmn)
        self.peak = self.initial
        self.halted = False
        self.last_status = 0.0

    # ---------------------------------------------------------------- order plumbing
    def _place(self, b: str, side: str, price: float, qty: float, t: float) -> None:
        sym = f"{b}TMN"
        assert_allowed(sym)
        r = mk.rules(self.markets, sym)
        qty = _floor(qty, r["qty_dec"])
        if qty <= 0 or qty * price < r["min_notional"]:
            self.log.write("fv_skip", t, pair=sym, side=side, reason="below min notional", qty=qty, price=price)
            return
        cid = f"fv-{b[:5]}-{uuid.uuid4().hex[:12]}"
        try:
            self.client.place_order(sym, side, f"{price:.{r['price_dec']}f}", f"{qty:.{r['qty_dec']}f}", cid)
        except WallexError as e:
            self.log.write("fv_order_error", t, pair=sym, side=side, price=price, qty=qty, error=str(e)[:300])
            return
        self.orders[b] = {"cid": cid, "side": side, "price": price, "qty": qty, "t": t}
        self.log.write("fv_order", t, pair=sym, side=side, price=price, qty=qty, client_id=cid)

    def _status(self, cid: str) -> tuple[str, float, float]:
        st, q, px, _ = WallexBroker._parse(self.client.get_order(cid))
        return st, q, px

    def _cancel(self, b: str, t: float, F: float) -> None:
        """Cancel our order on b; a partial fill found at cancel time is reported to the engine."""
        o = self.orders.pop(b, None)
        if not o:
            return
        try:
            self.client.cancel_order(o["cid"])
        except WallexError as e:
            log.info("cancel %s: %s", o["cid"], e)
        try:
            st, q, px = self._status(o["cid"])
        except WallexError:
            st, q, px = "UNKNOWN", 0.0, 0.0
        self.log.write("fv_cancel", t, pair=f"{b}TMN", client_id=o["cid"], status=st, executed_qty=q)
        if q > 0:
            self._on_fill(b, o["side"], px or o["price"], q, t, F, partial=True)

    def _on_fill(self, b: str, side: str, px: float, qty: float, t: float, F: float, partial=False) -> None:
        tr = self.engines[b].on_fill(side, px, qty, t, F)
        self.log.write("fv_fill", t, pair=f"{b}TMN", side=side, price=px, qty=qty, partial=partial,
                       pnl_tmn=tr.pnl_tmn if tr else None)

    def _poll(self, b: str, t: float, F: float) -> None:
        o = self.orders.get(b)
        if not o:
            return
        try:
            st, q, px = self._status(o["cid"])
        except WallexError as e:
            log.warning("status %s: %s", o["cid"], e)
            return
        if st in ("FILLED", "DONE"):
            self.orders.pop(b)
            self._on_fill(b, o["side"], px or o["price"], q or o["qty"], t, F)
        elif st in ("CANCELED", "CANCELLED", "REJECTED", "EXPIRED"):
            self.orders.pop(b)
            self.engines[b].order = None
            if q > 0:
                self._on_fill(b, o["side"], px or o["price"], q, t, F, partial=True)

    def _taker_exit(self, b: str, reason: str, t: float, F: float) -> None:
        self._cancel(b, t, F)
        eng = self.engines[b]
        if eng.pos is None:
            return
        qty = eng.pos["qty"]
        free = self.broker.free_balance(b)
        if free is not None:
            qty = min(qty, free)  # buy fees may be charged in the base asset
        fill = self.broker.sell(f"{b}TMN", qty)
        if fill is None:
            self.log.write("fv_exit_failed", t, pair=f"{b}TMN", reason=reason)
            eng.exiting = False  # retry next loop
            return
        tr = eng.on_taker_exit(fill.price, t, reason, F)
        self.log.write("fv_exit", t, pair=f"{b}TMN", reason=reason, price=fill.price, qty=fill.qty, orders=fill.orders,
                       pnl_tmn=tr.pnl_tmn if tr else None)

    # ---------------------------------------------------------------- main tick
    def equity(self, books: dict) -> float:
        tmn = self.broker.free_balance("TMN") or 0.0
        locked_buys = sum(o["price"] * o["qty"] for o in self.orders.values() if o["side"] == "BUY")
        coins = 0.0
        for b, e in self.engines.items():
            if e.pos is not None and f"{b}TMN" in books:
                coins += e.pos["qty"] * books[f"{b}TMN"][0][0][0]
        return tmn + locked_buys + coins

    def tick(self, snap: dict, now: float) -> None:
        t, books = snap["ts"], snap["b"]
        stale = now - t > self.max_age
        in_flatten = self.end_ts is not None and now >= self.end_ts - self.flatten_s
        no_entries = stale or self.halted or (self.end_ts is not None and now >= self.end_ts - self.no_entry_s)
        open_pos = sum(1 for e in self.engines.values() if e.pos is not None)
        pending_buys = sum(1 for o in self.orders.values() if o["side"] == "BUY")
        for b in self.bases:
            eng = self.engines[b]
            q = self.states[b].quote(t, books)
            if q is None:
                continue
            self._poll(b, now, q.F)
            if in_flatten or self.halted:
                if eng.pos is not None:
                    self._taker_exit(b, "kill_switch" if self.halted else "flatten_end", now, q.F)
                elif b in self.orders:
                    self._cancel(b, now, q.F)
                continue
            if eng.pos is None and b not in self.orders and (no_entries or open_pos + pending_buys >= self.max_positions):
                q.fresh = False  # blocks new entries without touching the engine's logic
            if stale and b in self.orders and self.orders[b]["side"] == "BUY":
                self._cancel(b, now, q.F)
                eng.order = None
            for e in eng.step(q):
                k = e["kind"]
                if k in ("place", "requote"):
                    if b in self.orders:
                        self._cancel(b, now, q.F)
                    if k == "place" or eng.order is not None:
                        side = e["side"]
                        qty = self.p.notional_tmn / e["price"] if side == "BUY" else (eng.pos["qty"] if eng.pos else 0.0)
                        if side == "SELL":
                            free = self.broker.free_balance(b)
                            qty = min(qty, free) if free is not None else qty
                        self._place(b, side, e["price"], qty, now)
                        if side == "BUY" and b in self.orders:
                            pending_buys += 1
                elif k == "cancel":
                    self._cancel(b, now, q.F)
                elif k == "exit_taker":
                    self._taker_exit(b, e["reason"], now, q.F)
        if now - self.last_status > 60:
            self.last_status = now
            eq = self.equity(books)
            self.peak = max(self.peak, eq)
            dd = 1 - eq / self.peak if self.peak else 0.0
            if not self.halted and dd >= self.kill_dd:
                self.halted = True
                self.log.write("risk_event", now, event="kill_switch", drawdown=dd, equity=eq)
            self._write_status(eq, dd)

    def _write_status(self, eq: float, dd: float) -> None:
        trades = [asdict(tr) for e in self.engines.values() for tr in e.trades]
        if trades:
            pd.DataFrame(trades).to_csv(self.run_dir / "trades.csv", index=False)
        with open(self.run_dir / "equity.csv", "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("ts,equity,drawdown,positions,open_orders\n")
            f.write(f"{datetime.now(timezone.utc).isoformat()},{eq:.0f},{dd:.6f},"
                    f"{sum(e.pos is not None for e in self.engines.values())},{len(self.orders)}\n")
        log.info(f"equity {eq:,.0f} ({(eq / self.initial - 1) * 100:+.2f}%), dd {dd:.2%}, trades {len(trades)}, "
                 f"positions {[b for b, e in self.engines.items() if e.pos]}, orders {list(self.orders)}")

    def cancel_all(self) -> None:
        for b in list(self.orders):
            self._cancel(b, time.time(), 0.0)


def run_fv_live(cfg) -> None:
    from engine.fv.data import bases_available, fair_value_frame, load_tob

    client = WallexClient.from_config(cfg)
    markets = mk.load(cfg, client, refresh=True)
    tob = load_tob(cfg.data.dir)
    bases = [b for b in (cfg.get("fv", {}).get("bases") or bases_available(tob))]
    run_dir = Path(cfg.run.output_dir) / ("fv-live-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(json.dumps({"mode": "fv-live", "bases": bases, "params": asdict(params_from_cfg(cfg)),
                                                   "challenge_end": cfg.challenge.end}, indent=2, default=str))
    lv = LiveFV(cfg, client, bases, markets, run_dir)
    for b in bases:
        lv.states[b].seed(fair_value_frame(tob, b))
    latest = Path(cfg.data.dir) / "live" / "latest.json"
    log.info("fv LIVE on %s, params %s", bases, asdict(lv.p))
    last_ts = 0.0
    try:
        while True:
            now = time.time()
            try:
                snap = json.loads(latest.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                snap = None
            if snap and snap["ts"] > last_ts:
                last_ts = snap["ts"]
                try:
                    lv.tick(snap, now)
                except (WallexError, OSError, KeyError, ValueError) as e:
                    log.exception("tick error: %s", e)
                    lv.log.write("error", now, error=str(e)[:300])
            elif snap and now - snap["ts"] > 30 and lv.orders:
                log.warning("recorder snapshot stale for %.0fs -> cancelling all orders", now - snap["ts"])
                lv.cancel_all()
            if lv.end_ts and now >= lv.end_ts and all(e.pos is None for e in lv.engines.values()) and not lv.orders:
                log.info("challenge over and flat in TMN -> stopping")
                break
            time.sleep(float(cfg.get("fv", {}).get("loop_s", 1.0)))
    finally:
        lv.cancel_all()
