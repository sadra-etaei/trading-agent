"""Fair-value shadow trader: live recorded books -> PairEngine (paper maker orders) for several
parameter variants at once. No API key, no orders, no extra API load: it consumes the
recorder's output (data/live/latest.json + trades/*.jsonl).

Each snapshot is processed `delay_s` after it was taken, so the public trades of that window have
already been recorded (the recorder polls each pair's trades every ~10-15s). Decisions are
therefore identical to the replay backtester on the same data.

python cli.py fv-shadow           -> runs/fv-shadow-<time>/ (decisions.jsonl, trades_<variant>.csv, status.md)
"""
from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.decision_log import DecisionLog
from engine.fv.data import bases_available, fair_value_frame, live_days, load_tob
from engine.fv.engine import FVParams, PairEngine, Quote
from engine.fv.sim import FEE_SCENARIOS, GRID_THETA, summarize
from engine.wallex import markets as mk

log = logging.getLogger(__name__)
OFFSET_WINDOW_S = 7200
STALE_S = 900


def variants() -> dict[str, FVParams]:
    out = {}
    for fee_name, (mf, tf) in FEE_SCENARIOS.items():
        for th in GRID_THETA:
            out[f"{'m10' if mf == 0.001 else 'm25'}_th{int(th * 1e4)}"] = FVParams(
                theta_in=th, maker_fee=mf, taker_fee=tf, min_profit=2 * mf + 0.0005)
    return out


class FairValueState:
    """Incremental version of fv.data.fair_value_frame for one base asset."""

    def __init__(self, base: str, max_usd_spread: float):
        self.base = base
        self.hist: deque = deque()  # (t, log(mid/F))
        self.last_change = {}
        self.last_val = {}
        self.max_usd_spread = max_usd_spread

    def _touch(self, key, val, t):
        if self.last_val.get(key) != val:
            self.last_val[key] = val
            self.last_change[key] = t

    def seed(self, d: pd.DataFrame) -> None:
        if d.empty:
            return
        t = d.index.as_unit("ns").asi8 / 1e9
        raw = np.log(d["mid"] / d["F"]).to_numpy()
        for ti, r in zip(t[-2000:], raw[-2000:]):
            self.hist.append((ti, r))

    def quote(self, t: float, books: dict) -> Quote | None:
        tmn, usd = books.get(f"{self.base}TMN"), books.get(f"{self.base}USDT")
        usdt = books.get("USDTTMN")
        if not (tmn and usd and usdt) or not all(x[0] and x[1] for x in (tmn, usd, usdt)):
            return None
        (bid, bq), (ask, aq) = tmn[0][0], tmn[1][0]
        u_mid = (usd[0][0][0] + usd[1][0][0]) / 2
        usdt_mid = (usdt[0][0][0] + usdt[1][0][0]) / 2
        F = u_mid * usdt_mid
        mid = (bid + ask) / 2
        if not (bid > 0 and ask > bid and F > 0):
            return None
        while self.hist and self.hist[0][0] < t - OFFSET_WINDOW_S:
            self.hist.popleft()
        offset = float(np.median([h[1] for h in self.hist])) if len(self.hist) >= 30 else None
        self.hist.append((t, math.log(mid / F)))
        self._touch("usd", u_mid, t)
        self._touch("usdt", usdt_mid, t)
        if offset is None:
            return None
        usd_spread = (usd[1][0][0] - usd[0][0][0]) / u_mid
        fresh = (t - self.last_change["usd"] <= STALE_S and t - self.last_change["usdt"] <= STALE_S
                 and usd_spread <= self.max_usd_spread)
        return Quote(t, bid, ask, bq, aq, F, offset, math.log(bid / F) - offset, fresh)


class TradeTail:
    """Follows data/live/<day>/trades/SYM.jsonl files (handles day rollover)."""

    def __init__(self, data_dir: str, symbol: str):
        self.data_dir, self.symbol = data_dir, symbol
        self.pos: dict[str, int] = {}
        self.buf: list[tuple] = []

    def poll(self) -> None:
        for day in live_days(self.data_dir)[-2:]:
            p = day / "trades" / f"{self.symbol}.jsonl"
            if not p.exists():
                continue
            key = str(p)
            with open(p, encoding="utf-8") as f:
                f.seek(self.pos.get(key, 0))
                while True:
                    line = f.readline()
                    if not line or not line.endswith("\n"):
                        break
                    self.pos[key] = f.tell()
                    try:
                        r = json.loads(line)
                        tt = pd.Timestamp(r["trade_ts"]).timestamp()
                        self.buf.append((tt, float(r["price"]), float(r["qty"]), bool(r["buy"])))
                    except (ValueError, KeyError, TypeError):
                        continue
        self.buf.sort()

    def take_until(self, t: float) -> list[tuple]:
        k = 0
        while k < len(self.buf) and self.buf[k][0] <= t:
            k += 1
        out, self.buf = self.buf[:k], self.buf[k:]
        return out


def run_shadow(cfg, delay_s: float = 20.0) -> None:
    data_dir = cfg.data.dir
    latest = Path(data_dir) / "live" / "latest.json"
    tob = load_tob(data_dir)
    bases = bases_available(tob) if not tob.empty else []
    if not bases:
        raise SystemExit("no recordings yet - start `python cli.py record` first")
    markets = mk.load(cfg)
    vs = variants()
    run_id = "fv-shadow-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = Path(cfg.run.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dlog = DecisionLog(run_dir, run_id, "fv-shadow")
    (run_dir / "meta.json").write_text(json.dumps({"run_id": run_id, "bases": bases, "delay_s": delay_s,
                                                   "variants": {k: asdict(v) for k, v in vs.items()},
                                                   "started_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    states = {}
    for b in bases:
        st = FairValueState(b, FVParams().max_usd_spread)
        st.seed(fair_value_frame(tob, b))  # warm the 2h offset window from recordings
        states[b] = st
    engines = {(v, b): PairEngine(f"{b}TMN", p, mk.rules(markets, f"{b}TMN")["price_dec"]) for v, p in vs.items() for b in bases}
    tails = {b: TradeTail(data_dir, f"{b}TMN") for b in bases}
    for t in tails.values():
        t.poll()
        t.buf = [x for x in t.buf if x[0] > time.time() - 60]
    pending: deque = deque()
    last_ts, last_status, last_beat = 0.0, 0.0, time.time()
    n_snap, n_quotes, last_gap = 0, 0, {}
    t_start = time.time()
    log.info("fv shadow %s: %d pairs x %d variants, delay %.0fs", run_id, len(bases), len(vs), delay_s)
    while True:
        try:
            snap = json.loads(latest.read_text(encoding="utf-8"))
            if snap["ts"] > last_ts:
                pending.append(snap)
                last_ts = snap["ts"]
        except (OSError, ValueError):
            pass
        for t in tails.values():
            t.poll()
        now = time.time()
        while pending and pending[0]["ts"] <= now - delay_s:
            snap = pending.popleft()
            ts = snap["ts"]
            n_snap += 1
            for b in bases:
                q = states[b].quote(ts, snap["b"])
                trades = tails[b].take_until(ts)
                if q is None:
                    continue
                n_quotes += 1
                last_gap[b] = round(100 * q.gap_bid, 3) if q.fresh else None
                for v in vs:
                    for e in engines[(v, b)].step(q, trades):
                        if e["kind"] in ("place", "fill", "exit", "cancel"):
                            dlog.write(f"fv_{e.pop('kind')}", ts, variant=v, pair=f"{b}TMN", **e)
        if now - last_beat > 300:
            last_beat = now
            log.info("alive: %d snapshots, %d quotes, gap_bid%% %s", n_snap, n_quotes, last_gap)
        if now - last_status > 60:
            last_status = now
            _status(run_dir, engines, vs, (now - t_start) / 3600, float(cfg.risk.initial_equity_tmn))
        time.sleep(1.0)


def _status(run_dir: Path, engines, vs, hours: float, equity: float) -> None:
    L = [f"# Fair-value shadow - {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC ({hours:.2f} h)", "",
         "| variant | trades | /day | win % | mean net/trade | P&L (TMN) | return on 5M | open positions | orders | fills |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for v in vs:
        trs = [t for (vv, _), e in engines.items() if vv == v for t in e.trades]
        df = pd.DataFrame([asdict(t) for t in trs])
        if len(df):
            df.to_csv(run_dir / f"trades_{v}.csv", index=False)
        s = summarize(df, max(hours, 1e-6), equity)
        opens = sum(1 for (vv, _), e in engines.items() if vv == v and e.pos is not None)
        orders = sum(e.stats["orders"] for (vv, _), e in engines.items() if vv == v)
        fills = sum(e.stats["fills"] for (vv, _), e in engines.items() if vv == v)
        L.append(f"| {v} | {s['trades']} | {s['trades_per_day']} | {s.get('win_rate_pct', '-')} | "
                 f"{s.get('mean_net_pct', '-')} | {s['pnl_tmn']:+,} | {s['return_pct']:+.3f}% | {opens} | {orders} | {fills} |")
    (run_dir / "status.md").write_text("\n".join(L) + "\n", encoding="utf-8")
