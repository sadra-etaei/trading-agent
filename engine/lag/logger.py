"""Gap logger (PLAN.md §15 item 1): measures how far Wallex USDT books trail Binance.

Every `interval_s`: one Binance bookTicker call for all pairs + the real Wallex /v1/depth books in
parallel (never the cached /v1/markets), plus Wallex /v1/trades for a few pairs (round-robin) to
measure how often each book actually trades. It never places orders.

  data/lag/YYYY-MM-DD/gaps.csv     one row per pair per snapshot
  data/lag/YYYY-MM-DD/trades.csv   new Wallex trades (deduplicated)
  data/lag/latest.json             newest row per pair, for live consumers
"""
from __future__ import annotations

import csv
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import httpx

from engine.lag.rules import bn_threshold, entry_signal, fair_price

log = logging.getLogger(__name__)
BN_BOOK = "https://api.binance.com/api/v3/ticker/bookTicker"
GAP_COLS = ["ts", "symbol", "bn_bid", "bn_ask", "wx_bid", "wx_ask", "wx_bid_qty", "wx_ask_qty", "wx_age_s",
            "basis", "fair", "gap_ask", "gap_bid", "spread", "bn_ret_3m", "signal"]


class Basis:
    """Rolling median of Wallex mid / Binance mid - 1 over `window_s` (time-based)."""

    def __init__(self, window_s: float):
        self.window_s, self.q = window_s, deque()

    def add(self, ts: float, value: float) -> None:
        self.q.append((ts, value))
        while self.q and self.q[0][0] < ts - self.window_s:
            self.q.popleft()

    def value(self) -> float | None:
        return median(v for _, v in self.q) if len(self.q) >= 5 else None


class History:
    """Binance mid history for the 3-minute return."""

    def __init__(self):
        self.q = deque()

    def add(self, ts: float, px: float) -> None:
        self.q.append((ts, px))
        while self.q and self.q[0][0] < ts - 400:
            self.q.popleft()

    def ret(self, ts: float, seconds: float = 180) -> float | None:
        old = [p for t, p in self.q if t <= ts - seconds]
        return self.q[-1][1] / old[-1] - 1 if old else None


class GapLogger:
    def __init__(self, cfg):
        self.cfg, self.lag = cfg, cfg.lag
        self.pairs = list(self.lag.pairs)
        self.base = cfg.wallex.base_url.rstrip("/")
        self.root = Path(cfg.data.dir) / "lag"
        self.http = httpx.Client(timeout=8, limits=httpx.Limits(max_connections=10, max_keepalive_connections=10))
        self.pool = ThreadPoolExecutor(7)
        self.basis = {s: Basis(self.lag.basis_window_min * 60) for s in self.pairs}
        self.hist = {s: History() for s in self.pairs}
        self.seen = {s: deque(maxlen=300) for s in self.pairs}

    def _binance(self) -> dict[str, tuple[float, float]]:
        r = self.http.get(BN_BOOK, params={"symbols": json.dumps(self.pairs, separators=(",", ":"))})
        r.raise_for_status()
        return {x["symbol"]: (float(x["bidPrice"]), float(x["askPrice"])) for x in r.json()}

    def _depth(self, s: str):
        t0 = time.time()
        try:
            res = self.http.get(f"{self.base}/v1/depth", params={"symbol": s}).json()["result"]
            bid, ask = res["bid"][0], res["ask"][0]
            return s, (float(bid["price"]), float(ask["price"]), float(bid["quantity"]), float(ask["quantity"]),
                       time.time() - t0)
        except Exception as e:  # one failed book must not stop the snapshot
            log.debug("depth %s: %s", s, e)
            return s, None

    def _trades(self, s: str, day: Path) -> None:
        rows = self.http.get(f"{self.base}/v1/trades", params={"symbol": s}).json()["result"]["latestTrades"]
        new = []
        for tr in reversed(rows):
            key = (tr.get("timestamp"), tr.get("price"), tr.get("quantity"), tr.get("isBuyOrder"))
            if key not in self.seen[s]:
                self.seen[s].append(key)
                new.append([round(time.time(), 2), s, tr.get("timestamp"), tr.get("price"), tr.get("quantity"),
                            int(bool(tr.get("isBuyOrder")))])
        _append(day / "trades.csv", ["ts", "symbol", "trade_ts", "price", "qty", "buy"], new)

    def snapshot(self, now: float) -> list[dict]:
        bn = self._binance()
        wx = dict(self.pool.map(self._depth, self.pairs))
        rows = []
        for s in self.pairs:
            if s not in bn or not wx.get(s):
                continue
            (bb, ba), (wb, wa, wbq, waq, age) = bn[s], wx[s]
            bmid, wmid = (bb + ba) / 2, (wb + wa) / 2
            self.hist[s].add(now, bmid)
            self.basis[s].add(now, wmid / bmid - 1)
            basis, r3 = self.basis[s].value(), self.hist[s].ret(now)
            fair = fair_price(bmid, basis) if basis is not None else None
            rows.append({
                "ts": round(now, 2), "symbol": s, "bn_bid": bb, "bn_ask": ba, "wx_bid": wb, "wx_ask": wa,
                "wx_bid_qty": wbq, "wx_ask_qty": waq, "wx_age_s": round(age, 2),
                "basis": _r(basis), "fair": _r(fair, 10),
                "gap_ask": _r(wa / fair - 1 if fair else None), "gap_bid": _r(wb / fair - 1 if fair else None),
                "spread": _r((wa - wb) / wmid), "bn_ret_3m": _r(r3),
                "signal": int(bool(fair and r3 is not None and entry_signal(self.lag, s, wa, fair, r3))),
            })
        return rows

    def run(self) -> None:
        log.info("gap logger: %d pairs every %ss into %s (no orders)", len(self.pairs), self.lag.interval_s, self.root)
        loop, errors, signals = 0, 0, 0
        while True:
            t0 = time.time()
            day = self.root / datetime.now(timezone.utc).strftime("%Y-%m-%d")
            day.mkdir(parents=True, exist_ok=True)
            try:
                rows = self.snapshot(t0)
                _append(day / "gaps.csv", GAP_COLS, [[r[c] for c in GAP_COLS] for r in rows])
                for r in rows:
                    if r["signal"]:
                        signals += 1
                        log.info("SIGNAL %s ask gap %.3f%% vs fair, Binance 3m %+.3f%% (needs %.2f%%)", r["symbol"],
                                 100 * r["gap_ask"], 100 * r["bn_ret_3m"], 100 * bn_threshold(self.lag, r["symbol"]))
                _atomic(self.root / "latest.json", json.dumps({r["symbol"]: r for r in rows}))
                for k in range(self.lag.trades_per_loop):
                    self._trades(self.pairs[(loop * self.lag.trades_per_loop + k) % len(self.pairs)], day)
                errors = 0
            except Exception as e:
                errors += 1
                log.warning("gap logger error #%d: %s", errors, e)
                time.sleep(min(60, 5 * errors))
            loop += 1
            if loop % 120 == 0:
                log.info("gap logger alive: %d snapshots, %d signals, last loop %.2fs", loop, signals, time.time() - t0)
            time.sleep(max(0.0, self.lag.interval_s - (time.time() - t0)))


def _r(x, nd: int = 6):
    return None if x is None else round(x, nd)


def _append(path: Path, cols: list[str], rows: list[list]) -> None:
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(cols)
        w.writerows(rows)


def _atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)
    except OSError:  # Windows: a reader has the file open; the next snapshot will succeed
        pass
