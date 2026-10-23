"""Live market recorder (Wallex has no history of order books or bid/ask).

Note: /v1/markets bid/ask is a cached summary (it matched the real book in <6% of checks and often
did not change for minutes), so it is NOT used. Instead, every `interval_s` the recorder fetches the
real order book (/v1/depth) of every pair in parallel over a keep-alive connection pool
(15 books in ~1s), plus public trades for a few pairs per loop (round-robin; /v1/trades returns
the last 50 trades, duplicates dropped).

Output (UTC dates, gzip members appended so files stay readable if the process is killed):
  data/live/YYYY-MM-DD/books.jsonl.gz    {"ts", "b": {SYM: [[[bid_px, qty] x L], [[ask_px, qty] x L]]}}
  data/live/YYYY-MM-DD/trades/SYM.jsonl  {"ts", "trade_ts", "price", "qty", "buy"}
  data/live/latest.json                  newest books snapshot (atomically replaced), for live consumers
"""
from __future__ import annotations

import gzip
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


def book_features(book: dict, levels: int = 10) -> dict:
    """Compact order book summary used by entry timing."""
    asks = [(float(a["price"]), float(a["quantity"])) for a in book.get("ask", [])[:levels]]
    bids = [(float(b["price"]), float(b["quantity"])) for b in book.get("bid", [])[:levels]]
    if not asks or not bids:
        return {}
    best_ask, best_bid = asks[0][0], bids[0][0]
    mid = (best_ask + best_bid) / 2
    ask_val = sum(p * q for p, q in asks)
    bid_val = sum(p * q for p, q in bids)
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "spread_pct": (best_ask - best_bid) / mid,
        "imbalance": (bid_val - ask_val) / (bid_val + ask_val) if bid_val + ask_val else 0.0,
        "bid_value_tmn": bid_val,
        "ask_value_tmn": ask_val,
    }


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class Recorder:
    def __init__(self, base_url: str, data_dir: str, symbols: list[str], interval_s: float = 2.0,
                 levels: int = 5, workers: int = 5, trades_per_loop: int = 3):
        self.base = base_url.rstrip("/")
        self.root = Path(data_dir) / "live"
        self.symbols = symbols
        self.interval_s = interval_s
        self.levels = levels
        self.trades_per_loop = trades_per_loop
        self.http = httpx.Client(timeout=10, limits=httpx.Limits(max_connections=workers + 2,
                                                                  max_keepalive_connections=workers + 2))
        self.pool = ThreadPoolExecutor(workers)
        self.seen = {s: deque(maxlen=300) for s in symbols}

    def _depth(self, s: str):
        try:
            r = self.http.get(f"{self.base}/v1/depth", params={"symbol": s})
            res = r.json()["result"]
            L = self.levels
            return s, [[[_f(x["price"]), _f(x["quantity"])] for x in res.get("bid", [])[:L]],
                       [[_f(x["price"]), _f(x["quantity"])] for x in res.get("ask", [])[:L]]]
        except Exception as e:  # one failed book must not stop the snapshot
            log.debug("depth %s: %s", s, e)
            return s, None

    def _trades(self, s: str, day: Path) -> None:
        r = self.http.get(f"{self.base}/v1/trades", params={"symbol": s})
        rows = r.json()["result"]["latestTrades"]
        p = day / "trades" / f"{s}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            for tr in reversed(rows):
                key = (tr.get("timestamp"), tr.get("price"), tr.get("quantity"), tr.get("isBuyOrder"))
                if key in self.seen[s]:
                    continue
                self.seen[s].append(key)
                f.write(json.dumps({"ts": round(time.time(), 2), "trade_ts": tr.get("timestamp"), "price": _f(tr.get("price")),
                                    "qty": _f(tr.get("quantity")), "buy": bool(tr.get("isBuyOrder"))}, separators=(",", ":")) + "\n")

    def run(self) -> None:
        log.info("recording real order books of %d pairs every %.1fs into %s", len(self.symbols), self.interval_s, self.root)
        loop, errors, failed_books = 0, 0, 0
        while True:
            t0 = time.time()
            day = self.root / datetime.now(timezone.utc).strftime("%Y-%m-%d")
            day.mkdir(parents=True, exist_ok=True)
            try:
                books = dict(self.pool.map(self._depth, self.symbols))
                failed_books += sum(v is None for v in books.values())
                rec = {"ts": round(t0, 2), "b": {s: v for s, v in books.items() if v}}
                line = json.dumps(rec, separators=(",", ":"))
                with gzip.open(day / "books.jsonl.gz", "at", encoding="utf-8") as f:
                    f.write(line + "\n")
                tmp = self.root / "latest.json.tmp"  # newest snapshot for live consumers (shadow trader)
                try:
                    tmp.write_text(line, encoding="utf-8")
                    tmp.replace(self.root / "latest.json")
                except OSError:  # Windows: a reader has the file open; next snapshot will succeed
                    pass
                for k in range(self.trades_per_loop):
                    self._trades(self.symbols[(loop * self.trades_per_loop + k) % len(self.symbols)], day)
                errors = 0
            except Exception as e:
                errors += 1
                log.warning("recorder error #%d: %s", errors, e)
                time.sleep(min(60, 5 * errors))
            loop += 1
            if loop % 300 == 0:
                log.info("recorder alive: %d snapshots, %d failed book fetches, last loop %.2fs", loop, failed_books, time.time() - t0)
            time.sleep(max(0.0, self.interval_s - (time.time() - t0)))


def record(base_url: str, data_dir: str, symbols: list[str], interval_s: float = 2.0) -> None:
    Recorder(base_url, data_dir, symbols, interval_s).run()
