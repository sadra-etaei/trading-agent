"""Loads the recorder output (data/live/) into aligned fair-value frames.

For each TMN pair X with a Wallex XUSDT pair:
  F       = mid(XUSDT) * mid(USDTTMN)            fair value of X in TMN
  offset  = rolling median of log(mid_TMN / F)   structural premium (past-only window)
  gap_bid = log(bid_TMN / F) - offset            < 0: TMN cheap to buy at the bid (maker entry)
  gap_ask = log(ask_TMN / F) - offset            exit side
Books are recorded live (~2s), so a flat mid just means a quiet book. Inputs whose mid has not moved
for `stale_s` (default 15 min) are flagged as a dead market and not traded against.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def epoch_s(idx) -> "np.ndarray":
    """Seconds since epoch for a DatetimeIndex, independent of its storage unit (s/ms/us/ns)."""
    return pd.DatetimeIndex(idx).as_unit("ns").asi8 / 1e9


def live_days(data_dir: str) -> list[Path]:
    root = Path(data_dir) / "live"
    return sorted(p for p in root.glob("*") if (p / "books.jsonl.gz").exists()) if root.exists() else []


def _lines(path: Path):
    """Yields parsed lines; tolerates a truncated last gzip member / line (recorder still writing)."""
    try:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except (EOFError, OSError):
        return


def load_tob(data_dir: str) -> pd.DataFrame:
    """Best bid/ask (+ quantity at best) from recorded order books.
    Wide frame: index = UTC timestamp, columns MultiIndex (symbol, bid|ask|bid_qty|ask_qty)."""
    recs = []
    for day in live_days(data_dir):
        for r in _lines(day / "books.jsonl.gz"):
            row = {"ts": r["ts"]}
            for s, (bids, asks) in r["b"].items():
                if bids and asks:
                    row[(s, "bid")], row[(s, "bid_qty")] = bids[0]
                    row[(s, "ask")], row[(s, "ask_qty")] = asks[0]
            recs.append(row)
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs).set_index("ts").sort_index()
    df.index = pd.to_datetime(df.index, unit="s", utc=True)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df.apply(pd.to_numeric, errors="coerce")


def load_trades(data_dir: str, symbol: str) -> pd.DataFrame:
    """Public trades (de-duplicated by the recorder). Columns: price, qty, buy; index = trade time (UTC)."""
    rows = []
    for day in live_days(data_dir):
        p = day / "trades" / f"{symbol}.jsonl"
        if not p.exists():
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(r)
    if not rows:
        return pd.DataFrame(columns=["price", "qty", "buy"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["trade_ts", "price", "qty", "buy"])
    df.index = pd.to_datetime(df["trade_ts"], utc=True)
    return df[["price", "qty", "buy"]].sort_index()


def _stale(s: pd.Series, stale_s: float) -> pd.Series:
    """True where the value has not changed for more than stale_s seconds."""
    changed = s.ne(s.shift())
    t = epoch_s(s.index)
    last_change = pd.Series(np.where(changed, t, np.nan), index=s.index).ffill()
    return pd.Series(t - last_change.to_numpy() > stale_s, index=s.index)


def fair_value_frame(tob: pd.DataFrame, base: str, offset_window: str = "2h", stale_s: float = 900) -> pd.DataFrame:
    tmn, usd = f"{base}TMN", f"{base}USDT"
    need = [tmn, usd, "USDTTMN"]
    if tob.empty or any(s not in tob.columns.get_level_values(0) for s in need):
        return pd.DataFrame()
    d = pd.DataFrame(index=tob.index)
    d["bid"], d["ask"] = tob[(tmn, "bid")], tob[(tmn, "ask")]
    d["bid_qty"], d["ask_qty"] = tob[(tmn, "bid_qty")], tob[(tmn, "ask_qty")]
    d["mid"] = (d["bid"] + d["ask"]) / 2
    d["usd_mid"] = (tob[(usd, "bid")] + tob[(usd, "ask")]) / 2
    d["usdt_mid"] = (tob[("USDTTMN", "bid")] + tob[("USDTTMN", "ask")]) / 2
    d["F"] = d["usd_mid"] * d["usdt_mid"]
    d = d[(d["bid"] > 0) & (d["ask"] > d["bid"]) & (d["F"] > 0)].copy()
    raw = np.log(d["mid"] / d["F"])
    d["offset"] = raw.rolling(offset_window, min_periods=30).median().shift(1)  # past-only
    d["gap_mid"] = raw - d["offset"]
    d["gap_bid"] = np.log(d["bid"] / d["F"]) - d["offset"]
    d["gap_ask"] = np.log(d["ask"] / d["F"]) - d["offset"]
    d["spread"] = (d["ask"] - d["bid"]) / d["mid"]
    d["usd_spread"] = (tob[(usd, "ask")] - tob[(usd, "bid")]).reindex(d.index) / d["usd_mid"]
    d["stale_usd"] = _stale(d["usd_mid"], stale_s)
    d["stale_tmn"] = _stale(d["mid"], stale_s)
    d["stale_usdt"] = _stale(d["usdt_mid"], stale_s)
    return d.dropna(subset=["offset"])


def bases_available(tob: pd.DataFrame) -> list[str]:
    syms = set(tob.columns.get_level_values(0)) if not tob.empty else set()
    return sorted(s[:-3] for s in syms if s.endswith("TMN") and s != "USDTTMN" and f"{s[:-3]}USDT" in syms)
