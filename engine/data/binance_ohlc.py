"""Binance spot hourly OHLCV candles as the MODEL'S data source (PLAN.md §15b candidate C).

Wallex USDT books trade in only 5-25% of minutes, so hourly bars built from Wallex trades are sparse and
stale. Wallex follows Binance within 1-2 minutes, so a model can learn from Binance's dense candles,
pooled over many liquid coins, and trade the few pairs Wallex lists (orders still fill on Wallex).
Wallex XUSDT and Binance XUSDT use the same symbol names.

  data/binance_ohlc_1h/<SYM>.parquet   ts (bar open, epoch s), open, high, low, close, volume,
                                        quote_volume, trades, taker_buy
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from engine.net import binance_spot

log = logging.getLogger(__name__)
SPOT = binance_spot() + "/api/v3/klines"
COLS = ["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy"]


def path(data_dir: str, sym: str) -> Path:
    return Path(data_dir) / "binance_ohlc_1h" / f"{sym}.parquet"


def _rows(k: list) -> list[tuple]:
    return [(x[0] // 1000, float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[7]), int(x[8]),
             float(x[9])) for x in k]


def fetch(client: httpx.Client, sym: str, start_ms: int | None = None, limit: int = 1000) -> pd.DataFrame:
    params = {"symbol": sym, "interval": "1h", "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    r = client.get(SPOT, params=params)
    r.raise_for_status()
    return pd.DataFrame(_rows(r.json()), columns=COLS)


def download(data_dir: str, symbols: list[str], days: int = 730) -> None:
    end = int(time.time() * 1000)
    with httpx.Client(timeout=20) as c:
        for s in symbols:
            t, parts = end - days * 86_400_000, []
            while t < end:
                try:
                    df = fetch(c, s, t)
                except httpx.HTTPError as e:
                    log.warning("binance %s: %s", s, e)
                    time.sleep(2)
                    continue
                if df.empty:
                    break
                parts.append(df)
                t = int(df["ts"].iloc[-1]) * 1000 + 3_600_000
                time.sleep(0.05)
            if not parts:
                log.warning("binance %s: no data", s)
                continue
            out = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
            p = path(data_dir, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(p, index=False)
            log.info("binance 1h %s: %d bars from %s", s, len(out), pd.to_datetime(out["ts"].iloc[0], unit="s"))


def to_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Same shape as store.resample output, plus the Binance flow columns used by the outside features."""
    b = df.copy()
    b.index = pd.to_datetime(b["ts"], unit="s", utc=True)
    b["active_min"] = (b["volume"] > 0).astype(int) * 60
    return b.drop(columns=["ts"])


def load_bars(data_dir: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    return {s: to_bars(pd.read_parquet(path(data_dir, s))) for s in symbols if path(data_dir, s).exists()}


def recent(client: httpx.Client, sym: str, hours: int = 500) -> pd.DataFrame:
    """Live: the last `hours` bars (the still-open bar is dropped by the caller's bar index)."""
    parts, start = [], int(time.time() * 1000) - hours * 3_600_000
    while True:
        df = fetch(client, sym, start)
        if df.empty:
            break
        parts.append(df)
        if len(df) < 1000:
            break
        start = int(df["ts"].iloc[-1]) * 1000 + 3_600_000
    return to_bars(pd.concat(parts).drop_duplicates("ts")) if parts else pd.DataFrame()
