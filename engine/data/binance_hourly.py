"""Outside data from Binance for the hourly model: spot 1h klines (price, volume, taker-buy flow)
and futures funding rates. Wallex follows global prices (STRATEGY.md §7.1f), so the global
price, order flow and positioning carry information the Wallex candles alone do not.

  data/binance_1h/<BASE>USDT.parquet   ts (bar open, epoch s), close, volume, quote_volume, trades, taker_buy
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
FUTURES_SYMBOL = {"PEPE": "1000PEPEUSDT", "SHIB": "1000SHIBUSDT", "FLOKI": "1000FLOKIUSDT"}  # perpetuals quoted per 1000 tokens


def _base(wallex_symbol: str) -> str:
    return wallex_symbol[:-4] if wallex_symbol.endswith("USDT") else wallex_symbol.removesuffix("TMN")


def spot_symbol(wallex_symbol: str) -> str | None:
    base = _base(wallex_symbol)
    return None if base in ("USDT", "XAUT", "") else f"{base}USDT"


def futures_symbol(wallex_symbol: str) -> str | None:
    base = _base(wallex_symbol)
    return None if base in ("USDT", "XAUT") else FUTURES_SYMBOL.get(base, f"{base}USDT")


def path(data_dir: str, bsym: str, interval: str = "1h") -> Path:
    return Path(data_dir) / f"binance_{interval}" / f"{bsym}.parquet"


STEP_MS = {"30m": 1_800_000, "1h": 3_600_000}


def _rows(k: list) -> list[tuple]:
    return [(x[0] // 1000, float(x[4]), float(x[5]), float(x[7]), int(x[8]), float(x[9])) for x in k]


COLS = ["ts", "close", "volume", "quote_volume", "trades", "taker_buy"]


def fetch(client: httpx.Client, bsym: str, start_ms: int | None = None, limit: int = 1000,
          interval: str = "1h") -> pd.DataFrame:
    params = {"symbol": bsym, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    r = client.get(SPOT, params=params)
    r.raise_for_status()
    return pd.DataFrame(_rows(r.json()), columns=COLS)


def download(data_dir: str, wallex_symbols: list[str], days: int = 400, interval: str = "1h") -> None:
    end = int(time.time() * 1000)
    with httpx.Client(timeout=20) as c:
        for ws in wallex_symbols:
            b = spot_symbol(ws)
            if not b:
                continue
            t, parts = end - days * 86_400_000, []
            while t < end:
                df = fetch(c, b, t, interval=interval)
                if df.empty:
                    break
                parts.append(df)
                t = int(df["ts"].iloc[-1]) * 1000 + STEP_MS[interval]
                time.sleep(0.05)
            out = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
            p = path(data_dir, b, interval)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(p, index=False)
            log.info("binance %s %s: %d bars", interval, b, len(out))


def load(data_dir: str, wallex_symbol: str, interval: str = "1h") -> pd.DataFrame | None:
    b = spot_symbol(wallex_symbol)
    if not b or not path(data_dir, b, interval).exists():
        return None
    df = pd.read_parquet(path(data_dir, b, interval))
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def recent(client: httpx.Client, wallex_symbol: str, hours: int = 500, interval: str = "1h") -> pd.DataFrame | None:
    """Live: the last `hours` closed and open bars (the open bar is dropped by the feature builder's index)."""
    b = spot_symbol(wallex_symbol)
    if not b:
        return None
    parts, start = [], int(time.time() * 1000) - hours * 3_600_000
    while True:
        df = fetch(client, b, start, interval=interval)
        if df.empty:
            break
        parts.append(df)
        if len(df) < 1000:
            break
        start = int(df["ts"].iloc[-1]) * 1000 + STEP_MS[interval]
    if not parts:
        return None
    out = pd.concat(parts).drop_duplicates("ts")
    out.index = pd.to_datetime(out["ts"], unit="s", utc=True)
    return out
