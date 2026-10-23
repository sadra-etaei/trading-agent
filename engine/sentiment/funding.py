"""Binance USDT-M futures funding rates (positioning sentiment). Optional feature: Binance may
not be reachable from where the bot runs, so it is off by default in config."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pandas as pd

from engine.net import binance_futures

log = logging.getLogger(__name__)
URL = binance_futures() + "/fapi/v1/fundingRate"


def binance_symbol(wallex_symbol: str) -> str | None:
    from engine.data.binance_hourly import futures_symbol

    return futures_symbol(wallex_symbol)


def fetch(symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows, t = [], start_ms
    with httpx.Client(timeout=20) as c:
        while t < end_ms:
            r = c.get(URL, params={"symbol": symbol, "startTime": t, "endTime": end_ms, "limit": 1000})
            r.raise_for_status()
            batch = r.json()
            if not batch:
                break
            rows += batch
            t = batch[-1]["fundingTime"] + 1
            time.sleep(0.2)
    if not rows:
        return pd.DataFrame(columns=["ts", "funding"])
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([x["fundingTime"] for x in rows], unit="ms", utc=True),
            "funding": [float(x["fundingRate"]) for x in rows],
        }
    )


def path(data_dir: str, wallex_symbol: str) -> Path:
    return Path(data_dir) / "sentiment" / f"funding_{wallex_symbol}.parquet"


def download(data_dir: str, symbols: list[str], days: int) -> None:
    end = int(time.time() * 1000)
    start = end - days * 86400 * 1000
    for s in symbols:
        b = binance_symbol(s)
        if not b:
            continue
        try:
            df = fetch(b, start, end)
        except httpx.HTTPError as e:
            log.warning("funding %s failed: %s", b, e)
            continue
        p = path(data_dir, s)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
        log.info("funding %s: %d rows", b, len(df))


def load(data_dir: str, wallex_symbol: str) -> pd.DataFrame | None:
    p = path(data_dir, wallex_symbol)
    if not p.exists() and wallex_symbol.endswith("USDT"):  # same coin's funding, stored under its Toman pair
        p = path(data_dir, wallex_symbol[:-4] + "TMN")
    return pd.read_parquet(p) if p.exists() else None
