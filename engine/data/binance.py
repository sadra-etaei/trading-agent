"""Binance spot 1-minute prices: a leading indicator for Wallex (Wallex follows Binance by ~1-2 min,
STRATEGY.md §7.1f). Used only by the crash guard: never traded on.

  data/binance_1m/<BASE>USDT.parquet   columns ts (epoch s), close
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from engine.net import binance_spot

log = logging.getLogger(__name__)
URL = binance_spot() + "/api/v3/klines"


def binance_symbol(wallex_symbol: str) -> str | None:
    base = wallex_symbol[:-4] if wallex_symbol.endswith("USDT") else wallex_symbol.removesuffix("TMN")
    return None if base in ("USDT", "XAUT", "") else f"{base}USDT"


def path(data_dir: str, bsym: str) -> Path:
    return Path(data_dir) / "binance_1m" / f"{bsym}.parquet"


def load(data_dir: str, bsym: str) -> pd.DataFrame:
    p = path(data_dir, bsym)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame(columns=["ts", "close"])


def klines(client: httpx.Client, bsym: str, start_ms: int, limit: int = 1000) -> list:
    r = client.get(URL, params={"symbol": bsym, "interval": "1m", "startTime": start_ms, "limit": limit})
    r.raise_for_status()
    return r.json()


def download(data_dir: str, wallex_symbols: list[str], days: int) -> None:
    end = int(time.time() * 1000)
    with httpx.Client(timeout=20) as c:
        for ws in wallex_symbols:
            b = binance_symbol(ws)
            if not b:
                continue
            old = load(data_dir, b)
            t = int(old["ts"].max() * 1000 + 60_000) if len(old) else end - days * 86_400_000
            rows = []
            while t < end:
                try:
                    k = klines(c, b, t)
                except httpx.HTTPError as e:
                    log.warning("binance %s: %s", b, e)
                    time.sleep(2)
                    continue
                if not k:
                    break
                rows += [(x[0] // 1000, float(x[4])) for x in k]
                t = k[-1][0] + 60_000
                time.sleep(0.05)
            df = pd.concat([old, pd.DataFrame(rows, columns=["ts", "close"])]).drop_duplicates("ts").sort_values("ts")
            p = path(data_dir, b)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(p, index=False)
            log.info("binance %s: %d minutes", b, len(df))


def ret_3m(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(ts array, 3-minute log return known at the END of each minute)."""
    s = pd.Series(np.log(df["close"].to_numpy(float)), index=df["ts"].to_numpy("int64"))
    s = s[~s.index.duplicated()].sort_index()
    full = s.reindex(np.arange(s.index.min(), s.index.max() + 60, 60)).ffill()
    return full.index.to_numpy("int64"), full.diff(3).to_numpy()


def live_ret_3m(client: httpx.Client, bsym: str) -> float | None:
    """Latest 3-minute return from the last closed 1-minute candles (live guard)."""
    try:
        r = client.get(URL, params={"symbol": bsym, "interval": "1m", "limit": 5}, timeout=5)
        r.raise_for_status()
        k = r.json()
    except (httpx.HTTPError, ValueError):
        return None
    closes = [float(x[4]) for x in k[:-1]]  # drop the still-open candle
    return float(np.log(closes[-1] / closes[-4])) if len(closes) >= 4 else None
