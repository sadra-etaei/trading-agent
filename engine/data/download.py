"""Paged 1-minute history downloader for Wallex (resumable)."""
from __future__ import annotations

import logging
import time

import pandas as pd

from engine.data import store
from engine.wallex.client import RES_1M, WallexClient

log = logging.getLogger(__name__)

CHUNK_S = 3 * 86400  # Wallex returns at most ~3 days of 1-minute candles per request


def download_symbol(client: WallexClient, data_dir: str, symbol: str, days: int, now: int | None = None) -> int:
    now = int(now or time.time())
    start_wanted = now - days * 86400
    existing = store.load_1m(data_dir, symbol)
    frames = [existing] if not existing.empty else []

    # Fill forward from the last stored candle; if history is missing at the start, fill that too.
    ranges = []
    if existing.empty:
        ranges.append((start_wanted, now))
    else:
        first, last = int(existing["ts"].min()), int(existing["ts"].max())
        if first > start_wanted + 3600:
            ranges.append((start_wanted, first))
        ranges.append((last, now))

    added = 0
    empty_streak = 0
    for lo, hi in ranges:
        t = lo
        while t < hi:
            t2 = min(t + CHUNK_S, hi)
            raw = client.candles(symbol, RES_1M, t, t2)
            df = store.udf_to_frame(raw)
            if df.empty:
                empty_streak += 1
                if empty_streak >= 20:  # probably before the pair was listed
                    log.info("%s: no data around %s, skipping ahead", symbol, pd.to_datetime(t, unit="s"))
                    empty_streak = 0
            else:
                empty_streak = 0
                df = df[(df["ts"] >= t) & (df["ts"] < t2)]
                frames.append(df)
                added += len(df)
                if len(frames) >= 30:  # checkpoint so an interrupted download resumes
                    store.save_1m(data_dir, symbol, pd.concat(frames, ignore_index=True))
                    frames = [store.load_1m(data_dir, symbol)]
            t = t2
        if frames:
            store.save_1m(data_dir, symbol, pd.concat(frames, ignore_index=True))
            frames = [store.load_1m(data_dir, symbol)]
    log.info("%s: +%d candles", symbol, added)
    return added


def download_all(client: WallexClient, data_dir: str, symbols: list[str], days: int) -> dict:
    for s in symbols:
        download_symbol(client, data_dir, s, days)
    return store.write_manifest(data_dir, symbols)
