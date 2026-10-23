"""Wallex hourly candles from `udf/history` (resolution 60), back to 2021 for the USDT pairs (PLAN.md §15c).
The 1-minute store only reaches back one year; hourly history is what a longer training set needs.

  data/wallex_1h/<SYM>.parquet   ts (bar open, epoch s), open, high, low, close, volume
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger(__name__)
URL = "https://api.wallex.ir/v1/udf/history"
CHUNK_S = 30 * 86400


def path(data_dir: str, sym: str) -> Path:
    return Path(data_dir) / "wallex_1h" / f"{sym}.parquet"


def download(data_dir: str, symbols: list[str], since: str = "2021-01-01") -> None:
    end = int(time.time())
    with httpx.Client(timeout=30) as c:
        for s in symbols:
            t, parts = int(pd.Timestamp(since, tz="UTC").timestamp()), []
            while t < end:
                for attempt in range(4):
                    try:
                        r = c.get(URL, params={"symbol": s, "resolution": "60", "from": t, "to": min(end, t + CHUNK_S)}).json()
                        break
                    except (httpx.HTTPError, ValueError) as e:
                        log.warning("wallex 1h %s: %s", s, e)
                        time.sleep(3 * (attempt + 1))
                else:
                    r = {}
                if r.get("s") == "ok" and r.get("t"):
                    parts.append(pd.DataFrame({"ts": r["t"], "open": pd.to_numeric(r["o"]), "high": pd.to_numeric(r["h"]),
                                               "low": pd.to_numeric(r["l"]), "close": pd.to_numeric(r["c"]),
                                               "volume": pd.to_numeric(r["v"])}))
                t += CHUNK_S
                time.sleep(0.1)
            if not parts:
                log.warning("wallex 1h %s: no data", s)
                continue
            out = pd.concat(parts).drop_duplicates("ts").sort_values("ts")
            p = path(data_dir, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            out.to_parquet(p, index=False)
            log.info("wallex 1h %s: %d bars from %s", s, len(out), pd.to_datetime(out["ts"].iloc[0], unit="s"))


def load_bars(data_dir: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        if path(data_dir, s).exists():
            b = pd.read_parquet(path(data_dir, s))
            b.index = pd.to_datetime(b["ts"], unit="s", utc=True)
            b = b.drop(columns=["ts"])
            b["active_min"] = (b["volume"] > 0).astype(int) * 60
            out[s] = b
    return out
