"""Crypto Fear & Greed Index (alternative.me). Daily values with history since 2018."""
from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pandas as pd

log = logging.getLogger(__name__)
URL = "https://api.alternative.me/fng/"


def fetch(limit: int = 0) -> pd.DataFrame:
    """limit=0 returns full history."""
    r = httpx.get(URL, params={"limit": limit, "format": "json"}, timeout=20)
    r.raise_for_status()
    rows = r.json()["data"]
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime([int(x["timestamp"]) for x in rows], unit="s", utc=True),
            "fng": [float(x["value"]) for x in rows],
        }
    )
    return df.sort_values("ts").reset_index(drop=True)


def path(data_dir: str) -> Path:
    return Path(data_dir) / "sentiment" / "fear_greed.parquet"


def download(data_dir: str) -> pd.DataFrame:
    df = fetch(0)
    p = path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    log.info("fear & greed: %d days (%s .. %s)", len(df), df.ts.min().date(), df.ts.max().date())
    return df


def load(data_dir: str) -> pd.DataFrame | None:
    p = path(data_dir)
    return pd.read_parquet(p) if p.exists() else None
