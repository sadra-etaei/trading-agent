"""Local candle store: one Parquet file per pair of 1-minute candles, plus resampling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

COLS = ["ts", "open", "high", "low", "close", "volume"]


def candle_path(data_dir: str | Path, symbol: str) -> Path:
    return Path(data_dir) / "candles_1m" / f"{symbol}.parquet"


def load_1m(data_dir: str | Path, symbol: str) -> pd.DataFrame:
    p = candle_path(data_dir, symbol)
    if not p.exists():
        return pd.DataFrame(columns=COLS)
    return pd.read_parquet(p)


def save_1m(data_dir: str | Path, symbol: str, df: pd.DataFrame) -> None:
    p = candle_path(data_dir, symbol)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = df[COLS].drop_duplicates("ts", keep="last").sort_values("ts").reset_index(drop=True)
    df.to_parquet(p, index=False)


def udf_to_frame(raw: dict) -> pd.DataFrame:
    if raw.get("s") != "ok" or not raw.get("t"):
        return pd.DataFrame(columns=COLS)
    df = pd.DataFrame(
        {
            "ts": np.asarray(raw["t"], dtype="int64"),
            "open": pd.to_numeric(raw["o"]),
            "high": pd.to_numeric(raw["h"]),
            "low": pd.to_numeric(raw["l"]),
            "close": pd.to_numeric(raw["c"]),
            "volume": pd.to_numeric(raw["v"]),
        }
    )
    return df


RULES = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}


def resample(df_1m: pd.DataFrame, bar: str) -> pd.DataFrame:
    """Resample 1-minute candles to `bar`. Index = bar open time (UTC). Adds `active_min`
    (minutes with volume > 0) so quiet / stale bars can be flagged."""
    if df_1m.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "active_min"])
    d = df_1m.copy()
    d.index = pd.to_datetime(d["ts"], unit="s", utc=True)
    d["active"] = (d["volume"] > 0).astype(int)
    out = d.resample(RULES[bar], label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "active": "sum"}
    )
    out = out.rename(columns={"active": "active_min"})
    out["close"] = out["close"].ffill()
    for c in ("open", "high", "low"):
        out[c] = out[c].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    out["active_min"] = out["active_min"].fillna(0).astype(int)
    return out.dropna(subset=["close"])


def load_bars(data_dir: str | Path, symbols: list[str], bar: str) -> dict[str, pd.DataFrame]:
    return {s: resample(load_1m(data_dir, s), bar) for s in symbols}


def write_manifest(data_dir: str | Path, symbols: list[str]) -> dict:
    """Records pair ranges + content hashes so a backtest can prove which data it used."""
    info = {}
    for s in symbols:
        p = candle_path(data_dir, s)
        if not p.exists():
            continue
        df = pd.read_parquet(p, columns=["ts"])
        info[s] = {
            "rows": int(len(df)),
            "first": pd.to_datetime(df["ts"].min(), unit="s", utc=True).isoformat(),
            "last": pd.to_datetime(df["ts"].max(), unit="s", utc=True).isoformat(),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16],
        }
    path = Path(data_dir) / "manifest.json"
    path.write_text(json.dumps(info, indent=2))
    return info
