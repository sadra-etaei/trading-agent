"""Technical indicators (pure pandas, past-only windows)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return wilder(true_range(df), n)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = wilder(d.clip(lower=0), n)
    down = wilder(-d.clip(upper=0), n)
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50.0)


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - ema(line, signal)


def stoch_k(df: pd.DataFrame, n: int = 14) -> pd.Series:
    lo = df["low"].rolling(n).min()
    hi = df["high"].rolling(n).max()
    return ((df["close"] - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = wilder(true_range(df), n)
    plus_di = 100 * wilder(plus_dm, n) / tr.replace(0, np.nan)
    minus_di = 100 * wilder(minus_dm, n) / tr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx.fillna(0), n)


def zscore(s: pd.Series, n: int) -> pd.Series:
    m = s.rolling(n, min_periods=n // 2).mean()
    sd = s.rolling(n, min_periods=n // 2).std()
    return (s - m) / sd.replace(0, np.nan)
