"""Market-trend filter: no new long entries when the overall market is falling.

Long-only spot bots lose most of their money buying into market-wide declines (shadow run
2026-10-22: 5 of 5 longs stopped out while BTC fell ~2.5%). Each rule reads market context
columns built in features/build.py and returns True where new entries are allowed.
The rule is chosen with `cli.py train` (out-of-sample comparison in the training report).
"""
from __future__ import annotations

import pandas as pd

RULES = {
    "none": None,
    "btc_above_ema50": ("btc_dist_ema50", 0.0),      # BTC above its 50-hour EMA
    "btc_above_ema200": ("btc_dist_ema200", 0.0),    # BTC above its 200-hour EMA
    "btc_ema50_rising": ("btc_slope_ema50", 0.0),    # BTC 50-hour EMA rising over the last 6 hours
    "btc_ret24_pos": ("btc_ret_24", 0.0),            # BTC up over the last 24 hours
    "breadth_50": ("mkt_breadth_ema50", 0.5),        # at least half of the pairs above their EMA50
    "mkt_ret24_pos": ("mkt_ret_24", 0.0),            # median pair up over the last 24 hours
}


def allowed(df: pd.DataFrame, rule: str) -> pd.Series:
    """Boolean mask per row: True = new entries allowed. Missing context (warm-up) allows entries."""
    spec = RULES[rule]
    if spec is None:
        return pd.Series(True, index=df.index)
    col, thr = spec
    if col not in df:
        return pd.Series(True, index=df.index)
    v = pd.to_numeric(df[col], errors="coerce")
    return (v > thr) | v.isna() if thr == 0.0 else (v >= thr) | v.isna()


def describe(rule: str) -> str:
    spec = RULES[rule]
    return "no filter" if spec is None else f"{spec[0]} {'>' if spec[1] == 0 else '>='} {spec[1]}"
