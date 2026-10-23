"""Triple-barrier labels: does a long entered at the next bar's open hit take-profit before
stop-loss within `horizon` bars? Uses the same TP/SL rules as the risk manager."""
from __future__ import annotations

import numpy as np
import pandas as pd


def barrier_pcts(atr_pct: np.ndarray, lab) -> tuple[np.ndarray, np.ndarray]:
    tp = np.maximum(lab.tp_atr * atr_pct, lab.min_tp_pct)
    sl = np.clip(lab.sl_atr * atr_pct, lab.min_sl_pct, lab.max_sl_pct)
    return tp, sl


def triple_barrier(bars: pd.DataFrame, atr_pct: pd.Series, lab) -> pd.DataFrame:
    """Returns per-row: label (1 = TP first), outcome (gross return of the trade), tp_pct, sl_pct.
    If TP and SL are both touched in the same bar the trade counts as a loss (conservative)."""
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    H = int(lab.horizon_bars)
    tp, sl = barrier_pcts(atr_pct.to_numpy(float), lab)
    label = np.full(n, np.nan)
    outcome = np.full(n, np.nan)
    hold = np.full(n, np.nan)
    for i in range(n - 1):
        if np.isnan(tp[i]):
            continue
        entry = o[i + 1]
        up, dn = entry * (1 + tp[i]), entry * (1 - sl[i])
        last = min(n - 1, i + H)
        if i + H > n - 1:  # not enough future data -> unlabeled
            continue
        res, r, k = 0, None, last
        for j in range(i + 1, last + 1):
            hit_dn = lo[j] <= dn
            hit_up = h[j] >= up
            if hit_dn:
                r, k = -sl[i], j
                break
            if hit_up:
                res, r, k = 1, tp[i], j
                break
        if r is None:
            r = c[last] / entry - 1
        label[i], outcome[i], hold[i] = res, r, k - i
    return pd.DataFrame({"label": label, "outcome": outcome, "hold_bars": hold, "tp_pct": tp, "sl_pct": sl}, index=bars.index)
