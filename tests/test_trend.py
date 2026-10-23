import numpy as np
import pandas as pd

from engine.trend.backtest import COINS, signals, simulate, stats


def _bars(prices):
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="1D", tz="UTC")
    return pd.DataFrame({"open": prices, "close": prices}, index=idx)


def test_trend_buys_only_after_the_signal_bar_and_pays_costs():
    up = np.r_[np.full(30, 100.0), np.linspace(100, 200, 30)]
    bars = {s: _bars(up) for s in COINS}
    sig = signals(bars, 10)
    first = sig["BTCUSDT"].idxmax()
    r = simulate(bars, 10, {s: 0.01 for s in COINS}, "2024-01-01", "2024-03-01")
    eq = r["equity"]
    assert eq.loc[:first].eq(1.0).all()          # nothing happens until the bar after the signal
    assert eq.iloc[-1] > 1.3 and r["trades"] == len(COINS)
    assert stats(eq)["max_dd_pct"] < 5
