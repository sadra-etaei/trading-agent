import numpy as np
import pandas as pd
import pytest

from engine.config import load_config
from engine.lag.backtest import add_signals, simulate
from engine.lag.logger import Basis, History
from engine.lag.replay import replay
from engine.lag.rules import entry_signal, exit_reason


@pytest.fixture
def lag():
    return dict(load_config("config.yaml").lag)


def test_entry_needs_gap_and_binance_move(lag):
    assert entry_signal(lag, "BTCUSDT", 99.5, 100.0, 0.005)
    assert not entry_signal(lag, "BTCUSDT", 99.8, 100.0, 0.005)   # gap too small
    assert not entry_signal(lag, "PEPEUSDT", 99.5, 100.0, 0.005)  # PEPE needs a 0.6% Binance move


def test_exit_reversal_beats_take_profit(lag):
    assert exit_reason(lag, 100, 1000, 998, 101, 1) == "bn_reverse"
    assert exit_reason(lag, 100, 1000, 1000, 100.5, 1) == "take_profit"
    assert exit_reason(lag, 100, 1000, 1000, 100.1, lag["time_stop_min"]) == "time_stop"
    assert exit_reason(lag, 100, 1000, 1000, 100.1, 1) is None


def test_basis_and_history():
    b, h = Basis(60), History()
    for t in range(10):
        b.add(t, 0.001 * t)
        h.add(t * 30, 100 + t)
    assert b.value() == pytest.approx(0.0045)
    assert h.ret(270) == pytest.approx(109 / 103 - 1)


def _minutes(n, wx, bn, active):
    idx = pd.date_range("2026-08-01", periods=n, freq="1min", tz="UTC")
    f = pd.DataFrame({"open": wx, "high": wx, "low": wx, "close": wx, "volume": active.astype(float),
                      "active": active, "bn": bn}, index=idx)
    return f


def test_candle_fill_needs_a_trade_at_the_limit(lag):
    n = 80
    wx = np.full(n, 100.0)
    bn = np.full(n, 100.0)
    bn[62:] = 101.0                       # Binance jumps 1%, Wallex has not moved yet
    active = np.ones(n, bool)
    f = add_signals(_minutes(n, wx, bn, active), lag, "BTCUSDT", 0.0001)
    assert f["signal"].iloc[62]
    wx2 = wx.copy()
    wx2[63:] = 101.0                      # Wallex catches up the next minute: no trade at our limit
    f2 = add_signals(_minutes(n, wx2, bn, active), lag, "BTCUSDT", 0.0001)
    assert simulate(f2, lag, "BTCUSDT", 0.0001, 0.0013) == []
    tr = simulate(f2, lag, "BTCUSDT", 0.0001, 0.0013, optimistic=True)
    assert tr and tr[0]["reason"] == "take_profit"


def test_book_replay_round_trip(lag):
    rows = []
    for k in range(10):
        ask = 100.0 if k == 0 else 100.6
        rows.append({"ts": 1000 + 5 * k, "symbol": "BTCUSDT", "bn_bid": 101, "bn_ask": 101, "wx_bid": ask - 0.05,
                     "wx_ask": ask, "wx_ask_qty": 10, "signal": int(k == 0), "gap_ask": -0.004})
    tr = replay(pd.DataFrame(rows), lag, 0.0013, 100)
    assert len(tr) == 1 and tr.iloc[0]["reason"] == "take_profit"
    assert tr.iloc[0]["net"] == pytest.approx(100.55 * (1 - 0.0013) / (100 * 1.0013) - 1)
