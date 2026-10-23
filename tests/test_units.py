import numpy as np
import pandas as pd
import pytest

from engine.broker.brokers import _floor, walk_book
from engine.config import Cfg, load_config
from engine.features import indicators as ind
from engine.model.labels import triple_barrier
from engine.model.predict import expected_value
from engine.risk.manager import RiskManager, exit_check
from engine.risk.portfolio import Portfolio, Position

T0 = pd.Timestamp("2026-01-01", tz="UTC")


@pytest.fixture
def cfg():
    return load_config("config.yaml")


def bars_from(prices_hlc):
    idx = pd.date_range(T0, periods=len(prices_hlc), freq="1h")
    df = pd.DataFrame(prices_hlc, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1.0
    return df


def test_rsi_bounds():
    s = pd.Series(np.cumsum(np.random.default_rng(0).normal(size=500)) + 100)
    r = ind.rsi(s)
    assert r.between(0, 100).all()


def test_atr_positive():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(size=300))
    df = pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c})
    assert (ind.atr(df).dropna() > 0).all()


def test_triple_barrier_tp_sl_and_tie():
    lab = Cfg(tp_atr=1.0, sl_atr=1.0, min_tp_pct=0.02, min_sl_pct=0.01, max_sl_pct=0.05, horizon_bars=3)
    #       open  high  low   close
    rows = [
        (100, 100, 100, 100),  # decision bar 0 -> entry at bar1 open = 100
        (100, 103, 99.5, 102),  # TP (2%) hit, SL (1%) not -> win
        (102, 102, 102, 102),
        (102, 102, 102, 102),
        (102, 102, 102, 102),
    ]
    b = bars_from(rows)
    atr = pd.Series(0.0, index=b.index)
    out = triple_barrier(b, atr, lab)
    assert out["label"].iloc[0] == 1
    assert out["outcome"].iloc[0] == pytest.approx(0.02)

    rows[1] = (100, 103, 98, 100)  # both touched in one bar -> conservative loss
    out = triple_barrier(bars_from(rows), atr, lab)
    assert out["label"].iloc[0] == 0
    assert out["outcome"].iloc[0] == pytest.approx(-0.01)

    rows = [(100, 100, 100, 100)] + [(100, 101, 99.5, 101)] * 4  # nothing hit -> timeout
    out = triple_barrier(bars_from(rows), atr, lab)
    assert out["label"].iloc[0] == 0
    assert out["outcome"].iloc[0] == pytest.approx(0.01)  # close of bar 3 = 101
    assert np.isnan(out["label"].iloc[-1])  # not enough future


def test_expected_value():
    ev = expected_value(np.array([0.5]), np.array([0.02]), np.array([0.01]), 0.006)
    assert ev[0] == pytest.approx(0.5 * 0.02 - 0.5 * 0.01 - 0.006)


def test_portfolio_pnl_with_fees():
    pf = Portfolio(cash=1_000_000, initial=1_000_000)
    pos = Position("BTCTMN", 1.0, 100_000, T0, 110_000, 95_000, 0.02, T0, "d1", entry_fee=250)
    pf.open(pos)
    assert pf.cash == pytest.approx(1_000_000 - 100_250)
    t = pf.close("BTCTMN", 110_000, 275, T0 + pd.Timedelta(hours=2), "take_profit")
    assert t.pnl_tmn == pytest.approx(10_000 - 525)
    assert pf.equity({}) == pytest.approx(1_000_000 + 10_000 - 525)


def test_risk_sizing_caps(cfg):
    risk = RiskManager(cfg)
    pf = Portfolio(cash=5_000_000, initial=5_000_000)
    d = risk.size(T0, "BTCTMN", sl_pct=0.01, size_scale=1.0, pf=pf, prices={}, min_notional=50_000)
    assert d.allowed
    assert d.notional == pytest.approx(5_000_000 * min(cfg.risk.max_position_frac, cfg.risk.risk_per_trade / 0.01))
    d = risk.size(T0, "BTCTMN", sl_pct=0.01, size_scale=0.001, pf=pf, prices={}, min_notional=50_000)
    assert not d.allowed  # below min notional


def test_risk_max_positions_and_no_averaging(cfg):
    risk = RiskManager(cfg)
    pf = Portfolio(cash=5_000_000, initial=5_000_000)
    for i, s in enumerate(["A", "B", "C"][: cfg.risk.max_positions]):
        pf.open(Position(s, 1, 100_000, T0, 1, 1, 0.01, T0, f"d{i}", 0))
    assert not risk.size(T0, "D", 0.01, 1.0, pf, {}, 50_000).allowed
    assert not risk.size(T0, "A", 0.01, 1.0, pf, {}, 50_000).allowed


def test_daily_loss_and_kill_switch(cfg):
    risk = RiskManager(cfg)
    pf = Portfolio(cash=5_000_000, initial=5_000_000)
    pf.mark(T0, {})
    pf.cash = 5_000_000 * (1 - cfg.risk.daily_loss_limit - 0.001)
    assert risk.entries_blocked(T0, pf, pf.equity({})) is not None
    pf.cash = 5_000_000 * (1 - cfg.risk.max_drawdown_kill - 0.001)
    assert "kill_switch" in risk.update(pf, pf.equity({}))
    assert risk.must_flatten(T0)


def test_flatten_before_end(cfg):
    risk = RiskManager(cfg)
    end = T0 + pd.Timedelta(days=3)
    risk.set_window(T0, end)
    no_entry = pd.Timedelta(minutes=cfg.challenge.no_new_entries_before_end_min)
    flat = pd.Timedelta(minutes=cfg.challenge.flatten_before_end_min)
    assert risk.entries_blocked(end - no_entry - pd.Timedelta(minutes=1), Portfolio(1, 1), 1) is None
    assert risk.entries_blocked(end - no_entry + pd.Timedelta(minutes=1), Portfolio(1, 1), 1) is not None
    assert not risk.must_flatten(end - flat - pd.Timedelta(minutes=1))
    assert risk.must_flatten(end - flat + pd.Timedelta(minutes=1))


def test_exit_check_sl_tp_trailing():
    def pos():
        return Position("X", 1, 100.0, T0, 103.0, 98.0, 0.01, T0 + pd.Timedelta(hours=12), "d", 0)

    assert exit_check(pos(), 101, 97.5, T0, 1.0)[0] == "stop_loss"
    assert exit_check(pos(), 103.5, 99, T0, 1.0)[0] == "take_profit"
    p = pos()
    assert exit_check(p, 101.5, 100.5, T0, 1.0) is None  # peak 101.5 > entry + 1 ATR -> trailing on
    assert p.trailing_active and p.sl_price >= 100.1
    assert exit_check(p, 100.2, 100.0, T0, 1.0)[0] == "trailing_stop"
    assert exit_check(pos(), 100.5, 99.5, T0 + pd.Timedelta(hours=13), 1.0)[0] == "time_stop"


def test_walk_book_and_floor():
    asks = [{"price": 100, "quantity": 1}, {"price": 101, "quantity": 5}]
    avg, q = walk_book(asks, notional=150)
    assert q == pytest.approx(1 + 50 / 101)
    assert 100 < avg < 101
    assert _floor(1.23456789, 3) == 1.234


def test_blocklist(cfg):
    from engine.config import universe
    from engine.wallex.allowed import BlockedMarketError, assert_allowed, is_allowed

    assert not is_allowed("ZECTMN") and not is_allowed("PAXGUSDT") and not is_allowed("1BBABYDOGETMN")
    assert is_allowed("BTCTMN") and is_allowed("XAUTTMN")
    with pytest.raises(BlockedMarketError):
        assert_allowed("TONTMN")
    cfg["universe"]["core"] = ["BTCTMN", "ZECTMN", "TONTMN"]
    assert universe(cfg) == ["BTCTMN"]


def test_dotenv(tmp_path, monkeypatch):
    import os

    from engine.env import load_dotenv

    f = tmp_path / ".env"
    f.write_text('# c\nA_KEY=abc\nB_KEY="q v"\nEMPTY=\nKEEP=fromfile\n')
    monkeypatch.setenv("KEEP", "real")
    for k in ("A_KEY", "B_KEY", "EMPTY"):
        monkeypatch.delenv(k, raising=False)
    assert sorted(load_dotenv(f)) == ["A_KEY", "B_KEY"]
    assert os.environ["A_KEY"] == "abc" and os.environ["B_KEY"] == "q v"
    assert "EMPTY" not in os.environ and os.environ["KEEP"] == "real"


def test_agent_budget_disables_and_trading_continues(tmp_path, cfg):
    import json as _json

    from engine.agent.llm import LLM
    from engine.agent.overlay import AgentOverlay

    llm = LLM("http://unused.invalid", str(tmp_path), api_key="k", budget_usd=1.0)
    (tmp_path / "_spend.json").write_text(_json.dumps({"usd": 1.2}))
    cfg["agent"]["enabled"] = True
    agent = AgentOverlay(cfg, llm)
    r = agent.review(T0, {"symbol": "BTCTMN"}, {}, {})
    assert r["action"] == "approve" and r["source"] == "disabled"   # ML + risk keep trading
    assert not agent.enabled and "budget" in agent.disabled_reason


def test_agent_repeated_errors_switch_off(tmp_path, cfg):
    from engine.agent.llm import LLM
    from engine.agent.overlay import MAX_CONSECUTIVE_ERRORS, AgentOverlay

    llm = LLM("http://127.0.0.1:9", str(tmp_path), api_key="k", timeout_s=0.5)  # nothing listens -> network error
    cfg["agent"]["enabled"] = True
    agent = AgentOverlay(cfg, llm)
    acts = [agent.review(T0, {"symbol": f"S{i}"}, {}, {})["action"] for i in range(MAX_CONSECUTIVE_ERRORS)]
    assert acts[:-1] == ["veto"] * (MAX_CONSECUTIVE_ERRORS - 1)  # single failures stay safe
    assert acts[-1] == "approve" and not agent.enabled


def test_binance_guard_modes():
    from engine.risk.manager import binance_guard

    g_exit = {"mode": "exit", "drop_3m": 0.005, "tighten_to": 0.003}
    g_tight = {"mode": "tighten", "drop_3m": 0.005, "tighten_to": 0.003}
    pos = Position("X", 1, 100.0, T0, 103.0, 98.0, 0.01, T0, "d", 0)
    assert binance_guard(pos, -0.004, 100.0, g_exit) is None       # small drop: nothing
    assert binance_guard(pos, None, 100.0, g_exit) is None          # no Binance data: nothing
    assert binance_guard(pos, -0.006, 100.0, g_exit) == "exit"
    assert binance_guard(pos, -0.006, 100.0, g_tight) is None and pos.sl_price == pytest.approx(99.7)
    assert pos.guard_fired
    binance_guard(pos, -0.02, 99.0, g_tight)                       # fires only once per position
    assert pos.sl_price == pytest.approx(99.7)


def test_binance_ret_3m_causal():
    import numpy as np

    from engine.data.binance import ret_3m

    df = pd.DataFrame({"ts": [0, 60, 120, 180, 240], "close": [100, 100, 100, 99, 98]})
    ts, r3 = ret_3m(df)
    assert list(ts) == [0, 60, 120, 180, 240]
    assert np.isnan(r3[2]) and r3[3] == pytest.approx(np.log(0.99)) and r3[4] == pytest.approx(np.log(0.98))
