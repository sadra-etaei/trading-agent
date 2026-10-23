"""Event-driven simulator.

Decisions are made at each bar close using that bar's features; entries fill at the next bar's
first 1-minute open; stop-loss / take-profit / trailing / time stops are checked on every
1-minute candle. Fees and slippage on both sides. Same Strategy, RiskManager and Portfolio
objects as live trading.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from engine.config import slippage
from engine.risk.manager import RiskManager, binance_guard, exit_check
from engine.risk.portfolio import Portfolio, Position
from engine.strategy.core import Intent, Strategy


class NullLog:
    def new_id(self) -> str:
        return "x"

    def write(self, kind, ts, **fields) -> str:
        return fields.get("decision_id", "x")


class MinuteData:
    """Fast slicing of 1-minute candles per symbol."""

    def __init__(self, m1: dict[str, pd.DataFrame]):
        self.d = {}
        for s, df in m1.items():
            df = df.sort_values("ts")
            self.d[s] = {c: df[c].to_numpy(float) for c in ("open", "high", "low", "close")}
            self.d[s]["ts"] = df["ts"].to_numpy("int64")

    def slice(self, sym: str, t0: int, t1: int):
        a = self.d[sym]
        i, j = np.searchsorted(a["ts"], [t0, t1])
        return {k: v[i:j] for k, v in a.items()}


def floor_dec(x: float, dec: int) -> float:
    f = 10 ** dec
    return math.floor(x * f) / f


def simulate(cfg, strategy: Strategy, feats: dict[str, pd.DataFrame], minutes: MinuteData,
             start: pd.Timestamp, end: pd.Timestamp, risk: RiskManager, dlog, bar: pd.Timedelta,
             markets: dict[str, dict], selector=None, guard: dict | None = None) -> tuple[Portfolio, pd.DataFrame]:
    """guard: {symbol: (ts array, Binance 3-min log return array)} enables the Binance crash guard."""
    gcfg = cfg.risk.get("binance_guard") or {}
    use_guard = bool(guard) and bool(gcfg.get("enabled"))
    fee = cfg.costs.fee_rate
    horizon = bar * int(cfg.labels.horizon_bars)
    pf = Portfolio(cash=float(cfg.risk.initial_equity_tmn), initial=float(cfg.risk.initial_equity_tmn))
    cash_cfg = cfg.get("cash") or {}
    usdt_rate = feats["USDTTMN"]["close"] if "USDTTMN" in feats else None
    if cash_cfg.get("idle") == "usdt" and usdt_rate is not None:  # user rule: idle capital in USDT
        pf.usdt_mode, pf.conv_fee = True, float(cash_cfg.get("conv_fee", 0.0031))

    def rate_at(ts):
        if usdt_rate is None:
            return None
        v = usdt_rate[:ts]
        return float(v.iloc[-1]) if len(v) else None

    def back_to_tmn():  # challenge end / kill switch: everything to Toman, stay in Toman
        if pf.usdt_mode:
            pf.to_tmn_all()
            pf.usdt_mode = False
    syms = [s for s in strategy.symbols if s in feats]
    # Wide table of closes for marking.
    closes = pd.DataFrame({s: feats[s]["close"] for s in syms}).ffill()
    rows_by_ts = {s: feats[s] for s in syms}
    bar_times = closes.index[(closes.index >= start) & (closes.index + bar <= end)]
    pending: list[Intent] = []
    curve = []
    flattened = False

    def close_pos(sym, price, ts, reason):
        p = pf.positions[sym]
        px = price * (1 - slippage(cfg, sym))
        t = pf.close(sym, px, p.qty * px * fee, ts, reason)
        dlog.write("exit", ts, decision_id=p.decision_id + "-x", entry_decision_id=p.decision_id, symbol=sym,
                   reason=reason, exit_price=px, pnl_tmn=t.pnl_tmn, ret_pct=t.ret_pct, hold_hours=t.hold_hours)

    if pf.usdt_mode and len(bar_times):
        pf.set_rate(rate_at(bar_times[0]))
        pf.to_usdt_all()
    for T in bar_times:
        t0, t1 = int(T.timestamp()), int((T + bar).timestamp())
        pf.set_rate(rate_at(T))
        # 1) fill entries decided at the previous close
        for it in pending:
            m = minutes.slice(it.symbol, t0, t1)
            if len(m["ts"]) == 0:
                dlog.write("fill_skipped", T, entry_decision_id=it.decision_id, symbol=it.symbol, reason="no data")
                continue
            k0 = min(int(cfg.execution.get("backtest_entry_delay_min", 0) or 0), len(m["ts"]) - 1)  # live needs ~1-3 min to act
            px = m["open"][k0] * (1 + slippage(cfg, it.symbol))
            qty = floor_dec(it.notional / px, markets.get(it.symbol, {}).get("qty_dec", 8))
            if qty <= 0:
                continue
            ets = pd.Timestamp(int(m["ts"][k0]), unit="s", tz="UTC")
            pos = Position(it.symbol, qty, px, ets, px * (1 + it.tp_pct), px * (1 - it.sl_pct), it.atr_pct,
                           ets + horizon, it.decision_id, qty * px * fee)
            pf.open(pos)
            dlog.write("fill", ets, decision_id=it.decision_id + "-f", entry_decision_id=it.decision_id,
                       symbol=it.symbol, side="BUY", price=px, qty=qty, notional_tmn=qty * px)
        pending = []

        # 2) minute-by-minute exits
        for sym in list(pf.positions):
            m = minutes.slice(sym, t0, t1)
            pos = pf.positions[sym]
            g = guard.get(sym) if use_guard else None
            for k in range(len(m["ts"])):
                ts = pd.Timestamp(int(m["ts"][k]), unit="s", tz="UTC")
                if risk.must_flatten(ts):
                    if not risk.day_halted(ts):  # a daily halt stays in USDT; only the end converts back
                        back_to_tmn()
                    close_pos(sym, m["close"][k], ts, risk.flatten_reason(ts))
                    break
                if g is not None:
                    # Binance minute that closed when this Wallex minute opened (causal)
                    j = np.searchsorted(g[0], int(m["ts"][k]) - 60)
                    r3 = g[1][j] if j < len(g[0]) and g[0][j] == int(m["ts"][k]) - 60 else None
                    if binance_guard(pos, r3, m["open"][k], gcfg) == "exit":
                        close_pos(sym, m["open"][k], ts, "binance_guard")
                        break
                hit = exit_check(pos, m["high"][k], m["low"][k], ts, cfg.risk.trailing_atr)
                if hit:
                    reason, price = hit
                    if price is None:
                        price = m["close"][k]
                    elif reason != "take_profit":
                        price = min(price, m["open"][k])  # gap through the stop fills at the open
                    close_pos(sym, price, ts, reason)
                    break

        # 3) bar close: mark, risk state, decide
        Tc = T + bar
        pf.set_rate(rate_at(Tc - pd.Timedelta(seconds=1)))
        prices = closes.loc[T].dropna().to_dict()
        eq = pf.mark(Tc, prices)
        for ev in risk.update(pf, eq):
            dlog.write("risk_event", Tc, event=ev, reason=risk.halt_reason, equity=eq)
        if risk.must_flatten(Tc):
            if not risk.day_halted(Tc):
                back_to_tmn()
            for sym in list(pf.positions):
                close_pos(sym, prices[sym], Tc, risk.flatten_reason(Tc))
            flattened = True
        if not flattened:
            rows = pd.DataFrame({s: rows_by_ts[s].loc[T] for s in syms if T in rows_by_ts[s].index}).T
            if len(rows):
                market = {
                    "btc_ret_24h_pct": _r(rows, "BTCTMN" if "BTCTMN" in rows.index else "BTCUSDT", "ret_24"),
                    "usdt_tmn_ret_24h_pct": _r(rows, "USDTTMN", "ret_24"),
                    "fear_greed": float(rows["fng"].iloc[0] * 100) if "fng" in rows and pd.notna(rows["fng"].iloc[0]) else None,
                }
                rows = rows.apply(pd.to_numeric, errors="coerce")
                if selector is not None:
                    strategy.allowed = selector(T)
                pending = strategy.decide_entries(Tc, rows, pf, prices, market)
        eq = pf.equity(prices)
        curve.append({"ts": Tc, "equity": eq, "cash": pf.cash, "exposure": pf.exposure(prices),
                      "positions": len(pf.positions), "drawdown": 1 - eq / pf.peak_equity})

    # close anything left at the final close
    if pf.positions and len(bar_times):
        Tl = bar_times[-1]
        prices = closes.loc[Tl].to_dict()
        back_to_tmn()
        for sym in list(pf.positions):
            close_pos(sym, prices[sym], Tl + bar, "end_of_test")
        eq = pf.equity({})
        curve.append({"ts": Tl + bar, "equity": eq, "cash": pf.cash, "exposure": 0.0, "positions": 0,
                      "drawdown": 1 - eq / max(pf.peak_equity, eq)})
    if pf.usdt_mode:  # flat at the end of the backtest: report the value back in Toman
        back_to_tmn()
        if curve:
            eq = pf.equity({})
            curve.append({"ts": curve[-1]["ts"], "equity": eq, "cash": pf.cash, "exposure": 0.0, "positions": 0,
                          "drawdown": 1 - eq / max(pf.peak_equity, eq)})
    return pf, pd.DataFrame(curve)


def _r(rows: pd.DataFrame, sym: str, col: str):
    if sym in rows.index and col in rows:
        v = pd.to_numeric(rows.loc[sym, col], errors="coerce")
        return None if pd.isna(v) else round(float(v) * 100, 2)
    return None
