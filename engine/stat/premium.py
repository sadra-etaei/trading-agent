"""Wallex-vs-Binance premium mean reversion on hourly bars (PLAN.md §17).

p = log(Wallex close / Binance close); d = p - median of p over the previous 168 hours; z = d / std of d over
the previous 168 hours (only hours in which Wallex traded). Buy at the next hour's Wallex open when z <= -k;
sell at the next open after d >= 0 or after H hours. One position per pair, 1/len(PAIRS) of equity each.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import slippage
from engine.data import binance_ohlc, wallex_hourly
from engine.trend.backtest import stats

PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "PEPEUSDT"]
WIN = 168


def frames(data_dir: str) -> dict[str, pd.DataFrame]:
    wx = wallex_hourly.load_bars(data_dir, PAIRS)
    bn = binance_ohlc.load_bars(data_dir, PAIRS)
    out = {}
    for s in PAIRS:
        if s not in wx or s not in bn:
            continue
        f = pd.DataFrame({"wx_open": wx[s]["open"], "wx_close": wx[s]["close"], "active": wx[s]["volume"] > 0,
                          "bn_close": bn[s]["close"]}).dropna(subset=["bn_close"])
        f = f[f.index.isin(wx[s].index)]
        p = np.log(f["wx_close"] / f["bn_close"]).where(f["active"])
        med = p.rolling(WIN, min_periods=48).median().shift(1)
        d = p - med
        f["d"] = d
        f["z"] = d / d.rolling(WIN, min_periods=48).std().shift(1)
        out[s] = f
    return out


def trades_for(f: pd.DataFrame, k: float, hold: int, cost: float, start, end) -> list[dict]:
    z, d, op = f["z"].to_numpy(), f["d"].to_numpy(), f["wx_open"].to_numpy()
    idx = f.index
    out, i, n = [], 0, len(f)
    lo, hi = idx.searchsorted(start), idx.searchsorted(end)
    i = lo
    while i < min(hi, n - 1):
        if np.isfinite(z[i]) and z[i] <= -k:
            e = i + 1
            j = e
            while j < min(n - 1, e + hold) and not (np.isfinite(d[j]) and d[j] >= 0):
                j += 1
            x = min(n - 1, j + 1 if j < e + hold else e + hold)
            if not (np.isfinite(op[e]) and np.isfinite(op[x])):
                i += 1
                continue
            ret = op[x] * (1 - cost) / (op[e] * (1 + cost)) - 1
            out.append({"entry": idx[e], "exit": idx[x], "ret": ret, "z": z[i]})
            i = x
        else:
            i += 1
    return out


def portfolio(trades: pd.DataFrame, start, end) -> pd.Series:
    """Each pair holds 1/len(PAIRS) of equity per trade; equity compounds trade by trade (exit order)."""
    eq, curve = 1.0, [(pd.Timestamp(start, tz="UTC"), 1.0)]
    for _, t in trades.sort_values("exit").iterrows():
        eq *= 1 + t["ret"] / len(PAIRS)
        curve.append((t["exit"], eq))
    curve.append((pd.Timestamp(end, tz="UTC"), eq))
    s = pd.Series(dict(curve)).sort_index()
    return s[~s.index.duplicated(keep="last")].resample("1h").last().ffill()


def evaluate(fr, k, hold, cfg, start, end) -> tuple[dict, pd.DataFrame]:
    a, b = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    tr = pd.DataFrame([t | {"symbol": s} for s, f in fr.items()
                       for t in trades_for(f, k, hold, cfg.costs.usdt_taker_fee + slippage(cfg, s), a, b)])
    if tr.empty:
        return {"trades": 0}, tr
    g, l_ = tr.loc[tr.ret > 0, "ret"].sum(), -tr.loc[tr.ret < 0, "ret"].sum()
    eq = portfolio(tr, start, end)
    st = stats(eq)
    return {"trades": int(len(tr)), "win_rate": round(float((tr.ret > 0).mean()), 3),
            "mean_trade_pct": round(100 * float(tr.ret.mean()), 3), "profit_factor": round(float(g / l_), 2) if l_ else None,
            **{k_: st[k_] for k_ in ("total_pct", "mean_month_pct", "positive_months_pct", "max_dd_pct")}}, tr


def run(cfg, dev=("2021-01-01", "2024-12-31"), held=("2025-01-01", "2026-10-22")) -> dict:
    fr = frames(cfg.data.dir)
    rows = []
    for k, hold in itertools.product((1.5, 2.0, 2.5, 3.0), (1, 3, 6)):
        m, _ = evaluate(fr, k, hold, cfg, *dev)
        rows.append({"k": k, "hold_h": hold, **m})
    grid = pd.DataFrame(rows)
    print("DEVELOPMENT", dev)
    print(grid.to_string(index=False))
    ok = grid[(grid["trades"] >= 100) & (grid["profit_factor"].fillna(0) >= 1.2)]
    out = {"dev": dev, "held": held, "grid": rows, "pick": None}
    if ok.empty:
        print("no variant with >= 100 trades and profit factor >= 1.2 on development -> nothing to confirm")
    else:
        p = ok.sort_values("total_pct").iloc[-1]
        m, tr = evaluate(fr, float(p["k"]), int(p["hold_h"]), cfg, *held)
        passes = bool(m.get("trades", 0) and (m.get("profit_factor") or 0) >= 1.2 and m.get("total_pct", 0) > 0)
        out["pick"] = {"k": float(p["k"]), "hold_h": int(p["hold_h"]), "held_out": m, "passes": passes,
                       "meets_4pct": bool(m.get("mean_month_pct", 0) >= 4)}
        print(f"PICK k={p['k']} H={int(p['hold_h'])} -> HELD-OUT {held}: {m}")
        print("passes:", passes, "| meets 4%/month:", out["pick"]["meets_4pct"])
        if len(tr):
            print(tr.groupby("symbol")["ret"].agg(["count", "mean", "sum"]).round(4))
    Path(cfg.run.output_dir, "premium_backtest.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
