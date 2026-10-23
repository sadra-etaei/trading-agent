"""Trend following on daily / 4-hour bars (PLAN.md §16): long-only spot, idle cash in USDT.

Per coin, at each bar close: in trend if close > SMA(N) and the N-bar return > 0. Trades happen at the next
bar's open (no look-ahead). Each coin in trend holds 1/len(coins) of equity, bought when its signal turns on
and sold in full when it turns off (no rebalancing in between). Costs per side: taker fee + the pair's
measured Wallex half-spread. Prices are Binance's (Wallex USDT pairs follow Binance within minutes).
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine.config import slippage
from engine.data import binance_ohlc

COINS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "PEPEUSDT"]
BARS = {"1d": ("1D", 1), "4h": ("4h", 6)}


def load(data_dir: str, bar: str) -> dict[str, pd.DataFrame]:
    rule = BARS[bar][0]
    out = {}
    for s, b in binance_ohlc.load_bars(data_dir, COINS).items():
        out[s] = b.resample(rule, label="left", closed="left").agg({"open": "first", "close": "last"}).dropna()
    return out


def signals(bars: dict[str, pd.DataFrame], n_bars: int) -> pd.DataFrame:
    sig = {}
    for s, b in bars.items():
        c = b["close"]
        sig[s] = (c > c.rolling(n_bars).mean()) & (c / c.shift(n_bars) - 1 > 0)
    return pd.DataFrame(sig).fillna(False)


def simulate(bars: dict[str, pd.DataFrame], n_bars: int, cost: dict[str, float], start: str, end: str) -> dict:
    sig = signals(bars, n_bars)
    opens = pd.DataFrame({s: b["open"] for s, b in bars.items()})
    idx = opens.index[(opens.index >= pd.Timestamp(start, tz="UTC")) & (opens.index <= pd.Timestamp(end, tz="UTC"))]
    cash, qty = 1.0, {s: 0.0 for s in bars}
    slot = 1.0 / len(COINS)
    curve, trades = [], 0
    prev = sig.shift(1).reindex(idx).fillna(False)  # signal at the previous close -> act at this open
    for t in idx:
        px = opens.loc[t]
        eq = cash + sum(q * px[s] for s, q in qty.items() if q and np.isfinite(px[s]))
        for s in bars:
            p = px[s]
            if not np.isfinite(p):
                continue
            want = bool(prev.at[t, s])
            if want and qty[s] == 0:
                spend = min(slot * eq, cash)
                if spend > 0:
                    qty[s] = spend * (1 - cost[s]) / p
                    cash -= spend
                    trades += 1
            elif not want and qty[s] > 0:
                cash += qty[s] * p * (1 - cost[s])
                qty[s] = 0.0
                trades += 1
        curve.append((t, cash + sum(q * px[s] for s, q in qty.items() if q and np.isfinite(px[s]))))
    eq = pd.Series(dict(curve))
    return {"equity": eq, "trades": trades}


def stats(eq: pd.Series, trades: int | None = None) -> dict:
    m = eq.resample("ME").last().pct_change().dropna()
    first = eq.resample("ME").last().iloc[0] / eq.iloc[0] - 1  # partial first month
    m = pd.concat([pd.Series([first], index=[m.index[0] - pd.offsets.MonthEnd(1)]), m]) if len(m) else m
    dd = 1 - eq / eq.cummax()
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    out = {"total_pct": round(100 * (eq.iloc[-1] / eq.iloc[0] - 1), 1),
           "cagr_pct": round(100 * ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1), 1) if years > 0 else None,
           "mean_month_pct": round(100 * m.mean(), 2), "median_month_pct": round(100 * m.median(), 2),
           "positive_months_pct": round(100 * (m > 0).mean(), 0), "worst_month_pct": round(100 * m.min(), 1),
           "best_month_pct": round(100 * m.max(), 1), "max_dd_pct": round(100 * dd.max(), 1),
           "sharpe_m": round(float(m.mean() / m.std() * np.sqrt(12)), 2) if m.std() > 0 else None, "months": int(len(m))}
    if trades is not None:
        out["trades"] = trades
        out["trades_per_month"] = round(trades / max(1, len(m)), 1)
    return out


def buy_hold(bars: dict[str, pd.DataFrame], coins: list[str], start: str, end: str) -> pd.Series:
    opens = pd.DataFrame({s: bars[s]["open"] for s in coins})
    o = opens[(opens.index >= pd.Timestamp(start, tz="UTC")) & (opens.index <= pd.Timestamp(end, tz="UTC"))]
    rel = o / o.bfill().iloc[0]  # coins listed later join at their first price (cash until then)
    return rel.fillna(1.0).mean(axis=1)


def run(cfg, dev=("2018-01-01", "2024-12-31"), held=("2025-01-01", "2026-10-22")) -> dict:
    cost = {s: cfg.costs.usdt_taker_fee + slippage(cfg, s) for s in COINS}
    rows, data = [], {b: load(cfg.data.dir, b) for b in BARS}
    for bar, n_days in itertools.product(BARS, (20, 50, 100)):
        r = simulate(data[bar], n_days * BARS[bar][1], cost, *dev)
        rows.append({"bar": bar, "n_days": n_days, **stats(r["equity"], r["trades"])})
    grid = pd.DataFrame(rows)
    ok = grid[grid["max_dd_pct"] <= 35]
    pick = ok.sort_values("mean_month_pct").iloc[-1] if len(ok) else None
    out = {"dev_window": dev, "held_window": held, "grid": rows, "costs": cost}
    print("DEVELOPMENT", dev)
    print(grid.to_string(index=False))
    for name, coins in (("equal-weight buy&hold", COINS), ("BTC buy&hold", ["BTCUSDT"])):
        for w in (dev, held):
            s = stats(buy_hold(data["1d"], coins, *w))
            out.setdefault("benchmarks", {})[f"{name} {w[0][:4]}-{w[1][:4]}"] = s
            print(f"{name:22s} {w[0]}..{w[1]}: mean month {s['mean_month_pct']}% | max DD {s['max_dd_pct']}% | total {s['total_pct']}%")
    if pick is None:
        print("no variant with max drawdown <= 35% on the development window -> nothing to confirm")
        out["pick"] = None
    else:
        r = simulate(data[pick["bar"]], int(pick["n_days"]) * BARS[pick["bar"]][1], cost, *held)
        h = stats(r["equity"], r["trades"])
        out["pick"] = {"bar": pick["bar"], "n_days": int(pick["n_days"]), "held_out": h,
                       "meets_4pct_target": h["mean_month_pct"] >= 4.0}
        print(f"PICK {pick['bar']} N={int(pick['n_days'])}d -> HELD-OUT {held}: {h}")
        print("meets the 4%/month target:", out["pick"]["meets_4pct_target"])
        r["equity"].to_csv(Path(cfg.run.output_dir) / "trend_heldout_equity.csv")
    Path(cfg.run.output_dir, "trend_backtest.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
