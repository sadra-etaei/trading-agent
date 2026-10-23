"""Step 2: event replay backtester for the fair-value gap strategy (PLAN.md §14.3–14.5).

Replays recorded order books (~2s) and public trades through the shared PairEngine
(engine/fv/engine.py: entry/exit rules and the conservative maker fill rule).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.fv.data import bases_available, epoch_s, fair_value_frame, load_tob, load_trades
from engine.fv.engine import FVParams, FVTrade, PairEngine, Quote
from engine.wallex import markets as mk


def simulate_pair(pair: str, d: pd.DataFrame, trades: pd.DataFrame, p: FVParams, price_dec: int = 0) -> tuple[list[FVTrade], dict]:
    eng = PairEngine(pair, p, price_dec)
    ts = epoch_s(d.index)
    if len(ts) == 0:
        return [], eng.stats
    tt = epoch_s(trades.index) if len(trades) else np.array([])
    tr = list(zip(tt, trades["price"].to_numpy(float), trades["qty"].to_numpy(float), trades["buy"].to_numpy(bool))) if len(trades) else []
    cols = {c: d[c].to_numpy(float) for c in ("bid", "ask", "bid_qty", "ask_qty", "F", "offset", "gap_bid")}
    fresh = ~(d["stale_usd"].to_numpy(bool) | d["stale_usdt"].to_numpy(bool)) & (d["usd_spread"].to_numpy(float) <= p.max_usd_spread)
    j = 0
    q = None
    for i, t in enumerate(ts):
        k = j
        while k < len(tr) and tr[k][0] <= t:
            k += 1
        window, j = tr[j:k], k
        q = Quote(t, cols["bid"][i], cols["ask"][i], cols["bid_qty"][i], cols["ask_qty"][i], cols["F"][i],
                  cols["offset"][i], cols["gap_bid"][i], bool(fresh[i]))
        eng.step(q, window)
    eng.finish(q)
    return eng.trades, eng.stats


def run_all(cfg, p: FVParams, tob=None, frames=None, trades=None) -> tuple[pd.DataFrame, dict]:
    tob = load_tob(cfg.data.dir) if tob is None else tob
    markets = mk.load(cfg)
    frames = frames or {b: fair_value_frame(tob, b) for b in bases_available(tob)}
    trades = trades or {b: load_trades(cfg.data.dir, f"{b}TMN") for b in frames}
    allt, stats = [], {}
    for b, d in frames.items():
        if d.empty:
            continue
        dec = mk.rules(markets, f"{b}TMN")["price_dec"]
        tr, st = simulate_pair(f"{b}TMN", d, trades[b], p, dec)
        allt += tr
        stats[b] = st
    return pd.DataFrame([asdict(t) for t in allt]), stats


def summarize(df: pd.DataFrame, hours: float, equity: float) -> dict:
    if df.empty:
        return {"trades": 0, "trades_per_day": 0.0, "pnl_tmn": 0.0, "return_pct": 0.0}
    return {
        "trades": int(len(df)),
        "trades_per_day": round(len(df) / max(hours, 1e-9) * 24, 1),
        "win_rate_pct": round(100 * float((df["pnl_tmn"] > 0).mean()), 1),
        "mean_net_pct": round(float(df["ret_pct"].mean()), 3),
        "pnl_tmn": round(float(df["pnl_tmn"].sum())),
        "return_pct": round(100 * float(df["pnl_tmn"].sum()) / equity, 3),
        "maker_exit_pct": round(100 * float((df["exit_kind"] == "maker_target").mean()), 1),
        "median_wait_fill_s": round(float(df["wait_fill_s"].median()), 1),
        "median_hold_s": round(float(df["hold_s"].median()), 1),
        "fair_move_median_pct": round(float(df["fair_move_pct"].median()), 3),
    }


GRID_THETA = [0.0025, 0.0035, 0.005, 0.007]
FEE_SCENARIOS = {"maker 0.10% / taker 0.20%": (0.001, 0.002), "maker 0.25% / taker 0.25%": (0.0025, 0.0025)}


def run_fv_backtest(cfg) -> Path:
    tob = load_tob(cfg.data.dir)
    if tob.empty:
        raise SystemExit("no recordings in data/live — run `python cli.py record` first")
    frames = {b: fair_value_frame(tob, b) for b in bases_available(tob)}
    trades = {b: load_trades(cfg.data.dir, f"{b}TMN") for b in frames}
    t0, t1 = tob.index.min(), tob.index.max()
    hours = (t1 - t0).total_seconds() / 3600
    equity = float(cfg.risk.initial_equity_tmn)
    mid = t0 + (t1 - t0) / 2
    rows, best = [], None
    run_dir = Path(cfg.run.output_dir) / ("fv-bt-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    for fee_name, (mf, tf) in FEE_SCENARIOS.items():
        for th in GRID_THETA:
            p = FVParams(theta_in=th, maker_fee=mf, taker_fee=tf, min_profit=2 * mf + 0.0005)
            df, st = run_all(cfg, p, tob, frames, trades)
            s = summarize(df, hours, equity)
            halves = []
            for lo, hi in ((t0, mid), (mid, t1)):
                h = df[(pd.to_datetime(df["entry_ts"]) >= lo) & (pd.to_datetime(df["entry_ts"]) < hi)] if len(df) else df
                halves.append(summarize(h, hours / 2, equity))
            orders = sum(v["orders"] for v in st.values())
            fills = sum(v["fills"] for v in st.values())
            rows.append({"fees": fee_name, "theta_in": th, **s, "fill_rate_pct": round(100 * fills / orders, 1) if orders else 0.0,
                         "first_half_mean_net_pct": halves[0].get("mean_net_pct"), "second_half_mean_net_pct": halves[1].get("mean_net_pct"),
                         "first_half_trades": halves[0]["trades"], "second_half_trades": halves[1]["trades"]})
            tag = f"{'m10' if mf == 0.001 else 'm25'}_th{int(th * 1e4)}"
            if len(df):
                df.to_csv(run_dir / f"trades_{tag}.csv", index=False)
    res = {"recorded_from": t0.isoformat(), "recorded_to": t1.isoformat(), "hours": round(hours, 2),
           "pairs": list(frames), "params_default": asdict(FVParams()), "grid": rows}
    (run_dir / "result.json").write_text(json.dumps(res, indent=2, default=str))
    md = _md(res)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return run_dir


def _md(res: dict) -> str:
    L = ["# Fair-value gap — event replay backtest", "",
         f"Recordings {res['recorded_from'][:19]} .. {res['recorded_to'][:19]} UTC ({res['hours']} h), pairs: {', '.join(res['pairs'])}.",
         "Conservative maker fills (trade-through or queue-ahead volume), 2s latency, 600k TMN per trade.", "",
         "| fees | θ_in | trades | /day | fill rate | win % | mean net/trade | P&L (TMN) | return on 5M | maker exits | 1st half mean | 2nd half mean | F move (median) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["grid"]:
        if not r["trades"]:
            L.append(f"| {r['fees']} | {r['theta_in']:.2%} | 0 | 0 | {r['fill_rate_pct']}% | — | — | 0 | 0% | — | — | — | — |")
            continue
        L.append(f"| {r['fees']} | {r['theta_in']:.2%} | {r['trades']} | {r['trades_per_day']} | {r['fill_rate_pct']}% | {r['win_rate_pct']}% | "
                 f"{r['mean_net_pct']:+.3f}% | {r['pnl_tmn']:+,} | {r['return_pct']:+.3f}% | {r['maker_exit_pct']}% | "
                 f"{r['first_half_mean_net_pct'] if r['first_half_mean_net_pct'] is not None else '—'} ({r['first_half_trades']}) | "
                 f"{r['second_half_mean_net_pct'] if r['second_half_mean_net_pct'] is not None else '—'} ({r['second_half_trades']}) | "
                 f"{r['fair_move_median_pct']:+.3f}% |")
    L += ["", "Go/no-go (PLAN.md §14.5): positive mean net per trade in BOTH halves with ≥ 10 trades/day."]
    return "\n".join(L) + "\n"
