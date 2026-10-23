"""Book replay of lag-buy on the gap logger's real order books (PLAN.md §15: the overnight shadow).

The 1-minute candle test cannot tell a lagging quote from a book that simply did not trade, so this
replay uses only logged books: buy at the logged Wallex ask when the shared entry rule fires (and the
ask level holds at least `size_usdt`), sell at a later logged Wallex bid (taker) when an exit rule
fires. Both sides pay the taker fee.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from engine.lag.rules import exit_reason


def load_gaps(data_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(data_dir) / "lag" / "*" / "gaps.csv")))
    if not files:
        return pd.DataFrame()
    return pd.concat(pd.read_csv(p) for p in files).sort_values("ts").reset_index(drop=True)


def replay(g: pd.DataFrame, cfg_lag, fee: float, size_usdt: float) -> pd.DataFrame:
    trades = []
    for s, d in g.groupby("symbol"):
        d = d.reset_index(drop=True)
        i = 0
        while i < len(d):
            r = d.iloc[i]
            if not (r["signal"] == 1 and r["wx_ask"] * r["wx_ask_qty"] >= size_usdt):
                i += 1
                continue
            entry, bn_entry = r["wx_ask"], (r["bn_bid"] + r["bn_ask"]) / 2
            j, reason = i + 1, None
            while j < len(d):
                x = d.iloc[j]
                held_min = (x["ts"] - r["ts"]) / 60
                # take-profit needs a real bid at the target: we sell into it as takers
                reason = exit_reason(cfg_lag, entry, bn_entry, (x["bn_bid"] + x["bn_ask"]) / 2, x["wx_bid"], held_min)
                if reason:
                    break
                j += 1
            if not reason:
                break
            x = d.iloc[j]
            net = x["wx_bid"] * (1 - fee) / (entry * (1 + fee)) - 1
            trades.append({"symbol": s, "entry_ts": r["ts"], "exit_ts": x["ts"], "entry": entry, "exit": x["wx_bid"],
                           "reason": reason, "seconds": round(x["ts"] - r["ts"]), "net": net, "gap": r["gap_ask"]})
            i = j + 1
    return pd.DataFrame(trades)


def run(cfg) -> dict:
    g = load_gaps(cfg.data.dir)
    if g.empty:
        print("no gap logger data yet (run `python cli.py lag-log`)")
        return {}
    hours = (g["ts"].max() - g["ts"].min()) / 3600
    size = cfg.risk.initial_equity_tmn * cfg.lag.size_frac / 185_000  # USDT per trade at ~185k TMN/USDT
    tr = replay(g, cfg.lag, cfg.costs.usdt_taker_fee, size)
    stats = g.groupby("symbol").agg(spread_med=("spread", "median"), gap_ask_p05=("gap_ask", lambda x: x.quantile(0.05)),
                                    signals=("signal", "sum"))
    out = {"hours_logged": round(hours, 2), "snapshots": int(g["ts"].nunique()), "trades": int(len(tr)),
           "net_sum_pct": round(100 * float(tr["net"].sum()), 3) if len(tr) else 0.0,
           "win_rate": round(float((tr["net"] > 0).mean()), 3) if len(tr) else None,
           "by_pair": stats.round(5).to_dict(orient="index")}
    print(json.dumps(out, indent=2, default=str))
    if len(tr):
        print(tr.to_string(index=False))
    path = Path(cfg.run.output_dir) / "lag_replay"
    path.mkdir(parents=True, exist_ok=True)
    tr.to_csv(path / "trades.csv", index=False)
    (path / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
