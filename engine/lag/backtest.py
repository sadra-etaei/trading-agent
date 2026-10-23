"""Historical lag-buy test on 1-minute candles (PLAN.md §15 item 2), with the go rule fixed in advance.

Wallex has no order book history, so the test is built to be pessimistic:
- the Wallex price is the last trade of each minute; the ask is that price plus half the spread
  (`lag.half_spread_est`, measured from live books);
- the basis uses only minutes in which Wallex traded (a stale last trade is not a price);
- the signal is computed at the END of minute t; the buy limit (the ask at the signal) only fills
  in minute t+1 if a real Wallex trade printed at or below it there (one minute late, needs a seller);
- every exit pays the taker fee; a stop or time exit sells at the lower of the Wallex last trade and
  the Binance-implied fair price, minus half the spread;
- when a Binance reversal and the take-profit fall in the same minute, the reversal wins.
`--optimistic` also reports the result when the entry fills without needing a Wallex trade.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data import binance as bn1m
from engine.data.store import load_1m
from engine.lag.rules import bn_threshold, exit_reason

log = logging.getLogger(__name__)


def minute_frame(data_dir: str, symbol: str) -> pd.DataFrame | None:
    wx = load_1m(data_dir, symbol)
    b = bn1m.binance_symbol(symbol)
    bn = bn1m.load(data_dir, b) if b else None
    if wx.empty or bn is None or bn.empty:
        return None
    wx = wx.set_index("ts")[["open", "high", "low", "close", "volume"]]
    bn = bn.drop_duplicates("ts").set_index("ts")["close"].rename("bn")
    idx = np.arange(max(wx.index.min(), bn.index.min()), min(wx.index.max(), bn.index.max()) + 60, 60)
    f = wx.reindex(idx)
    f["active"] = f["volume"].fillna(0) > 0
    f["close"] = f["close"].ffill()
    f["bn"] = bn.reindex(idx).ffill()
    f.index = pd.to_datetime(f.index, unit="s", utc=True)
    return f.dropna(subset=["close", "bn"])


def add_signals(f: pd.DataFrame, cfg_lag, symbol: str, hs: float) -> pd.DataFrame:
    ratio = (f["close"] / f["bn"] - 1).where(f["active"])
    f["basis"] = ratio.rolling(f"{cfg_lag['basis_window_min']}min", min_periods=5).median()
    f["fair"] = f["bn"] * (1 + f["basis"])
    f["ask"] = f["close"] * (1 + hs)
    f["bn_ret_3m"] = f["bn"] / f["bn"].shift(3) - 1
    f["signal"] = (f["ask"] < f["fair"] * (1 - cfg_lag["entry_gap"])) & (f["bn_ret_3m"] >= bn_threshold(cfg_lag, symbol))
    return f


def simulate(f: pd.DataFrame, cfg_lag, symbol: str, hs: float, fee: float, optimistic: bool = False) -> list[dict]:
    trades, i, n = [], 0, len(f)
    sig = np.flatnonzero(f["signal"].to_numpy())
    close, high, low = f["close"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy()
    active, bnp, fair, ask = f["active"].to_numpy(), f["bn"].to_numpy(), f["fair"].to_numpy(), f["ask"].to_numpy()
    busy_until = -1
    for t in sig:
        if t <= busy_until or t + 1 >= n:
            continue
        e = t + 1  # order is live during the next minute
        limit = ask[t]
        filled = optimistic or (active[e] and low[e] <= limit)
        if not filled:
            continue
        entry, bn_entry = limit, bnp[e]
        reason, exit_px, m = None, None, e
        for m in range(e + 1, min(n, e + 1 + 60)):
            wx_high = high[m] if active[m] else -np.inf
            reason = exit_reason(cfg_lag, entry, bn_entry, bnp[m], wx_high, m - e)
            if reason == "take_profit":
                exit_px = entry * (1 + cfg_lag["take_profit"])
                break
            if reason:
                implied = fair[m] if np.isfinite(fair[m]) else close[m]
                exit_px = min(close[m], implied) * (1 - hs)
                break
        if exit_px is None:
            continue
        net = (exit_px * (1 - fee)) / (entry * (1 + fee)) - 1
        trades.append({"symbol": symbol, "signal_time": f.index[t], "exit_time": f.index[m], "entry": entry,
                       "exit": exit_px, "reason": reason, "minutes": m - e, "net": net,
                       "gap": ask[t] / fair[t] - 1, "bn_ret_3m": f["bn_ret_3m"].iat[t]})
        busy_until = m
    return trades


def summarize(trades: pd.DataFrame, days: float, size_frac: float) -> dict:
    if trades.empty:
        return {"trades": 0, "per_day": 0.0, "win_rate": None, "mean_net_pct": None, "profit_factor": None,
                "sum_net_pct": 0.0, "account_pct": 0.0}
    r = trades["net"]
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    acct = float(np.prod(1 + size_frac * trades.sort_values("exit_time")["net"]) - 1)
    return {"trades": int(len(r)), "per_day": round(len(r) / days, 2), "win_rate": round(float((r > 0).mean()), 3),
            "mean_net_pct": round(100 * float(r.mean()), 3),
            "profit_factor": round(float(gains / losses), 2) if losses > 0 else float("inf"),
            "sum_net_pct": round(100 * float(r.sum()), 2), "account_pct": round(100 * acct, 2)}


def windows(frames: dict[str, pd.DataFrame], cfg) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    end = min(f.index.max() for f in frames.values())
    test_start = end - pd.Timedelta(days=cfg.model.test_days)
    return {"validation": (test_start - pd.Timedelta(days=cfg.model.val_days), test_start), "test": (test_start, end)}


def run(cfg, optimistic: bool = False) -> dict:
    lag = cfg.lag
    fee = cfg.costs.usdt_taker_fee
    frames = {}
    for s in lag.pairs:
        f = minute_frame(cfg.data.dir, s)
        if f is None:
            log.warning("%s: no Wallex or Binance 1-minute data, skipped", s)
            continue
        frames[s] = add_signals(f, lag, s, lag.half_spread_est[s])
    wins = windows(frames, cfg)
    out = {"fee": fee, "params": {k: v for k, v in lag.items()}, "windows": {}}
    for mode in ([False, True] if optimistic else [False]):
        name = "optimistic" if mode else "strict"
        all_tr = pd.DataFrame([t for s, f in frames.items()
                               for t in simulate(f, lag, s, lag.half_spread_est[s], fee, mode)])
        for w, (a, b) in wins.items():
            tr = all_tr[(all_tr["signal_time"] >= a) & (all_tr["signal_time"] < b)] if len(all_tr) else all_tr
            days = (b - a) / pd.Timedelta(days=1)
            res = {"from": str(a), "to": str(b), "all": summarize(tr, days, lag.size_frac),
                   "by_pair": {s: summarize(g, days, lag.size_frac) for s, g in tr.groupby("symbol")} if len(tr) else {},
                   "by_reason": tr["reason"].value_counts().to_dict() if len(tr) else {}}
            out["windows"][f"{name}/{w}"] = res
            if len(tr):
                path = Path(cfg.run.output_dir) / "lag_backtest"
                path.mkdir(parents=True, exist_ok=True)
                tr.to_csv(path / f"trades_{name}_{w}.csv", index=False)
    rule = lag.go_rule
    strict = [out["windows"][f"strict/{w}"]["all"] for w in wins]
    out["go"] = all(s["trades"] >= rule["min_trades"] and s["sum_net_pct"] > 0
                    and (s["profit_factor"] or 0) >= rule["min_profit_factor"] for s in strict)
    path = Path(cfg.run.output_dir) / "lag_backtest"
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    for k, v in out["windows"].items():
        a = v["all"]
        print(f"{k:22s} trades {a['trades']:4d} ({a['per_day']}/day) | win {a['win_rate']} | mean {a['mean_net_pct']}% "
              f"| PF {a['profit_factor']} | sum {a['sum_net_pct']}% | account @{lag.size_frac:.0%}/trade {a['account_pct']}% "
              f"| exits {v['by_reason']}")
        for s, p in v["by_pair"].items():
            print(f"    {s:9s} trades {p['trades']:4d} | win {p['win_rate']} | mean {p['mean_net_pct']}% | PF {p['profit_factor']}")
    print("GO (pre-registered rule, strict fills, both windows):", out["go"])
    return out
