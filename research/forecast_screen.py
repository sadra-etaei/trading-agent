"""Fast offline four-hour return forecast screening; no production changes."""
from pathlib import Path
import os
import sys
import json
import socket
import hashlib
from datetime import datetime, timezone
import math

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from engine.config import load_config
from engine.data.store import load_1m, resample
from engine.features.build import build_features, feature_list
from engine.wallex.allowed import is_allowed

FEE = .0013
BUFFER = .002
HORIZON = 4
SIZE = .90
STOP = .02
INITIAL = 18.3
PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "PEPEUSDT", "BNBUSDT", "DOGEUSDT", "TRXUSDT"]
SLIP = {"BTCUSDT": .0003, "ETHUSDT": .0009, "XRPUSDT": .0008, "SOLUSDT": .001,
        "PEPEUSDT": .0023, "BNBUSDT": .0035, "DOGEUSDT": .0015, "TRXUSDT": .0015}


def simulate(start, end, forecasts, frames, rules, store_curves=True):
    """Spot, idle USDT. Require traded minutes for entries and exits; no zero-volume fills."""
    ix = pd.date_range(start, end, freq="1min", inclusive="left")
    arr = {}
    for sym, frame in frames.items():
        d = frame.set_index(pd.to_datetime(frame.ts, unit="s", utc=True))
        d = d.reindex(ix)
        arr[sym] = {c: d[c].to_numpy(float) for c in ["open", "high", "low", "close", "volume"]}
    # Scores keyed by the hour of execution, derived from the preceding completed hour.
    signals = {}
    for sym, pred in forecasts.items():
        for ts, value in pred.items():
            signals.setdefault(ts + pd.Timedelta(hours=1), []).append((float(value) - 2 * (FEE + SLIP[sym]), sym, float(value)))
    for candidates in signals.values():
        candidates.sort(reverse=True)
    cash, peak, ddmax, day_start = INITIAL, INITIAL, 0., INITIAL
    pos, trades, curve = None, [], []
    cooldown, last_prices = {}, {}
    current_day = None
    halted = False

    def exit_position(i, raw, reason):
        nonlocal cash, pos
        sym = pos["symbol"]
        price = raw * (1 - SLIP[sym])
        proceeds = pos["qty"] * price * (1 - FEE)
        pnl = proceeds - pos["spent"]
        cash += proceeds
        trades.append({**pos, "exit_ts": str(ix[i]), "exit_price": price,
                       "pnl_usdt": pnl, "return_pct": pnl / pos["spent"] * 100, "reason": reason})
        if pnl < 0:
            cooldown[sym] = ix[i] + pd.Timedelta(hours=1)
        pos = None

    for i, t in enumerate(ix):
        for sym, a in arr.items():
            if np.isfinite(a["open"][i]):
                last_prices[sym] = a["open"][i]
        eq = cash + (pos["qty"] * last_prices.get(pos["symbol"], pos["entry_price"]) if pos else 0)
        day = (t + pd.Timedelta(hours=3.5)).date()
        if day != current_day:
            current_day, day_start = day, eq
        peak = max(peak, eq)
        ddmax = max(ddmax, 1 - eq / peak)
        if 1 - eq / peak >= .30:
            halted = True
        if pos:
            a = arr[pos["symbol"]]
            if a["volume"][i] > 0 and np.isfinite(a["open"][i]):
                reason, px = None, a["open"][i]
                if halted:
                    reason = "drawdown_halt"
                elif t >= pd.Timestamp(pos["entry_ts"]) + pd.Timedelta(hours=HORIZON):
                    reason = "four_hour_exit"
                elif a["low"][i] <= pos["entry_price"] * (1 - STOP):
                    reason, px = "stop_loss", min(a["open"][i], pos["entry_price"] * (1 - STOP))
                if reason:
                    exit_position(i, px, reason)
        if not pos and not halted and t.minute == 1 and t + pd.Timedelta(hours=HORIZON) < end:
            if cash >= day_start * .92:
                for net, sym, pred in signals.get(t.floor("h"), []):
                    if net <= BUFFER or t < cooldown.get(sym, start):
                        continue
                    a = arr[sym]
                    if not (a["volume"][i] > 0 and np.isfinite(a["open"][i])):
                        continue
                    rule = rules[sym]
                    px = a["open"][i] * (1 + SLIP[sym])
                    step = 10 ** int(rule["qty_dec"])
                    qty = math.floor(cash * SIZE / (px * (1 + FEE)) * step) / step
                    gross = qty * px
                    if gross < float(rule["min_notional"]) or qty < float(rule["min_qty"]):
                        continue
                    spent = gross * (1 + FEE)
                    cash -= spent
                    pos = {"symbol": sym, "entry_ts": str(t), "entry_price": px, "qty": qty,
                           "spent": spent, "forecast_return_pct": pred * 100, "forecast_net_pct": net * 100}
                    # Entry pays spread immediately. Stops can trigger within the entry minute.
                    if a["low"][i] <= px * (1 - STOP):
                        exit_position(i, min(a["open"][i], px * (1 - STOP)), "stop_loss")
                    break
        for sym, a in arr.items():
            if np.isfinite(a["close"][i]):
                last_prices[sym] = a["close"][i]
        eq = cash + (pos["qty"] * last_prices.get(pos["symbol"], pos["entry_price"]) if pos else 0)
        peak = max(peak, eq)
        ddmax = max(ddmax, 1 - eq / peak)
        if store_curves and (t.minute == 59 or i == len(ix) - 1):
            curve.append({"ts": str(t), "equity_usdt": eq, "cash_usdt": cash, "drawdown": 1 - eq / peak})
    unfinished = bool(pos)
    final = cash + (pos["qty"] * last_prices[pos["symbol"]] * (1 - SLIP[pos["symbol"]]) * (1 - FEE) if pos else 0)
    # Unfinished positions are marked at estimated liquidation value, never recorded as executed sells.
    tt = pd.DataFrame(trades)
    gains = sum(max(0, r["pnl_usdt"]) for r in trades)
    losses = -sum(min(0, r["pnl_usdt"]) for r in trades)
    m = {"start": str(start), "end": str(end), "return_usdt_pct": (final / INITIAL - 1) * 100,
         "trades": len(trades), "profit_factor": gains / losses if losses else (float("inf") if gains else 0),
         "max_drawdown_pct": ddmax * 100, "unfinished": unfinished, "final_usdt": final}
    return m, tt, pd.DataFrame(curve)


def main():
    def offline(*args, **kwargs):
        raise RuntimeError("Offline experiment: network disabled")
    socket.create_connection = offline
    socket.socket.connect = offline
    out = ROOT / "research" / ("forecast-screen-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    out.mkdir()
    cfg = load_config()
    protected = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / "config.yaml", ROOT / "models/latest.joblib"]}
    pairs = [s for s in PAIRS if is_allowed(s, cfg.universe.blocked_file)]
    protocol = {"written_before_results": True, "model": "LightGBM regression, 200 trees, 15 leaves, lr .03, min_child_samples 200, lambda 5, seed 42, 4 threads",
        "target": "4-hour simple return of Wallex USDT hourly close; no price level or Binance/funding inputs",
        "test": "monthly expanding chronological fits July-October 2026, 5-hour label purge; all settings fixed, no threshold search",
        "strategy": "hourly rank by forecast return minus round-trip costs; trade only net forecast >0.20%; 90% size, 1 position, 4h exit, 2% stop, idle USDT, 8% day-entry pause, 30% DD halt, 1h cooldown after loss",
        "execution": "1-minute entry delay; active/traded minute required for entry and exit; cached market minimums and quantity precision; .13% taker per side plus measured half-spread or conservative .15% fallback",
        "initial_usdt": INITIAL, "pairs": pairs, "slippage_each_side": SLIP,
        "go_rule": "positive compounded USDT return, >=30 completed trades, pooled PF>=1.2, >=3 positive months, no unfinished positions; otherwise no promotion",
        "limitations": "quick screening, not a proven deployment; stored candle fills approximate executable quotes; present-day costs/rules applied historically; no GPT; historical periods previously explored; forecast endpoint target differs from stopped trading payoff; conversion costs from initial/final Toman omitted"}
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2))
    print("ARTIFACT_DIR=" + str(out), flush=True)
    frames = {s: load_1m(cfg.data.dir, s) for s in pairs}
    bars = {s: resample(d, "1h") for s, d in frames.items()}
    assert all(len(b) > 1000 for b in bars.values())
    # No Toman context: all modeled inputs come from the eight Wallex USDT markets.
    feats = build_features(bars)
    cols = feature_list(False, False, cfg.model.exclude_features, False)
    assert not any(c.startswith("bn_") or "funding" in c for c in cols)
    tab = []
    for sym, f in feats.items():
        f["pair_id"] = pairs.index(sym)
        f["target"] = bars[sym].close.shift(-HORIZON) / bars[sym].close - 1
        f["symbol"] = sym
        tab.append(f)
    cols += ["pair_id"]
    table = pd.concat(tab).sort_index().replace([np.inf, -np.inf], np.nan)
    rules = json.loads((ROOT / "data/markets.json").read_text())
    available_end = min(b.index.max() for b in bars.values())
    rows, trade_parts, curves, pred_parts = [], [], [], []
    for start in pd.date_range("2026-07-01", "2026-10-01", freq="MS", tz="UTC"):
        end = min(start + pd.offsets.MonthBegin(1), available_end)
        fit_end = start - pd.Timedelta(hours=HORIZON + 1)
        tr = table[(table.index < fit_end) & table.target.notna() & table.dist_ema200.notna()]
        test = table[(table.index >= start - pd.Timedelta(hours=1)) & (table.index < end)]
        print(f"FIT {start:%Y-%m}: {len(tr):,} rows; fit ends before {fit_end}", flush=True)
        model = lgb.LGBMRegressor(n_estimators=200, learning_rate=.03, num_leaves=15, min_child_samples=200,
            reg_lambda=5., subsample=.8, subsample_freq=1, colsample_bytree=.8, random_state=42, n_jobs=4, verbosity=-1)
        model.fit(tr[cols], tr.target)
        raw = model.predict(test[cols])
        test = test.assign(prediction=raw)
        prediction = {s: g.loc[g.stale == 0, "prediction"] for s, g in test.groupby("symbol")}
        m, trades, curve = simulate(start, end, prediction, frames, rules)
        scored = test[test.target.notna() & (test.stale == 0) & (test.index < end - pd.Timedelta(hours=HORIZON + 1))]
        m.update({"forecast_mae_pct": float(np.mean(np.abs(scored.prediction - scored.target)) * 100),
            "naive_zero_return_mae_pct": float(np.mean(np.abs(scored.target)) * 100), "fit_end": str(fit_end), "train_rows": len(tr)})
        rows.append(m)
        fold = out / start.strftime("%Y%m")
        fold.mkdir()
        joblib.dump({"model": model, "features": cols, "fit_end": str(fit_end), "protocol": protocol}, fold / "model.joblib")
        trades.to_csv(fold / "trades.csv", index=False)
        curve.to_csv(fold / "equity.csv", index=False)
        scored[["symbol", "prediction", "target", "close"]].to_csv(fold / "predictions.csv")
        trade_parts.append(trades)
        curves.append(curve)
        pred_parts.append(scored)
        (fold / "metrics.json").write_text(json.dumps(m, indent=2))
        print("RESULT " + json.dumps(m), flush=True)
    trades = pd.concat(trade_parts, ignore_index=True)
    g = trades.loc[trades.pnl_usdt > 0, "pnl_usdt"].sum() if len(trades) else 0
    loss = -trades.loc[trades.pnl_usdt < 0, "pnl_usdt"].sum() if len(trades) else 0
    pooled_pf = float(g / loss) if loss else (float("inf") if g else 0)
    compounded = float((np.prod([1 + m["return_usdt_pct"] / 100 for m in rows]) - 1) * 100)
    eligible = compounded > 0 and len(trades) >= 30 and pooled_pf >= 1.2 and sum(m["return_usdt_pct"] > 0 for m in rows) >= 3 and not any(m["unfinished"] for m in rows)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in protected.items())
    allpred = pd.concat(pred_parts)
    summary = {"compound_return_usdt_pct": compounded, "hold_usdt_return_pct": 0., "trades": len(trades),
        "pooled_profit_factor": pooled_pf, "positive_months": int(sum(m["return_usdt_pct"] > 0 for m in rows)),
        "worst_month_dd_pct": max(m["max_drawdown_pct"] for m in rows), "forecast_mae_pct": float(np.mean(np.abs(allpred.prediction - allpred.target)) * 100),
        "naive_zero_return_mae_pct": float(np.mean(np.abs(allpred.target)) * 100), "eligible": bool(eligible), "protected_unchanged": True, "folds": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# Four-hour return forecasting: quick screening", "", "Offline Wallex-only USDT markets; no live changes. All returns below are in USDT. Benchmark: hold USDT (0%).", "",
        f"Compounded monthly return: **{compounded:+.2f}%**. Completed trades: **{len(trades)}**. Pooled PF: **{pooled_pf:.2f}**. Eligible under prewritten rule: **{eligible}**.", "",
        "| Test month | USDT return | Trades | PF | Max monthly DD | Forecast MAE | No-change MAE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for m in rows:
        lines.append(f"| {m['start'][:7]} | {m['return_usdt_pct']:+.2f}% | {m['trades']} | {m['profit_factor']:.2f} | {m['max_drawdown_pct']:.2f}% | {m['forecast_mae_pct']:.3f}% | {m['naive_zero_return_mae_pct']:.3f}% |")
    lines += ["", "Monthly accounts restart at 18.3 USDT; compounded returns chain those percentages. DD is worst within-month, not full compounded-period DD. 4h endpoint prediction target differs from actual stopped trade payoff. Point forecasts only: uncertainty model not included. Costs: 0.13% fee each side plus per-pair half-spread; fixed net entry buffer 0.20%; one-minute delay and active minutes required. 90% exposure, 2% stops, one position. Cash idle in USDT. Initial/final Toman conversion costs excluded. Candle fills approximate historical quotes; historical periods previously examined; no GPT. This is a screening test, not an untouched prospective validation."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # Example chart: compare forecasts at their target time, never at their input time.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    btc = allpred[allpred.symbol == "BTCUSDT"].sort_index().tail(100).copy()
    target_time = btc.index + pd.Timedelta(hours=HORIZON + 1)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [2, 1]})
    ax.plot(target_time, btc.close * (1 + btc.target), label="Actual target price", color="#235b9a")
    ax.plot(target_time, btc.close * (1 + btc.prediction), label="4h forecast", color="#e19028", alpha=.9)
    ax.set_title("BTC/USDT · out-of-sample 4-hour forecasts · Wallex data")
    ax.set_ylabel("Target price · USDT")
    ax.legend()
    ax.grid(alpha=.25)
    factor = 1.
    for curve in curves:
        if len(curve):
            ax2.plot(pd.to_datetime(curve.ts), (curve.equity_usdt / INITIAL * factor - 1) * 100, color="#235b9a")
            factor *= float(curve.equity_usdt.iloc[-1]) / INITIAL
    ax2.axhline(0, color="gray", ls="--", label="Hold USDT")
    ax2.set_ylabel("Strategy return · % USDT")
    ax2.legend()
    ax2.grid(alpha=.25)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.text(.07, .015, "Historical screening. Forecast lines aligned to target time. No Binance inputs. Trading costs included; Toman conversions excluded.", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, 1])
    fig.savefig(out / "forecast-comparison.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
