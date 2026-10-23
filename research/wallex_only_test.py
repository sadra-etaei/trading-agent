"""Offline, isolated comparison of models for a Binance-unreachable deployment.

Run from any working directory: python <absolute path to this file>.
Only writes research artifacts; never changes the live model or configuration.
"""
from __future__ import annotations

import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from engine.agent.overlay import AgentOverlay
from engine.backtest.runner import challenge_windows, load_guard
from engine.backtest.sim import MinuteData, NullLog, simulate
from engine.config import load_config, universe
from engine.data import store
from engine.features.build import build_features, feature_list
from engine.model.dataset import load_inputs
from engine.model.labels import triple_barrier
from engine.model.predict import EntryModel
from engine.model.train import make_model, tune_threshold
from engine.reports.report import metrics
from engine.risk.manager import RiskManager
from engine.strategy.core import Strategy
from engine.wallex import markets as mk

BAR = pd.Timedelta(hours=1)
PURGE = pd.Timedelta(hours=13)


def blocked_network(*args, **kwargs):
    raise RuntimeError("This experiment is offline: network access is disabled")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_from(feats, bars, symbols, cfg):
    rows = []
    for sym in symbols:
        f = feats[sym]
        t = f.join(triple_barrier(bars[sym], f.atr_pct, cfg.labels))
        t["symbol"] = sym
        rows.append(t)
    t = pd.concat(rows).sort_index()
    return t[t.label.notna() & t.dist_ema200.notna()]


def run_sim(cfg, model, feats, minutes, markets, symbols, start, end, guard=None):
    risk = RiskManager(cfg)
    risk.set_window(start, end)
    strat = Strategy(cfg, model, risk, AgentOverlay(cfg, None), NullLog(), markets, symbols)
    pf, curve = simulate(cfg, strat, feats, minutes, start, end, risk, NullLog(), BAR, markets, guard=guard)
    trades = pd.DataFrame([asdict(t) for t in pf.trades])
    m = metrics(trades, curve, float(cfg.risk.initial_equity_tmn))
    m["open_at_end"] = len(pf.positions)
    return m, trades, curve


def main():
    socket.create_connection = blocked_network
    socket.socket.connect = blocked_network
    logging.basicConfig(level=logging.WARNING)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out = ROOT / "research" / ("wallex-only-" + stamp)
    out.mkdir(parents=True)
    protected = [ROOT / "config.yaml", ROOT / "models/latest.joblib"]
    before = {str(p): sha(p) for p in protected}
    cfg = load_config()
    cfg.agent.enabled = False
    cfg.challenge.start = cfg.challenge.end = None
    cfg.sentiment.fear_greed = False
    cfg.sentiment.funding = False
    cfg.outside.binance = False
    cfg.risk.binance_guard.enabled = False
    cfg.model.lgbm.n_jobs = 4
    symbols = universe(cfg)
    protocol = {
        "written_before_results": True,
        "network": "disabled; stored data only; GPT/news/event gates not simulated",
        "candidates": {"local": "Wallex TMN features only", "local_premium": "local plus fresh Wallex USDT/TMN premium features", "disconnected": "same-fold outside-data model, Binance features missing at execution", "full_reference": "same outside-data model with Binance features and crash guard (unavailable on Liara)"},
        "test_months": ["2026-07", "2026-08", "2026-09", "2026-10 (partial through available candles)"],
        "splits": "Expanding fit; separate preceding 30-day calibration and 30-day threshold-selection periods, 13-hour purge at each boundary; no test-month tuning",
        "threshold": "existing tune_threshold grid and row-based selection, on threshold period only; overlapping signal rows are not actual trades",
        "costs": "0.30% per side + 15bp per side, 1-minute entry delay; same S1 sizing and exits, no Binance guard except full_reference",
        "premium_freshness": "premium features missing when either reference USDT market or USDTTMN has <10% active minutes in current hour",
        "selection_rule": "Eligible if compounded return >0, pooled PF>=1.2, >=30 trades, >=3 strictly positive test months, worst rolling 3-day return >=-5%; highest compound return among local/local_premium; otherwise no recommendation",
        "limitations": "Historical periods already explored in project; chronological refits reduce leakage but are not fresh prospective evidence. Three-day windows overlap. Final month partial. No production activation.",
        "protected_before": before,
        "config": cfg,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    print(f"ARTIFACT_DIR={out}", flush=True)
    print("Loading local candles and building features", flush=True)
    bars, _, _, _ = load_inputs(cfg, symbols)
    local = build_features(bars)
    pcfg = copy.deepcopy(cfg)
    pcfg.universe.usdt_premium = True
    pbars, _, _, _ = load_inputs(pcfg, symbols)
    premium = build_features(pbars)
    coverage = {}
    for sym in symbols:
        ref = sym.removesuffix("TMN") + "USDT"
        if sym == "USDTTMN" or ref not in pbars:
            coverage[sym] = 0.0
            continue
        active = (pbars[ref].active_min >= 6) & (pbars["USDTTMN"].active_min.reindex(pbars[ref].index) >= 6)
        mask = active.reindex(premium[sym].index).fillna(False).astype(bool)
        premium[sym].loc[~mask, ["prem", "prem_z", "prem_chg_3"]] = np.nan
        coverage[sym] = float(premium[sym].prem.notna().mean())
    (out / "premium_coverage.json").write_text(json.dumps(coverage, indent=2))
    extcfg = copy.deepcopy(cfg)
    extcfg.outside.binance = True
    extcfg.sentiment.funding = True
    ebars, _, fund, bn = load_inputs(extcfg, symbols)
    external = build_features(ebars, None, fund, bn)
    missing = copy.deepcopy(external)
    ext_feats = feature_list(True, False, cfg.model.exclude_features, True)
    local_feats = feature_list(False, False, cfg.model.exclude_features, False)
    for f in missing.values():
        for col in set(ext_feats) - set(local_feats):
            f[col] = np.nan
    minutes = MinuteData({s: store.load_1m(cfg.data.dir, s) for s in symbols})
    markets = mk.load(cfg)
    assert markets, "Local markets cache required; no network fallback allowed"
    datasets = {"local": (cfg, local, bars, local_feats), "local_premium": (pcfg, premium, pbars, feature_list(False, True, cfg.model.exclude_features, False)), "disconnected": (extcfg, external, ebars, ext_feats)}
    tables = {name: table_from(f, b, symbols, c) for name, (c, f, b, cols) in datasets.items()}
    available_end = min(bars[s].index.max() for s in symbols)  # exclude possibly incomplete last hour
    data_manifest = {}
    for sym in sorted(pbars):
        p = store.candle_path(cfg.data.dir, sym)
        if p.exists():
            data_manifest[sym] = {"sha256": sha(p), "start": str(pbars[sym].index.min()), "end": str(pbars[sym].index.max())}
    (out / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2))
    rows, all_trades, windows = [], {}, {}
    for start in pd.date_range("2026-07-01", "2026-10-01", freq="MS", tz="UTC"):
        end = min(start + pd.offsets.MonthBegin(1), available_end)
        tune_end, tune_start = start - PURGE, start - PURGE - pd.Timedelta(days=30)
        cal_end, cal_start = tune_start - PURGE, tune_start - PURGE - pd.Timedelta(days=30)
        fit_end = cal_start - PURGE
        for name, (train_cfg, feat, b, cols) in datasets.items():
            print(f"FIT {name} {start:%Y-%m}: fit<{fit_end}, calibration {cal_start}..{cal_end}, threshold {tune_start}..{tune_end}", flush=True)
            tab = tables[name]
            tr = tab[tab.index < fit_end]
            cal = tab[(tab.index >= cal_start) & (tab.index < cal_end)]
            tune = tab[(tab.index >= tune_start) & (tab.index < tune_end)]
            test = tab[(tab.index >= start) & (tab.index < end - PURGE)]
            assert len(tr) and len(cal) and len(tune) and len(test)
            clf = make_model(train_cfg)
            clf.fit(tr[cols], tr.label.astype(int))
            iso = IsotonicRegression(out_of_bounds="clip").fit(clf.predict_proba(cal[cols])[:, 1], cal.label.astype(int))
            threshold, grid = tune_threshold(tune, iso.predict(clf.predict_proba(tune[cols])[:, 1]), cfg)
            bundle = {"model": clf, "calibrator": iso, "features": cols, "version": f"{stamp}-{name}-{start:%Y%m}", "thresholds": {"min_ev_pct": threshold}, "symbols": symbols, "splits": {"fit_end": str(fit_end), "cal_start": str(cal_start), "cal_end": str(cal_end), "tune_start": str(tune_start), "tune_end": str(tune_end), "test_start": str(start), "test_end": str(end)}}
            fold_dir = out / f"{name}-{start:%Y%m}"
            fold_dir.mkdir()
            joblib.dump(bundle, fold_dir / "model.joblib")
            (fold_dir / "threshold_grid.json").write_text(json.dumps(grid, indent=2))
            model = EntryModel(bundle)
            variants = [(name, cfg if name == "disconnected" else train_cfg, missing if name == "disconnected" else feat, None)]
            if name == "disconnected":
                gc = copy.deepcopy(extcfg)
                gc.risk.binance_guard.enabled = True
                variants.append(("full_reference", gc, external, load_guard(gc, symbols)))
            for label, run_cfg, run_feat, guard in variants:
                m, trades, curve = run_sim(run_cfg, model, run_feat, minutes, markets, symbols, start, end, guard)
                dest = out / f"{label}-{start:%Y%m}"
                dest.mkdir(exist_ok=True)
                trades.to_csv(dest / "trades.csv", index=False)
                curve.to_csv(dest / "equity.csv", index=False)
                win = challenge_windows(run_cfg, model, run_feat, minutes, start, end, BAR, markets, symbols, guard=guard)
                win.to_csv(dest / "challenge_windows.csv", index=False)
                windows.setdefault(label, []).append(win)
                all_trades.setdefault(label, []).append(trades)
                auc = roc_auc_score(test.label, clf.predict_proba(test[cols])[:, 1])
                row = {"candidate": label, "month": str(start.date()), "end": str(end), "threshold": threshold, "auc_raw": float(auc), **m}
                rows.append(row)
                (dest / "metrics.json").write_text(json.dumps(row, indent=2, default=str))
                print(f"RESULT {label} {start:%Y-%m}: {m['net_return_pct']:+.2f}%, DD {m['max_drawdown_pct']:.2f}%, {m['trades']} trades, PF {m.get('profit_factor', 0):.2f}", flush=True)
                pd.DataFrame(rows).to_json(out / "fold_results.json", orient="records", indent=2)
    summary = []
    for name, trade_parts in all_trades.items():
        tt = pd.concat(trade_parts, ignore_index=True)
        ww = pd.concat(windows[name], ignore_index=True)
        rr = [r for r in rows if r["candidate"] == name]
        gains = tt.loc[tt.pnl_tmn > 0, "pnl_tmn"].sum() if len(tt) else 0
        losses = -tt.loc[tt.pnl_tmn < 0, "pnl_tmn"].sum() if len(tt) else 0
        pf = float(gains / losses) if losses else (float("inf") if gains else 0)
        total = float((np.prod([1 + r["net_return_pct"] / 100 for r in rr]) - 1) * 100)
        positive = sum(r["net_return_pct"] > 0 for r in rr)
        worst = float(ww.return_pct.min())
        eligible = total > 0 and pf >= 1.2 and len(tt) >= 30 and positive >= 3 and worst >= -5
        summary.append({"candidate": name, "compound_return_pct": total, "profit_factor": pf, "trades": len(tt), "positive_months": positive, "worst_month_dd_pct": max(r["max_drawdown_pct"] for r in rr), "three_day_mean_pct": float(ww.return_pct.mean()), "three_day_worst_pct": worst, "three_day_positive_pct": float((ww.return_pct > 0).mean() * 100), "three_day_windows": len(ww), "flat_windows": bool((ww.open_positions_at_end == 0).all()), "eligible": bool(eligible)})
    eligible = [s for s in summary if s["candidate"] in ("local", "local_premium") and s["eligible"]]
    chosen = max(eligible, key=lambda s: s["compound_return_pct"])["candidate"] if eligible else "none"
    assert before == {str(p): sha(p) for p in protected}, "Protected production file changed during test"
    result = {"summary": summary, "selection": chosen, "protected_unchanged": True}
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    lines = ["# Wallex-only deployment experiment", "", "Protocol: protocol.json. Historical evaluation only; no live configuration changed.", "", "Four chronological monthly folds. Fit, calibration, threshold selection and test are separate, with 13-hour purges. Fees 0.30% and slippage 0.15% per side; entry delay 1 minute. Monthly accounts restart at 5M TMN; compounded return chains their percentage returns. Maximum drawdown below is the worst within-month drawdown, not continuous-period drawdown. GPT/news omitted.", "", "| Candidate | Compound return | Trades | Pooled PF | Positive months | Worst month DD | 3-day mean | 3-day worst | Eligible |", "|---|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(f"| {s['candidate']} | {s['compound_return_pct']:+.2f}% | {s['trades']} | {s['profit_factor']:.2f} | {s['positive_months']}/4 | {s['worst_month_dd_pct']:.2f}% | {s['three_day_mean_pct']:+.2f}% | {s['three_day_worst_pct']:+.2f}% | {s['eligible']} |")
    lines += ["", f"Selection under prewritten rule: **{chosen}**.", "", "Monthly details:", "", "| Candidate | Month | Return | DD | Trades | PF |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['candidate']} | {r['month']} | {r['net_return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | {r['trades']} | {r.get('profit_factor', 0):.2f} |")
    lines += ["", "Limitations: these historical periods have already been explored elsewhere in this project. This is chronological testing, not an untouched prospective holdout. Rolling windows overlap. Premium-reference coverage is in premium_coverage.json; missing references are not synthesized. Full-reference benchmark needs Binance and is not deployable under the stated constraints. Source data hashed in data_manifest.json. Production config and latest.joblib hashes unchanged."]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
