"""Backtest of record + rolling 3-day "challenge" windows.

Writes runs/bt-<time>[-tag]/:
  meta.json            config, config hash, model version, data manifest, window
  decisions.jsonl      every scan / entry / veto / risk block / fill / exit
  trades.csv, equity.csv
  challenge_windows.csv  one row per simulated 3-day challenge (fresh 5M TMN, flatten at end)
  report.md + charts
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from engine.agent.llm import from_config as llm_from_config
from engine.agent.overlay import AgentOverlay
from engine.backtest.sim import MinuteData, NullLog, simulate
from engine.config import config_hash, universe
from engine.data import store
from engine.decision_log import DecisionLog
from engine.features.build import build_features
from engine.model.dataset import load_inputs
from engine.model.predict import EntryModel
from engine.reports.report import build_report
from engine.risk.manager import RiskManager
from engine.strategy.core import Strategy
from engine.wallex import markets as mk
from engine.wallex.client import WallexClient

log = logging.getLogger(__name__)

BAR = {"15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}


def _agent(cfg, use_agent: bool):
    if not use_agent:
        return AgentOverlay(cfg, None)
    llm = llm_from_config(cfg)
    a = AgentOverlay(cfg, llm, news=None)  # no historical headlines exist -> market-state review only
    a.enabled = True
    return a


def run_backtest(cfg, start=None, end=None, model_path=None, tag="", use_agent=False, windows=True) -> Path:
    model = EntryModel.load(model_path or Path(cfg.model.dir) / "latest.joblib")
    symbols = [s for s in universe(cfg) if s in model.bundle["symbols"]]
    bars, fng, fund, bn = load_inputs(cfg, symbols)
    feats = build_features(bars, fng, fund, bn)
    if model.bundle.get("sequence"):  # GRU only: PyTorch is loaded only when needed
        from engine.model.seq import add_seq_proba

        add_seq_proba(model.bundle, feats)
    m1 = {s: store.load_1m(cfg.data.dir, s) for s in symbols}
    minutes = MinuteData(m1)
    markets = mk.load(cfg) or mk.load(cfg, WallexClient.from_config(cfg))
    bar = BAR[cfg.data.bar]

    splits = model.bundle["splits"]
    start = pd.Timestamp(start or splits["test_start"])
    end = pd.Timestamp(end or splits["data_end"]) + bar
    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    if start < pd.Timestamp(splits["val_end"]):
        log.warning("backtest window starts before the held-out test window: results are in-sample")

    run_id = "bt-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "")
    run_dir = Path(cfg.run.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    dlog = DecisionLog(run_dir, run_id, "backtest")
    meta = {
        "run_id": run_id, "mode": "backtest", "window": [start.isoformat(), end.isoformat()],
        "model_version": model.version, "model_splits": splits, "symbols": symbols,
        "config_hash": config_hash(cfg), "config": cfg, "agent": bool(use_agent),
        "data_manifest": store.write_manifest(cfg.data.dir, symbols),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    risk = RiskManager(cfg)
    strat = Strategy(cfg, model, risk, _agent(cfg, use_agent), dlog, markets, symbols)
    log.info("backtest %s .. %s on %s (EV threshold %.4f)", start, end, symbols, strat.ev_thr)
    guard = load_guard(cfg, symbols)
    pf, curve = simulate(cfg, strat, feats, minutes, start, end, risk, dlog, bar, markets, guard=guard)
    pd.DataFrame([asdict(t) for t in pf.trades]).to_csv(run_dir / "trades.csv", index=False)
    curve.to_csv(run_dir / "equity.csv", index=False)

    if windows:
        wins = challenge_windows(cfg, model, feats, minutes, start, end, bar, markets, symbols, guard=guard)
        wins.to_csv(run_dir / "challenge_windows.csv", index=False)
    print(build_report(run_dir))
    return run_dir


def load_guard(cfg, symbols) -> dict | None:
    """Binance 3-minute returns per Wallex symbol, if the crash guard is enabled."""
    if not (cfg.risk.get("binance_guard") or {}).get("enabled"):
        return None
    from engine.data import binance

    out = {}
    for s in symbols:
        b = binance.binance_symbol(s)
        df = binance.load(cfg.data.dir, b) if b else None
        if df is not None and len(df):
            out[s] = binance.ret_3m(df)
    log.info("crash guard: Binance data for %d/%d pairs", len(out), len(symbols))
    return out


def challenge_windows(cfg, model, feats, minutes, start, end, bar, markets, symbols,
                      days: int = 3, step_days: int = 1, guard: dict | None = None) -> pd.DataFrame:
    """Re-runs the exact challenge rules (fresh capital, ramp-up, no entries in the last hours,
    flatten to TMN at the end) on every 3-day window in the test period."""
    rows = []
    t = start
    while t + pd.Timedelta(days=days) <= end:
        w_end = t + pd.Timedelta(days=days)
        risk = RiskManager(cfg)
        risk.set_window(t, w_end)
        strat = Strategy(cfg, model, risk, AgentOverlay(cfg, None), NullLog(), markets, symbols)
        pf, curve = simulate(cfg, strat, feats, minutes, t, w_end, risk, NullLog(), bar, markets, guard=guard)
        final = curve["equity"].iloc[-1] if len(curve) else pf.initial
        rows.append({
            "start": t.isoformat(), "end": w_end.isoformat(),
            "return_pct": (final / pf.initial - 1) * 100,
            "max_drawdown_pct": float(curve["drawdown"].max() * 100) if len(curve) else 0.0,
            "trades": len(pf.trades),
            "volume_tmn": sum(2 * tr.notional_tmn for tr in pf.trades),
            "fees_tmn": pf.fees_paid,
            "open_positions_at_end": len(pf.positions),
        })
        t += pd.Timedelta(days=step_days)
    return pd.DataFrame(rows)
