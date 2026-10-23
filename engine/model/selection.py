"""Two-stage selection model experiment (PLAN.md 3.2).

Stage 1 ranks pairs every 4h by predicted next-24h return *relative to the other pairs*
(cross-sectional momentum / relative strength). Stage 2 (the entry model) only trades the top-k.
It is kept only if it beats plain EV ranking on the held-out window; otherwise it is reported as
tested-and-rejected.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.agent.overlay import AgentOverlay
from engine.backtest.runner import BAR, challenge_windows
from engine.backtest.sim import MinuteData, NullLog, simulate
from engine.config import universe
from engine.data import store
from engine.features.build import build_features
from engine.model.dataset import load_inputs
from engine.model.predict import EntryModel
from engine.reports.report import metrics
from engine.risk.manager import RiskManager
from engine.strategy.core import Strategy
from engine.wallex import markets as mk

log = logging.getLogger(__name__)

SEL_FEATURES = ["ret_24", "ret_72", "rel_btc_24", "rel_btc_72", "rvol_24", "atr_pct", "atr_rank", "vol_z",
                "adx", "dist_ema50", "dist_ema200", "rsi", "flow_24", "fng"]
FWD_BARS = 24
EVERY_H = 4


def selection_table(feats: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s, f in feats.items():
        if s == "USDTTMN":
            continue
        t = f[SEL_FEATURES].copy()
        t["fwd"] = np.log(f["close"].shift(-FWD_BARS) / f["close"])
        t["symbol"] = s
        rows.append(t[t.index.hour % EVERY_H == 0])
    tab = pd.concat(rows)
    tab["y"] = tab["fwd"] - tab.groupby(level=0)["fwd"].transform("mean")
    return tab


def train_selector(tab: pd.DataFrame, train_end: pd.Timestamp, seed: int = 42):
    import lightgbm as lgb

    tr = tab[(tab.index < train_end - pd.Timedelta(hours=FWD_BARS)) & tab["y"].notna()]
    m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=100,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    m.fit(tr[SEL_FEATURES], tr["y"])
    return m


class Selector:
    def __init__(self, model, feats: dict[str, pd.DataFrame], top_k: int):
        self.model, self.feats, self.top_k = model, feats, top_k
        self.current: set[str] | None = None

    def __call__(self, T: pd.Timestamp) -> set[str] | None:
        if T.hour % EVERY_H and self.current is not None:
            return self.current
        X = {s: f.loc[T, SEL_FEATURES] for s, f in self.feats.items() if s != "USDTTMN" and T in f.index}
        if not X:
            return self.current
        X = pd.DataFrame(X).T.apply(pd.to_numeric, errors="coerce")
        pred = pd.Series(self.model.predict(X), index=X.index).sort_values(ascending=False)
        self.current = set(pred.index[: self.top_k])
        return self.current


def run_experiment(cfg) -> Path:
    model = EntryModel.load(Path(cfg.model.dir) / "latest.joblib")
    symbols = [s for s in universe(cfg) if s in model.bundle["symbols"]]
    bars, fng, fund, bn = load_inputs(cfg, symbols)
    feats = build_features(bars, fng, fund, bn)
    minutes = MinuteData({s: store.load_1m(cfg.data.dir, s) for s in symbols})
    markets = mk.load(cfg)
    bar = BAR[cfg.data.bar]
    sp = model.bundle["splits"]
    start, end = pd.Timestamp(sp["test_start"]), pd.Timestamp(sp["data_end"]) + bar

    tab = selection_table(feats)
    sel_model = train_selector(tab, pd.Timestamp(sp["val_end"]), cfg.run.seed)
    test = tab[(tab.index >= start) & tab["y"].notna()]
    ic = test.groupby(level=0).apply(lambda g: g["y"].corr(pd.Series(sel_model.predict(g[SEL_FEATURES]), index=g.index), method="spearman"))

    results = {}
    for name, selector in [("ev_ranking", None), ("two_stage", Selector(sel_model, feats, int(cfg.strategy.selection.top_k)))]:
        risk = RiskManager(cfg)
        strat = Strategy(cfg, model, risk, AgentOverlay(cfg, None), NullLog(), markets, symbols)
        pf, curve = simulate(cfg, strat, feats, minutes, start, end, risk, NullLog(), bar, markets, selector=selector)
        tr = pd.DataFrame([t.__dict__ for t in pf.trades])
        m = metrics(tr, curve, pf.initial)
        results[name] = {k: m.get(k) for k in ("net_return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "profit_factor")}
    keep = (results["two_stage"]["net_return_pct"] > results["ev_ranking"]["net_return_pct"]
            and results["two_stage"]["max_drawdown_pct"] <= results["ev_ranking"]["max_drawdown_pct"] * 1.2)
    out = {"window": [start.isoformat(), end.isoformat()], "pairs": symbols, "top_k": cfg.strategy.selection.top_k,
           "rank_ic_mean": float(ic.mean()), "rank_ic_positive_share": float((ic > 0).mean()),
           "results": results, "decision": "KEEP two-stage" if keep else "REJECT two-stage (EV ranking wins)"}
    run_dir = Path(cfg.run.output_dir) / ("selection-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    L = ["# Selection model experiment", "", f"Held-out window {start.date()} .. {end.date()}, pairs: {', '.join(symbols)}, top-k {out['top_k']}", "",
         f"Rank IC (Spearman, predicted vs. realized relative 24h return): mean {out['rank_ic_mean']:+.3f}, "
         f"positive in {out['rank_ic_positive_share']:.0%} of 4h slots", "",
         "| variant | net return | max DD | trades | win rate | profit factor |", "|---|---|---|---|---|---|"]
    for k, r in results.items():
        L.append(f"| {k} | {r['net_return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | {r['trades']} | "
                 f"{(r['win_rate_pct'] or 0):.1f}% | {(r['profit_factor'] or 0):.2f} |")
    L += ["", f"**Decision: {out['decision']}**"]
    (run_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return run_dir
