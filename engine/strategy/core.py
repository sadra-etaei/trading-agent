"""Strategy core shared by backtest, shadow and live.

At each bar close:  features -> entry model p_win -> EV ranking -> GPT overlay -> risk sizing.
Every step is written to the decision log.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.config import slippage
from engine.agent.overlay import AgentOverlay
from engine.decision_log import DecisionLog
from engine.features.build import REGIME_NAMES
from engine.model.labels import barrier_pcts
from engine.model.predict import EntryModel, expected_value
from engine.risk.manager import RiskManager
from engine.risk.portfolio import Portfolio
from engine.strategy import trend

EV_FULL_SIZE_MARGIN = 0.01  # EV this far above the threshold gets full size
ENTRY_MIN_MULT = 1.5        # entries >= 1.5 x the minimum order, so the exit stays sellable after a loss


@dataclass
class Intent:
    symbol: str
    notional: float
    p_win: float
    ev: float
    tp_pct: float
    sl_pct: float
    atr_pct: float
    decision_id: str
    invalidation: str = ""


class Strategy:
    def __init__(self, cfg, model: EntryModel, risk: RiskManager, agent: AgentOverlay, dlog: DecisionLog,
                 markets: dict[str, dict], symbols: list[str]):
        self.cfg = cfg
        self.model = model
        self.risk = risk
        self.agent = agent
        self.log = dlog
        self.markets = markets
        self.symbols = [s for s in symbols if s in model.bundle.get("symbols", symbols)]
        s = cfg.strategy
        tuned = model.thresholds.get("min_ev_pct") if s.use_tuned_threshold else None
        self.ev_thr = float(tuned if tuned is not None else s.min_ev_pct)
        self.min_p = float(s.min_p_win)
        self.cost = 2 * cfg.costs.fee_rate  # + per-pair slippage on both sides (config.slippage)
        if (cfg.get("cash") or {}).get("idle") == "usdt":  # USDT -> Toman before the buy, Toman -> USDT after the sell
            self.cost += 2 * float(cfg.cash.get("conv_fee", 0.0031))
            self.symbols = [s for s in self.symbols if s != "USDTTMN"]  # USDT is the idle asset, not a trade
        self.allowed: set[str] | None = None  # optional selection-model filter
        self.trend_rule = s.get("trend_filter", "none") or "none"

    def score(self, rows: pd.DataFrame) -> pd.DataFrame:
        rows = rows.copy()
        rows["p_win"] = self.model.proba(rows)
        tp, sl = barrier_pcts(rows["atr_pct"].to_numpy(float), self.cfg.labels)
        rows["tp_pct"], rows["sl_pct"] = tp, sl
        cost = self.cost + 2 * np.array([slippage(self.cfg, s) for s in rows.index])
        rows["ev"] = expected_value(rows["p_win"].to_numpy(), tp, sl, cost)
        return rows

    def decide_entries(self, ts: pd.Timestamp, rows: pd.DataFrame, pf: Portfolio, prices: dict[str, float],
                       market: dict) -> list[Intent]:
        """`rows`: one feature row per symbol for the bar that just closed (index = symbol)."""
        rows = rows.loc[[s for s in rows.index if s in self.symbols]]
        if rows.empty:
            return []
        eq = pf.equity(prices)
        scored = self.score(rows)
        scan = {s: {"p": float(r.p_win), "ev": float(r.ev), "regime": REGIME_NAMES.get(int(r.regime), "?")}
                for s, r in scored.iterrows()}

        blocked = self.risk.entries_blocked(ts, pf, eq)
        if blocked is None and self.agent.is_risk_off(ts):
            blocked = f"agent risk-off until {self.agent.risk_off_until}"
        self.log.write("scan", ts, equity=eq, ev_threshold=self.ev_thr, blocked=blocked, scores=scan,
                       positions=list(pf.positions))
        if blocked:
            return []

        cand = scored[(scored["ev"] > self.ev_thr) & (scored["p_win"] >= self.min_p) & (scored["stale"] == 0)]
        if self.allowed is not None:
            cand = cand[cand.index.isin(self.allowed)]
        if self.trend_rule != "none" and len(cand):
            ok = trend.allowed(cand, self.trend_rule)
            for sym, r in cand[~ok].iterrows():  # logged so the filter's effect is auditable
                self.log.write("trend_block", ts, symbol=sym, rule=self.trend_rule, condition=trend.describe(self.trend_rule),
                               value=r.get(trend.RULES[self.trend_rule][0]),
                               ml={"model_version": self.model.version, "p_win": float(r.p_win), "ev": float(r.ev)})
            cand = cand[ok]
        cand = cand[~cand.index.isin(list(pf.positions))].sort_values("ev", ascending=False)
        cand = cand.head(int(self.cfg.strategy.max_new_entries_per_bar))

        intents: list[Intent] = []
        pending = 0.0
        pf_summary = {
            "equity_tmn": round(eq),
            "cash_tmn": round(pf.available()),
            "open_positions": {s: {"entry": p.entry_price, "now": prices.get(s)} for s, p in pf.positions.items()},
            "drawdown_from_peak": round(1 - eq / pf.peak_equity, 4) if pf.peak_equity else 0.0,
            "daily_pnl": round(eq / pf.day_start_equity - 1, 4) if pf.day_start_equity else 0.0,
        }
        cands = {}
        for sym, r in cand.iterrows():
            cands[sym] = {
                "symbol": sym,
                "p_win": round(float(r.p_win), 3),
                "expected_value_pct": round(float(r.ev) * 100, 3),
                "take_profit_pct": round(float(r.tp_pct) * 100, 2),
                "stop_loss_pct": round(float(r.sl_pct) * 100, 2),
                "regime": REGIME_NAMES.get(int(r.regime), "?"),
                "ret_24h_pct": round(float(r.ret_24) * 100, 2),
                "rsi": round(float(r.rsi), 1),
                "atr_pct": round(float(r.atr_pct) * 100, 2),
                "rule_signals": {"pullback": int(r.sig_pullback), "mean_reversion": int(r.sig_meanrev)},
            }
        reviews = self.agent.review_batch(ts, list(cands.values()), pf_summary, market)  # one call per bar
        for sym, r in cand.iterrows():
            did = self.log.new_id()
            feats = {f: r[f] for f in self.model.features}
            agent = reviews[sym]
            base_scale = 0.5 + 0.5 * float(np.clip((r.ev - self.ev_thr) / EV_FULL_SIZE_MARGIN, 0, 1))
            record = {
                "symbol": sym,
                "ml": {"model_version": self.model.version, "p_win": float(r.p_win), "ev": float(r.ev),
                       "ev_threshold": self.ev_thr, "tp_pct": float(r.tp_pct), "sl_pct": float(r.sl_pct)},
                "features": feats,
                "agent": agent,
                "market": market,
            }
            if agent["action"] == "veto":
                self.log.write("veto", ts, decision_id=did, **record)
                continue
            size = self.risk.size(ts, sym, float(r.sl_pct), base_scale * agent["size_multiplier"], pf, prices,
                                  ENTRY_MIN_MULT * self.markets.get(sym, {}).get("min_notional", 100_000), pending, len(intents))
            record["risk"] = {"allowed": size.allowed, "notional_tmn": size.notional, "checks": size.checks,
                              "reason": size.reason, "size_scale": base_scale * agent["size_multiplier"]}
            if not size.allowed:
                self.log.write("risk_block", ts, decision_id=did, **record)
                continue
            # Entry decision uses the model's own target (its probability is calibrated for it); the exit
            # target can be set farther (strategy.exit_tp_atr) to let winners run, with the trailing stop.
            tp_exit = float(r.tp_pct)
            if self.cfg.strategy.get("exit_tp_atr"):
                tp_exit = max(float(self.cfg.strategy.exit_tp_atr) * float(r.atr_pct), float(self.cfg.labels.min_tp_pct))
                record["exit_tp_pct"] = tp_exit
            # hard limit (PLAN.md §15 item 4): no single position may lose more than risk.max_stop_pct
            sl_exit = min(float(r.sl_pct), float(self.cfg.risk.get("max_stop_pct") or 1.0))
            if sl_exit < float(r.sl_pct):
                record["exit_sl_pct"] = sl_exit
            self.log.write("entry_signal", ts, decision_id=did, **record)
            intents.append(Intent(sym, size.notional, float(r.p_win), float(r.ev), tp_exit, sl_exit,
                                  float(r.atr_pct), did, agent.get("invalidation", "")))
            pending += size.notional
        return intents
