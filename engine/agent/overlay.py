"""GPT overlay agent (PLAN.md §15 item 6).

Role: the ML entry model proposes trades; the agent reviews them with context the model cannot see
(news and events, unusual market state). Design follows what the published agent ablations support
(FinCon, CryptoTrade; reports/Trading agent ideas review.md): hard risk rules stay in code, the LLM
only filters, it sees position and P&L facts, and it reflects on its own closed trades.
  - ONE batched call per bar reviews every candidate (cheap, consistent across candidates);
  - approve by default; a veto needs a reason code from a fixed list, otherwise it counts as approve
    (LLMs are too cautious in bull markets, which would kill trade count);
  - every approved trade gets an exit plan: take-profit and stop come from the model, the agent adds
    a one-sentence invalidation condition; the hourly exit review may close a position early only
    when that condition is met, and must say what evidence shows it;
  - every `reflect_every_h` hours it turns recent closed trades into 3 short lessons that are added to
    later prompts as hints (never as rules);
  - it can never open a trade on its own or enlarge one; the risk manager runs after it.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from engine.agent.llm import LLM, LLMBudgetError, LLMError

MAX_CONSECUTIVE_ERRORS = 3
VETO_CODES = ("major_negative_news", "market_shock", "exchange_or_regulatory_risk", "illiquid_or_stale_market",
              "extreme_move_already_happened")

log = logging.getLogger(__name__)

MARKET_TEXT = {
    "USDT": ("Wallex (Iranian exchange) USDT-quoted pairs, long-only spot. Idle cash is held in USDT and profit is "
             "measured in US dollars, so only global crypto prices matter (the rial does not)."),
    "TMN": ("Wallex (Iranian exchange) Toman-quoted pairs, long-only spot. Rial moves change all TMN prices "
            "(rial weakness raises them)."),
}

REVIEW_SYSTEM = """You are the risk-overlay agent of an automated crypto trading bot: {market}
A calibrated ML model (price history, Binance order flow and premium are already in it) proposes long trades.
Judge only what statistics cannot see: news and events, exchange or regulatory risk, abnormal market states.

Rules:
- Default to approve. Do not second-guess the model on ordinary technical grounds.
- veto only with one reason_code from: {codes}, and a concrete stated reason.
- reduce (size_multiplier 0.25-0.75) when risk is elevated but not clearly negative.
- risk_off=true only for market-wide shocks; it pauses all new entries for 4 hours.
- For every approved or reduced trade, write `invalidation`: one sentence naming an observable condition
  (news, event, Binance move) under which the trade idea is wrong and the position should be closed early.
- `lessons` (if present) are hints from the bot's own past trades; they never override these rules.
Respond with JSON only:
{{"reviews": [{{"symbol": "...", "action": "approve"|"reduce"|"veto", "size_multiplier": 0.0-1.0,
  "reason_code": "<code or empty>", "invalidation": "<one sentence>", "confidence": 0.0-1.0,
  "reasoning": "<= 2 sentences, concrete"}}], "risk_off": bool}}"""

EXIT_SYSTEM = """You are the risk-overlay agent of a long-only crypto spot bot: {market}
Each open position has an `invalidation` condition written at entry. Close a position early ONLY if its
invalidation condition is now met by the evidence given (headlines, market data); quote the evidence.
Take-profit, stop-loss and time exits are handled by code: never exit just because of a small loss or profit.
Respond with JSON only: {{"exit": [{{"symbol": "...", "evidence": "<what shows the condition is met>"}}],
 "reasoning": "<= 2 sentences"}}"""

REFLECT_SYSTEM = """You review the closed trades of a crypto trading bot ({market}) together with what its
risk-overlay agent said when approving them. Write exactly 3 short lessons (each <= 60 words) that would help
the agent review FUTURE trades better: when its approvals, reductions or invalidation conditions helped or
hurt. Be specific and evidence-based; no generic trading advice.
Respond with JSON only: {"lessons": ["...", "...", "..."]}"""

DEFAULT = {"action": "approve", "size_multiplier": 1.0, "risk_off": False, "confidence": 0.0, "reason_code": "",
           "invalidation": ""}


class AgentOverlay:
    def __init__(self, cfg, llm: LLM | None, news=None):
        self.cfg = cfg.agent
        self.enabled = bool(cfg.agent.enabled) and llm is not None
        self.llm = llm
        self.news = news
        quote = "USDT" if all(s.endswith("USDT") for s in cfg.universe.core) else "TMN"
        self.market = MARKET_TEXT[quote]
        self.review_system = REVIEW_SYSTEM.format(market=self.market, codes=", ".join(VETO_CODES))
        self.exit_system = EXIT_SYSTEM.format(market=self.market)
        self.reflect_system = REFLECT_SYSTEM.replace("{market}", self.market)
        self.risk_off_until: pd.Timestamp | None = None
        self.errors = 0
        self.disabled_reason = ""
        self.lessons_path = Path(cfg.agent.cache_dir) / "lessons.json"
        self.lessons, self.last_reflect = self._load_lessons()

    def _disable(self, reason: str) -> None:
        """Agent off for the rest of the run: trading continues on ML + risk manager alone."""
        self.enabled = False
        self.disabled_reason = reason
        log.warning("agent disabled: %s -> continuing with ML + risk rules only", reason)

    def is_risk_off(self, ts: pd.Timestamp) -> bool:
        return self.risk_off_until is not None and ts < self.risk_off_until

    # ------------------------------------------------------------ entry review
    def review(self, ts: pd.Timestamp, candidate: dict, portfolio: dict, market: dict) -> dict:
        return self.review_batch(ts, [candidate], portfolio, market)[candidate["symbol"]]

    def review_batch(self, ts: pd.Timestamp, candidates: list[dict], portfolio: dict, market: dict) -> dict[str, dict]:
        syms = [c["symbol"] for c in candidates]
        if not candidates:
            return {}
        if not self.enabled:
            why = f"agent disabled{': ' + self.disabled_reason if self.disabled_reason else ''}"
            return {s: {**DEFAULT, "reasoning": why, "source": "disabled"} for s in syms}
        payload = {"time_utc": ts.isoformat(), "candidates": candidates, "portfolio": portfolio, "market": market,
                   "headlines": self.news.digest() if self.news else [], "lessons": self.lessons}
        try:
            res, key, cached = self.llm.json_chat(self.cfg.decision_model, self.review_system,
                                                  json.dumps(payload, ensure_ascii=False, default=str))
            got = {r.get("symbol"): r for r in res.get("reviews", []) if isinstance(r, dict)}
            out = {s: self._validate(got.get(s, {})) | {"source": "cache" if cached else "llm", "cache_key": key,
                                                         "model": self.cfg.decision_model, "batch": syms} for s in syms}
            self.errors = 0
            if res.get("risk_off"):
                self.risk_off_until = ts + pd.Timedelta(hours=4)
                for o in out.values():
                    o["risk_off"] = True
            return out
        except LLMBudgetError as e:
            self._disable(str(e)[:200])
            return {s: {**DEFAULT, "reasoning": f"agent disabled (budget/credit): {e}"[:300], "source": "disabled"} for s in syms}
        except (LLMError, KeyError, TypeError, ValueError) as e:
            self.errors += 1
            if self.errors >= MAX_CONSECUTIVE_ERRORS:
                self._disable(f"{self.errors} consecutive errors, last: {e}"[:200])
                return {s: {**DEFAULT, "reasoning": f"agent disabled after repeated errors: {e}"[:300], "source": "disabled"}
                        for s in syms}
            log.warning("agent review failed: %s", e)
            if self.cfg.on_error == "approve":
                return {s: {**DEFAULT, "reasoning": f"agent error -> approve: {e}", "source": "error"} for s in syms}
            return {s: {**DEFAULT, "action": "veto", "size_multiplier": 0.0, "reason_code": "agent_error",
                        "reasoning": f"agent error -> veto: {e}", "source": "error"} for s in syms}

    @staticmethod
    def _validate(res: dict) -> dict:
        action = res.get("action", "approve")
        if action not in ("approve", "reduce", "veto"):
            action = "approve"
        code = str(res.get("reason_code") or "")
        note = ""
        if action == "veto" and code not in VETO_CODES:
            action, note = "approve", f"veto without a valid reason code ({code or 'none'}) -> approve"
        try:
            mult = max(0.0, min(1.0, float(res.get("size_multiplier", 1.0))))
        except (TypeError, ValueError):
            mult = 1.0
        if action == "veto":
            mult = 0.0
        elif action == "approve":
            mult = 1.0
        elif mult == 0:
            mult = 0.5
        try:
            conf = max(0.0, min(1.0, float(res.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        out = {"action": action, "size_multiplier": mult, "risk_off": False, "confidence": conf,
               "reason_code": code if action == "veto" else "", "invalidation": str(res.get("invalidation", ""))[:300],
               "reasoning": str(res.get("reasoning", ""))[:500]}
        if note:
            out["note"] = note
        return out

    # ------------------------------------------------------------ exit review
    def review_exits(self, ts: pd.Timestamp, positions: list[dict]) -> dict:
        """`positions`: symbol, entry, now, pnl_pct, invalidation (written at entry)."""
        if not self.enabled or not positions or not self.news:
            return {"exit": [], "reasoning": "skipped"}
        heads = self.news.digest(hours=6, min_severity=2)
        if not heads:
            return {"exit": [], "reasoning": "no significant news"}
        user = json.dumps({"time_utc": ts.isoformat(), "positions": positions, "headlines": heads}, ensure_ascii=False)
        try:
            res, key, cached = self.llm.json_chat(self.cfg.decision_model, self.exit_system, user)
            syms = {p["symbol"] for p in positions}
            ex = [e for e in res.get("exit", []) if isinstance(e, dict) and e.get("symbol") in syms and e.get("evidence")]
            return {"exit": [e["symbol"] for e in ex], "evidence": {e["symbol"]: str(e["evidence"])[:300] for e in ex},
                    "reasoning": str(res.get("reasoning", ""))[:500], "cache_key": key}
        except (LLMError, KeyError, TypeError) as e:
            log.warning("agent exit review failed: %s", e)
            return {"exit": [], "reasoning": f"error: {e}"}

    # ------------------------------------------------------------ reflection
    def _load_lessons(self) -> tuple[list[str], float]:
        try:
            d = json.loads(self.lessons_path.read_text(encoding="utf-8"))
            return list(d.get("lessons", []))[:3], float(d.get("at", 0))
        except (OSError, ValueError):
            return [], 0.0

    def maybe_reflect(self, ts: pd.Timestamp, trades: list[dict], reviews: dict[str, dict]) -> dict | None:
        """Every `reflect_every_h` hours, if trades closed since the last reflection: 3 new lessons.
        `reviews`: entry decision_id -> the agent review recorded for that entry."""
        every = float(self.cfg.get("reflect_every_h") or 0)
        if not self.enabled or not every or time.time() - self.last_reflect < every * 3600:
            return None
        recent = [t for t in trades if pd.Timestamp(t["exit_ts"]).timestamp() > self.last_reflect]
        if not recent:
            return None
        rows = [{k: t.get(k) for k in ("symbol", "entry_ts", "exit_ts", "ret_pct", "exit_reason", "hold_hours")}
                | {"agent": {k: (reviews.get(t["decision_id"]) or {}).get(k) for k in ("action", "invalidation", "reasoning")}}
                for t in recent[-20:]]
        try:
            res, key, _ = self.llm.json_chat(self.cfg.decision_model, self.reflect_system,
                                             json.dumps({"previous_lessons": self.lessons, "closed_trades": rows},
                                                        ensure_ascii=False, default=str))
        except LLMError as e:
            log.warning("reflection failed: %s", e)
            return None
        self.lessons = [str(x)[:400] for x in res.get("lessons", [])][:3]
        self.last_reflect = time.time()
        self.lessons_path.write_text(json.dumps({"lessons": self.lessons, "at": self.last_reflect}, ensure_ascii=False,
                                                indent=1), encoding="utf-8")
        return {"lessons": self.lessons, "trades": len(rows), "cache_key": key}
