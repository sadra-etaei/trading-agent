"""News monitor: RSS headlines (crypto + Persian/Iranian) scored by a GPT model.

Headlines have no usable history, so they never enter the ML model. They feed the agent
overlay, which can only veto, reduce, exit early or pause. All scores are logged.
"""
from __future__ import annotations

import hashlib
import logging
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import httpx

from engine.agent.llm import LLM, LLMBudgetError, LLMError

log = logging.getLogger(__name__)

SCORE_SYSTEM = """You score news headlines for a short-term (hours to 3 days) crypto spot trading bot
on Wallex, an Iranian exchange where prices are quoted in Iranian Toman (TMN).
Two things move TMN crypto prices: global crypto prices, and the USDT/TMN rate (rial weakness
raises every TMN price; rial strength lowers them). Headlines may be English or Persian.

For each headline return:
- id (copy it)
- score: -1.0 (very bearish for holding crypto in TMN) .. +1.0 (very bullish); 0 if irrelevant
- category: one of global_crypto, regulation, hack_or_exchange, macro_global, iran_macro, iran_politics, other
- assets: list of affected tickers (e.g. ["BTC"], ["ALL"] for market-wide, [] if none)
- severity: 0 (noise) .. 3 (market-moving shock)
Respond as JSON: {"items": [ ... ]}. Be calibrated: most headlines are noise (score near 0, severity 0)."""


def fetch_feed(url: str, timeout: float = 15) -> list[dict]:
    r = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    root = ET.fromstring(r.content)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        if not title:
            continue
        pub = it.findtext("pubDate")
        try:
            ts = parsedate_to_datetime(pub).timestamp() if pub else time.time()
        except (TypeError, ValueError):
            ts = time.time()
        items.append({"title": title, "source": url, "published": ts})
    return items


class NewsMonitor:
    def __init__(self, cfg, llm: LLM | None, logger=None):
        self.cfg = cfg.sentiment.news
        self.model = cfg.agent.screen_model
        self.llm = llm
        self.logger = logger
        self.items: dict[str, dict] = {}
        self.last_refresh = 0.0

    def refresh(self, force: bool = False) -> None:
        if not self.cfg.enabled:
            return
        if not force and time.time() - self.last_refresh < self.cfg.refresh_min * 60:
            return
        self.last_refresh = time.time()
        new = []
        for url in self.cfg.feeds:
            try:
                for it in fetch_feed(url)[: self.cfg.max_headlines]:
                    hid = hashlib.sha1(it["title"].encode()).hexdigest()[:12]
                    if hid not in self.items and time.time() - it["published"] < 48 * 3600:
                        it["id"] = hid
                        new.append(it)
            except Exception as e:  # a broken feed must never stop trading
                log.warning("feed %s failed: %s", url, e)
        if not new:
            return
        scores = self._score(new)
        for it in new:
            s = scores.get(it["id"], {})
            it.update(score=float(s.get("score", 0.0)), category=s.get("category", "other"),
                      assets=s.get("assets", []), severity=int(s.get("severity", 0)), scored=bool(s))
            self.items[it["id"]] = it
            if self.logger:
                self.logger.write("news", time.time(), headline=it)

    def _score(self, items: list[dict]) -> dict[str, dict]:
        if self.llm is None:
            return {}
        user = "\n".join(f'{it["id"]}: {it["title"]}' for it in items)
        try:
            res, _, _ = self.llm.json_chat(self.model, SCORE_SYSTEM, user)
            return {x["id"]: x for x in res.get("items", []) if "id" in x}
        except LLMBudgetError as e:
            log.warning("headline scoring off (budget/credit): %s", e)
            self.llm = None
            return {}
        except (LLMError, KeyError, TypeError) as e:
            log.warning("headline scoring failed: %s", e)
            return {}

    def digest(self, hours: float = 12, min_severity: int = 1) -> list[dict]:
        cutoff = time.time() - hours * 3600
        rel = [i for i in self.items.values() if i["published"] >= cutoff and i.get("severity", 0) >= min_severity]
        rel.sort(key=lambda i: (-i["severity"], -i["published"]))
        return [{k: i[k] for k in ("title", "score", "category", "assets", "severity")} |
                {"age_h": round((time.time() - i["published"]) / 3600, 1)} for i in rel[:15]]

    def market_score(self, hours: float = 12) -> float:
        d = self.digest(hours)
        if not d:
            return 0.0
        w = [x["severity"] for x in d]
        return sum(x["score"] * s for x, s in zip(d, w)) / sum(w)
