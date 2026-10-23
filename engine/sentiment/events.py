"""Event gate (PLAN.md §15 items 5 + 5b): a risk gate driven by market-moving posts and releases.

Sources (all free, polled from a background thread): Trump's Truth Social posts (trumpstruth.org),
Federal Reserve releases and speeches, Binance announcements (listings, delistings, news), crypto news
RSS and world news. A keyword filter keeps irrelevant posts from the LLM; everything is logged.

Scoring is cheap and framed in US dollars (the bot holds USDT when idle): gpt-6-luna scores every
item; gpt-6.1-sol scores Fed statements/minutes/chair speeches against the previous statement and
confirms any "sell everything" decision. The gate only ever REDUCES risk:
  - block new entries for `block_h` hours after a fresh, severe, negative, novel item;
  - sell held coins to USDT on a confirmed severity-3 shock (or a hack/delisting of a held coin);
  - Fed: no entries inside each calendar event's window (e.g. 2 h before an FOMC decision); after the
    release, a hawkish surprise blocks entries for hours, a dovish one only for 15 min (no chasing).
Positive news never triggers a buy. Every scored item goes to data/sentiment/events/<date>.jsonl, a
dataset for testing sentiment as a model feature after the challenge.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path

import httpx

from engine.agent.llm import LLMBudgetError, LLMError
from engine.sentiment import fed

log = logging.getLogger(__name__)
UA = {"User-Agent": "Mozilla/5.0"}
BINANCE_CMS = "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query"

SCORE_SYSTEM = """You score market-moving items for a crypto spot trading bot. The bot holds USDT when idle and buys
coins against USDT (BTC, ETH, XRP, SOL, PEPE, BNB, XAUT); profit is measured in US dollars.
Items come from Donald Trump's Truth Social posts, Federal Reserve releases, Binance announcements, crypto news
and world news, in English or Persian.

For each item return:
- id (copy it)
- score: -1.0 (very bearish for crypto prices in USD over the next hours) .. +1.0 (very bullish); 0 if irrelevant
- severity: 0 (noise) .. 3 (market-moving shock, e.g. new tariffs on China, war escalation, exchange hack)
- category: one of tariff_trade, geopolitics, fed_monetary, us_macro, crypto_policy, hack_exchange,
  listing_delisting, global_crypto, iran, other
- assets: affected tickers (e.g. ["BTC"]), ["ALL"] for market-wide, [] if none
- novelty: 0 (repeat or already-known news) .. 1 (genuinely new information)
Respond as JSON: {"items": [ ... ]}. Be calibrated: most items are noise (score near 0, severity 0)."""

TMN_NOTE = """
The bot currently trades TOMAN-quoted pairs and holds Toman when idle: besides global crypto prices, the USDT/TMN
rate moves every price (rial weakness raises Toman prices, rial strength or a USDT/TMN drop lowers them).
Score the effect on crypto prices IN TOMAN; Iranian economic and political news belongs in category iran."""

CONFIRM_SYSTEM = """A risk gate of a crypto spot bot (profit in USD, idle cash in USDT) wants to SELL coins to USDT now
because of the item below. Confirm only if it is a genuine, new shock likely to push crypto prices down
over the next hours; reject repeats, opinions, old news and anything already priced in.
Return JSON: {"confirm": true|false, "reason": "<one sentence>"}."""


def _strip(html: str) -> str:
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html or ""))).strip()


def _ts(pub: str | None) -> float:
    try:
        return parsedate_to_datetime(pub).timestamp() if pub else time.time()
    except (TypeError, ValueError):
        return time.time()


def fetch_rss(url: str, timeout: float = 20) -> list[dict]:
    r = httpx.get(url, timeout=timeout, follow_redirects=True, headers=UA)
    r.raise_for_status()
    out = []
    for it in ET.fromstring(r.content).iter("item"):
        title = _strip(it.findtext("title") or "")
        if title:
            out.append({"title": title[:500], "text": _strip(it.findtext("description") or "")[:600],
                        "link": (it.findtext("link") or "").strip(), "published": _ts(it.findtext("pubDate"))})
    return out


def fetch_binance(catalogs: list[int], timeout: float = 20) -> list[dict]:
    r = httpx.get(BINANCE_CMS, params={"type": 1, "pageNo": 1, "pageSize": 10}, headers=UA, timeout=timeout)
    r.raise_for_status()
    out = []
    for c in r.json()["data"]["catalogs"]:
        if c["catalogId"] in catalogs:
            for a in c["articles"]:
                out.append({"title": f'[Binance {c["catalogName"]}] {a["title"]}', "text": "",
                            "link": f'https://www.binance.com/en/support/announcement/{a["code"]}',
                            "published": a["releaseDate"] / 1000})
    return out


def base_of(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol.removesuffix("TMN")


class EventGate:
    def __init__(self, cfg, llm=None, dlog=None, symbols=()):
        self.ev = cfg.sentiment.events
        self.rules = self.ev.rules
        self.fed_cfg = self.ev.fed
        self.screen_model, self.decision_model = cfg.agent.screen_model, cfg.agent.decision_model
        self.llm, self.dlog = llm, dlog
        self.bases = {base_of(s): s for s in symbols}
        self.cal = fed.FedCalendar(self.fed_cfg, cfg.data.dir)
        self.log_dir = Path(cfg.data.dir) / "sentiment" / "events"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.items: dict[str, dict] = {}
        self.block_until, self.block_reason = 0.0, ""
        self.pending_exits: set[str] = set()      # base tickers, or "ALL"
        self.last_poll: dict[str, float] = {}
        self.prev_statement: str | None = None
        self.lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        toman = all(s.endswith("TMN") for s in cfg.universe.core)
        self.score_system = SCORE_SYSTEM + (TMN_NOTE if toman else "")
        self.kw = re.compile("|".join(re.escape(k) for k in self.ev.keywords), re.I)

    # ------------------------------------------------------------ public API (main thread)
    def start(self) -> None:
        if not self.ev.enabled or self._thread:
            return
        self.cal.refresh(force=True)
        self._thread = threading.Thread(target=self._loop, name="event-gate", daemon=True)
        self._thread.start()
        log.info("event gate started: %d sources, %d Fed calendar events", len(self.ev.sources), len(self.cal.events))

    def stop(self) -> None:
        self._stop.set()

    def refresh(self, force: bool = False) -> None:
        """Synchronous poll when no background thread runs (the runner calls this every bar)."""
        if self._thread is None and self.ev.enabled:
            self.poll(time.time(), force=force)

    def entries_blocked(self, now: float | None = None) -> tuple[bool, str]:
        now = time.time() if now is None else now
        with self.lock:
            if now < self.block_until:
                return True, self.block_reason
        e = self.cal.active(datetime.fromtimestamp(now, timezone.utc))
        if e:
            return True, f"Fed calendar window: {e['title']} at {e['time']:%Y-%m-%d %H:%M} UTC"
        return False, ""

    def take_exits(self, held_symbols) -> list[str]:
        with self.lock:
            if not self.pending_exits:
                return []
            out = [s for s in held_symbols if "ALL" in self.pending_exits or base_of(s) in self.pending_exits]
            self.pending_exits.clear()
            return out

    def status(self) -> dict:
        blocked, reason = self.entries_blocked()
        now = datetime.now(timezone.utc)
        return {"entries_blocked": blocked, "reason": reason,
                "fed_upcoming": [f"{e['title']} {e['time']:%m-%d %H:%M}Z" for e in self.cal.upcoming(now, 72)]}

    def digest(self, hours: float = 12, min_severity: int = 1) -> list[dict]:
        cutoff = time.time() - hours * 3600
        with self.lock:
            rel = [i for i in self.items.values() if i.get("scored") and i["published"] >= cutoff
                   and i.get("severity", 0) >= min_severity]
        rel.sort(key=lambda i: (-i["severity"], -i["published"]))
        return [{k: i.get(k) for k in ("title", "score", "category", "assets", "severity", "source")} |
                {"age_h": round((time.time() - i["published"]) / 3600, 1)} for i in rel[:15]]

    def market_score(self, hours: float = 12) -> float:
        d = self.digest(hours)
        w = [x["severity"] for x in d]
        return sum(x["score"] * s for x, s in zip(d, w)) / sum(w) if w else 0.0

    # ------------------------------------------------------------ polling (background thread)
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll(time.time())
            except Exception as e:  # the gate must never die silently
                log.exception("event gate poll error: %s", e)
            self._stop.wait(5)

    def _due(self, src: dict, now: float) -> bool:
        every = src["poll_s"]
        if src["kind"] == "fed":
            nowdt = datetime.fromtimestamp(now, timezone.utc)
            if any(-300 <= (nowdt - e["time"]).total_seconds() <= 1800 for e in self.cal.events):
                every = self.fed_cfg["fast_poll_s"]  # an event is due: poll the Fed feed every few seconds
        return now - self.last_poll.get(src["name"], 0) >= every

    def poll(self, now: float, force: bool = False) -> None:
        self.cal.refresh()
        due = [s for s in self.ev.sources if force or self._due(s, now)]
        for s in due:
            self.last_poll[s["name"]] = now

        def fetch(src):
            try:
                return src, fetch_binance(src["catalogs"]) if src["kind"] == "binance" else fetch_rss(src["url"])
            except Exception as e:  # a broken feed must never stop trading
                log.warning("event source %s failed: %s", src["name"], e)
                return src, []

        new = []
        with ThreadPoolExecutor(6) as pool:
            fetched = list(pool.map(fetch, due))
        for src, got in fetched:
            for it in got:
                iid = hashlib.sha1((src["name"] + (it["link"] or it["title"])).encode()).hexdigest()[:12]
                if iid in self.items:
                    continue
                it.update(id=iid, source=src["name"], kind=src["kind"], seen=now, scored=False,
                          relevant=not src.get("prefilter") or bool(self.kw.search(it["title"] + " " + it["text"])))
                new.append(it)
        if not new:
            return
        with self.lock:
            for it in new:
                self.items[it["id"]] = it
        self._score(new, now)
        for it in new:
            self._write(it)
            self._act(it, now)

    # ------------------------------------------------------------ scoring
    def _is_fed_key(self, it: dict) -> bool:
        t = it["title"].lower()
        return it["kind"] == "fed" and ("fomc statement" in t or "minutes of the federal open market committee" in t
                                        or t.startswith(self.fed_cfg["chair"].lower()))

    def _score(self, items: list[dict], now: float) -> None:
        if self.llm is None:
            return
        fresh = lambda it: now - it["published"] <= self.rules["act_max_age_min"] * 60  # noqa: E731
        for it in items:
            if self._is_fed_key(it) and fresh(it):
                try:
                    text = fed.page_text(it["link"]) if it["link"] else it["text"]
                    it.update(fed=fed.score_release(self.llm, self.decision_model, it["title"], text, self.prev_statement),
                              scored=True)
                    if "fomc statement" in it["title"].lower():
                        self.prev_statement = text
                except (LLMError, httpx.HTTPError, ValueError) as e:
                    log.warning("Fed scoring failed: %s", e)
        todo = [it for it in items if it["relevant"] and "fed" not in it]
        for k in range(0, len(todo), 30):
            chunk = todo[k:k + 30]
            user = "\n".join(f'{it["id"]} | {it["source"]} | {it["title"]} -- {it["text"][:300]}' for it in chunk)
            try:
                res, _, _ = self.llm.json_chat(self.screen_model, self.score_system, user)
            except LLMBudgetError as e:
                log.warning("event scoring off (budget/credit): %s", e)
                self.llm = None
                return
            except (LLMError, KeyError, TypeError) as e:
                log.warning("event scoring failed: %s", e)
                continue
            got = {x.get("id"): x for x in res.get("items", []) if isinstance(x, dict)}
            for it in chunk:
                s = got.get(it["id"])
                if s:
                    it.update(score=_num(s.get("score")), severity=int(_num(s.get("severity"))),
                              category=str(s.get("category", "other")), assets=list(s.get("assets") or []),
                              novelty=_num(s.get("novelty")), scored=True)

    def _confirm(self, it: dict) -> tuple[bool, str]:
        if self.llm is None:
            return False, "no LLM"
        try:
            res, _, _ = self.llm.json_chat(self.decision_model, CONFIRM_SYSTEM,
                                           json.dumps({k: it.get(k) for k in ("source", "title", "text", "score", "severity",
                                                                              "category", "assets")}, ensure_ascii=False))
            return bool(res.get("confirm")), str(res.get("reason", ""))[:300]
        except LLMError as e:
            return False, f"confirm failed: {e}"

    # ------------------------------------------------------------ rules (only ever reduce risk)
    def _block(self, until: float, reason: str, it: dict) -> None:
        with self.lock:
            if until > self.block_until:
                self.block_until, self.block_reason = until, reason
        self._event("event_gate", action="block_entries", until=datetime.fromtimestamp(until, timezone.utc).isoformat(),
                    reason=reason, item=it["id"], title=it["title"][:200])

    def _derisk(self, bases: set[str], reason: str, it: dict) -> None:
        with self.lock:
            self.pending_exits |= bases
        self._event("event_gate", action="sell_to_usdt", assets=sorted(bases), reason=reason, item=it["id"],
                    title=it["title"][:200])

    def _act(self, it: dict, now: float) -> None:
        if not it.get("scored") or now - it["published"] > self.rules["act_max_age_min"] * 60:
            return  # old items are logged for the dataset, never acted on
        t0 = min(it["published"], now)
        if "fed" in it:
            f, c = it["fed"], self.fed_cfg
            if f["stance"] <= -1 and f["surprise"] >= 0.5:
                self._block(t0 + c["hawkish_block_h"] * 3600, f"hawkish Fed surprise: {f['key_change']}", it)
                if f["stance"] <= -2:
                    self._derisk({"ALL"}, f"strongly hawkish Fed surprise: {f['key_change']}", it)
            elif f["stance"] >= 1:
                self._block(t0 + c["dovish_wait_min"] * 60, "dovish Fed release: wait, do not chase the first spike", it)
            else:
                self._block(t0 + c["neutral_block_min"] * 60, "Fed release as expected: short pause", it)
            return
        r = self.rules
        bad = (it.get("severity", 0) >= r["min_severity"] and it.get("score", 0) <= r["max_score"]
               and it.get("novelty", 0) >= r["min_novelty"] and it.get("category") in r["risk_categories"])
        if not bad:
            return
        self._block(t0 + r["block_h"] * 3600, f'{it["source"]}: {it["title"][:160]}', it)
        assets = {a.upper() for a in it.get("assets", [])}
        held_hit = assets & set(self.bases)
        shock = it.get("severity", 0) >= r["derisk_severity"] and it.get("score", 0) <= r["derisk_score"]
        coin_hit = it.get("category") in ("hack_exchange", "listing_delisting") and held_hit
        if shock or coin_hit:
            ok, why = self._confirm(it)
            self._event("event_gate_confirm", item=it["id"], confirm=ok, reason=why)
            if ok:
                self._derisk({"ALL"} if (shock and (not assets or "ALL" in assets)) else (held_hit or {"ALL"}), why, it)

    # ------------------------------------------------------------ logging
    def _event(self, kind: str, **fields) -> None:
        log.info("%s %s", kind, fields)
        if self.dlog:
            self.dlog.write(kind, datetime.now(timezone.utc), **fields)

    def _write(self, it: dict) -> None:
        day = datetime.fromtimestamp(it["seen"], timezone.utc).strftime("%Y-%m-%d")
        with open(self.log_dir / f"{day}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(it, ensure_ascii=False, default=str) + "\n")
        if self.dlog and it.get("scored") and ("fed" in it or it.get("severity", 0) >= 1):
            self.dlog.write("event", datetime.now(timezone.utc), item={k: it.get(k) for k in (
                "id", "source", "title", "published", "score", "severity", "category", "assets", "novelty", "fed")})


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0
