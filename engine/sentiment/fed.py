"""Federal Reserve module (PLAN.md §15 item 5b): the event calendar and statement scoring.

The calendar is parsed from federalreserve.gov's FOMC page (decision = 2nd meeting day, 14:00
New York time) and merged with `data/fed_calendar.yaml`, a manual file for anything the page does
not list (chair speeches, CPI, payrolls). The parsed list is cached in data/fed_calendar_auto.json so a
failed refresh never leaves the bot without a calendar.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import yaml

log = logging.getLogger(__name__)
FOMC_PAGE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
STATEMENT_URL = "https://www.federalreserve.gov/newsevents/pressreleases/monetary{d}a.htm"
NY = ZoneInfo("America/New_York")
MONTHS = {m: i for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August",
                                      "September", "October", "November", "December"], 1)}
UA = {"User-Agent": "Mozilla/5.0"}

FED_SYSTEM = """You are a Federal Reserve analyst for a crypto spot trading bot whose profit is measured in US dollars.
You get a new Fed release (FOMC statement, minutes, or a speech by the Fed chair) and, when available, the previous
FOMC statement. Judge what is NEW relative to the previous statement and to what markets expected.
Return JSON:
- stance: -2 (much more hawkish than expected) .. +2 (much more dovish than expected); 0 = as expected
- surprise: 0 (fully expected) .. 1 (large surprise)
- key_change: one sentence on what changed
- crypto_effect: -1 (bearish for crypto in USD over the next hours) .. +1 (bullish)
Be calibrated: most releases are close to expectations (stance 0, surprise below 0.3)."""


def parse_fomc_page(html: str) -> list[dict]:
    """FOMC decision times (UTC) for every year listed on the Fed's calendar page."""
    out = []
    for year in sorted(set(re.findall(r"(\d{4}) FOMC Meetings", html))):
        i = html.find(f"{year} FOMC Meetings")
        j = html.find("FOMC Meetings", i + 20)
        seg = html[i: j if j > 0 else i + 40000]
        for month, days in re.findall(r"fomc-meeting__month[^>]*>\s*<strong>([^<]+)</strong>.*?fomc-meeting__date[^>]*>([^<]+)<",
                                      seg, re.S):
            m = month.split("/")[-1].strip()
            d = re.findall(r"\d+", days)
            if m not in MONTHS or not d:
                continue
            # "April/May 30-1": the decision day is in the second month
            local = datetime(int(year), MONTHS[m], int(d[-1]), 14, 0, tzinfo=NY)
            out.append({"time": local.astimezone(timezone.utc).isoformat(), "type": "fomc",
                        "title": f"FOMC decision {year}-{MONTHS[m]:02d}-{int(d[-1]):02d}",
                        "projections": "*" in days})
    return out


def statement_dates(html: str) -> list[str]:
    return sorted(set(re.findall(r"monetary(\d{8})a\.htm", html)))


class FedCalendar:
    def __init__(self, cfg_fed, data_dir: str):
        self.cfg = cfg_fed
        self.manual = Path(cfg_fed["calendar_file"])
        self.auto = Path(data_dir) / "fed_calendar_auto.json"
        self.events: list[dict] = []
        self.refreshed = 0.0

    def refresh(self, force: bool = False) -> None:
        if not force and time.time() - self.refreshed < 12 * 3600:
            return
        self.refreshed = time.time()
        auto = []
        try:
            auto = parse_fomc_page(httpx.get(FOMC_PAGE, headers=UA, timeout=20).text)
            if auto:
                self.auto.write_text(json.dumps(auto, indent=1), encoding="utf-8")
        except Exception as e:  # never trade blind on a failed refresh: fall back to the cached copy
            log.warning("FOMC calendar refresh failed: %s", e)
        if not auto and self.auto.exists():
            auto = json.loads(self.auto.read_text(encoding="utf-8"))
        manual = []
        if self.manual.exists():
            manual = (yaml.safe_load(self.manual.read_text(encoding="utf-8")) or {}).get("events") or []
        self.events = sorted(({**e, "time": _utc(e["time"])} for e in auto + manual), key=lambda e: e["time"])

    def window(self, e: dict) -> tuple[datetime, datetime]:
        w = self.cfg["windows_min"].get(e["type"], self.cfg["windows_min"]["default"])
        t = e["time"]
        return t - timedelta(minutes=w[0]), t + timedelta(minutes=w[1])

    def active(self, now: datetime) -> dict | None:
        """The event whose blackout window contains `now`, if any."""
        for e in self.events:
            a, b = self.window(e)
            if a <= now <= b:
                return e
        return None

    def upcoming(self, now: datetime, hours: float = 48) -> list[dict]:
        return [e for e in self.events if now <= e["time"] <= now + timedelta(hours=hours)]


def _utc(t) -> datetime:
    d = t if isinstance(t, datetime) else datetime.fromisoformat(str(t))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def page_text(url: str, max_chars: int = 6000) -> str:
    html = httpx.get(url, headers=UA, timeout=20, follow_redirects=True).text
    m = re.search(r'<div[^>]+id="article"[^>]*>(.*?)<div[^>]+class="[^"]*lastUpdate', html, re.S)
    body = m.group(1) if m else html
    body = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body, flags=re.S)
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", body))).strip()[:max_chars]


def score_release(llm, model: str, title: str, text: str, previous: str | None) -> dict:
    user = f"NEW RELEASE: {title}\n{text}\n\nPREVIOUS FOMC STATEMENT:\n{previous or '(not available)'}"
    res, _, _ = llm.json_chat(model, FED_SYSTEM, user)
    return {"stance": float(res.get("stance", 0)), "surprise": float(res.get("surprise", 0)),
            "key_change": str(res.get("key_change", ""))[:300], "crypto_effect": float(res.get("crypto_effect", 0))}
