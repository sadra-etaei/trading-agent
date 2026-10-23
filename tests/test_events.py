import time
from datetime import datetime, timedelta, timezone

import pytest

from engine.config import load_config
from engine.sentiment import fed
from engine.sentiment.events import EventGate

PAGE = """<h4>2026 FOMC Meetings</h4>
<div class="fomc-meeting__month col-xs-5"><strong>October</strong></div>
<div class="fomc-meeting__date col-xs-4">27-28</div>
<div class="fomc-meeting__month col-xs-5"><strong>December</strong></div>
<div class="fomc-meeting__date col-xs-4">8-9*</div>
<a href="/newsevents/pressreleases/monetary20260916a.htm">HTML</a>"""


class FakeLLM:
    def __init__(self, confirm=True):
        self.confirm, self.calls = confirm, 0

    def json_chat(self, model, system, user):
        self.calls += 1
        return {"confirm": self.confirm, "reason": "test"}, "k", False


@pytest.fixture
def gate():
    g = EventGate(load_config("config.yaml"), None, None, ["BTCUSDT", "SOLUSDT"])
    g.cal.events = []
    return g


def item(**kw):
    base = {"id": "x1", "source": "trump", "title": "New 100% tariffs on China", "text": "", "kind": "rss",
            "published": time.time() - 60, "scored": True, "score": -0.6, "severity": 2, "novelty": 0.8,
            "category": "tariff_trade", "assets": ["ALL"]}
    return base | kw


def test_fomc_page_parse_and_dst():
    ev = fed.parse_fomc_page(PAGE)
    assert [e["time"] for e in ev] == ["2026-10-28T18:00:00+00:00", "2026-12-09T19:00:00+00:00"]  # EDT then EST
    assert ev[1]["projections"] and fed.statement_dates(PAGE) == ["20260916"]


def test_negative_news_blocks_entries(gate):
    gate._act(item(), time.time())
    blocked, why = gate.entries_blocked()
    assert blocked and "tariff" in why
    assert gate.take_exits(["BTCUSDT"]) == []  # severity 2: block only, no selling


def test_old_or_positive_items_never_act(gate):
    gate._act(item(published=time.time() - 5 * 3600), time.time())
    gate._act(item(id="x2", score=0.9), time.time())
    assert gate.entries_blocked() == (False, "")


def test_shock_sells_only_when_confirmed(gate):
    gate.llm = FakeLLM(confirm=False)
    gate._act(item(severity=3, score=-0.9), time.time())
    assert gate.take_exits(["BTCUSDT"]) == []
    gate.llm = FakeLLM(confirm=True)
    gate._act(item(id="x3", severity=3, score=-0.9), time.time())
    assert gate.take_exits(["BTCUSDT", "SOLUSDT"]) == ["BTCUSDT", "SOLUSDT"]


def test_hack_of_held_coin_exits_that_coin(gate):
    gate.llm = FakeLLM()
    gate._act(item(category="hack_exchange", assets=["SOL"], severity=2, score=-0.7), time.time())
    assert gate.take_exits(["BTCUSDT", "SOLUSDT"]) == ["SOLUSDT"]


def test_fed_rules(gate):
    now = time.time()
    gate._act(item(fed={"stance": 1, "surprise": 0.6, "key_change": "cut"}, kind="fed"), now)
    assert gate.block_until == pytest.approx(now - 60 + 15 * 60, abs=2)
    gate._act(item(id="f2", fed={"stance": -2, "surprise": 0.7, "key_change": "hike"}, kind="fed"), now)
    assert gate.block_until == pytest.approx(now - 60 + 4 * 3600, abs=2)
    assert gate.take_exits(["BTCUSDT"]) == ["BTCUSDT"]


def test_calendar_window_blocks(gate):
    t = datetime.now(timezone.utc) + timedelta(minutes=90)
    gate.cal.events = [{"time": t, "type": "fomc", "title": "FOMC decision"}]
    assert gate.entries_blocked()[0]  # inside the 2 h pre-FOMC window
    gate.cal.events = [{"time": t + timedelta(hours=2), "type": "fomc", "title": "FOMC decision"}]
    assert not gate.entries_blocked()[0]
