import pandas as pd
import pytest

from engine.agent.overlay import AgentOverlay
from engine.config import load_config

TS = pd.Timestamp("2026-10-22T19:00:00Z")


class FakeLLM:
    def __init__(self, response):
        self.response, self.calls, self.users = response, 0, []

    def json_chat(self, model, system, user):
        self.calls += 1
        self.users.append(user)
        return self.response, "key", False


class FakeNews:
    def digest(self, hours=12, min_severity=1):
        return [{"title": "Exchange X hacked", "severity": 3, "score": -0.8}]


@pytest.fixture
def cfg(tmp_path):
    c = load_config("config.yaml")
    c["agent"]["enabled"] = True
    c["agent"]["cache_dir"] = str(tmp_path)
    return c


def test_batch_one_call_and_veto_needs_reason_code(cfg):
    llm = FakeLLM({"reviews": [
        {"symbol": "BTCUSDT", "action": "veto", "reason_code": "i_dont_like_it", "reasoning": "vague"},
        {"symbol": "ETHUSDT", "action": "veto", "reason_code": "major_negative_news", "reasoning": "hack"},
        {"symbol": "SOLUSDT", "action": "approve", "invalidation": "Binance SOL falls 2% in an hour"}]})
    a = AgentOverlay(cfg, llm)
    out = a.review_batch(TS, [{"symbol": s} for s in ("BTCUSDT", "ETHUSDT", "SOLUSDT")], {}, {})
    assert llm.calls == 1
    assert out["BTCUSDT"]["action"] == "approve" and "note" in out["BTCUSDT"]
    assert out["ETHUSDT"]["action"] == "veto" and out["ETHUSDT"]["size_multiplier"] == 0
    assert out["SOLUSDT"]["invalidation"].startswith("Binance SOL")
    usdt = all(s.endswith("USDT") for s in cfg.universe.core)  # prompt framing follows the traded quote
    assert ("rial does not" in a.review_system) == usdt and (("Toman" in a.review_system) != usdt)


def test_exit_needs_evidence(cfg):
    llm = FakeLLM({"exit": [{"symbol": "BTCUSDT", "evidence": ""}, {"symbol": "ETHUSDT", "evidence": "hack headline"},
                            {"symbol": "XRPUSDT", "evidence": "not held"}]})
    a = AgentOverlay(cfg, llm, FakeNews())
    out = a.review_exits(TS, [{"symbol": "BTCUSDT"}, {"symbol": "ETHUSDT"}])
    assert out["exit"] == ["ETHUSDT"]


def test_reflection_writes_lessons_once(cfg):
    llm = FakeLLM({"lessons": ["a", "b", "c", "d"]})
    a = AgentOverlay(cfg, llm)
    trades = [{"symbol": "BTCUSDT", "decision_id": "d1", "entry_ts": "2026-10-22T10:00:00Z",
               "exit_ts": "2026-10-22T12:00:00Z", "ret_pct": -0.01, "exit_reason": "stop_loss", "hold_hours": 2}]
    assert a.maybe_reflect(TS, trades, {"d1": {"action": "approve"}})["lessons"] == ["a", "b", "c"]
    assert a.maybe_reflect(TS, trades, {}) is None  # not again within reflect_every_h
    assert AgentOverlay(cfg, llm).lessons == ["a", "b", "c"]  # persisted
