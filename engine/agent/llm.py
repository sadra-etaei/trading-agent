"""Minimal OpenAI-compatible chat client with a content-addressed response cache.

Uses the REST API directly (no SDK) so any OpenAI-compatible gateway works via `agent.base_url`.
Every request/response pair is stored under agent_cache/<sha256>.json: re-running the same input
replays the same decision, which makes agent decisions reproducible for the judges.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path

import httpx

log = logging.getLogger(__name__)
_SPEND_LOCK = threading.Lock()  # the event gate scores from its own thread


class LLMError(RuntimeError):
    pass


# Models that must never be called (user rule: too expensive for the $2 budget).
FORBIDDEN_MODEL_WORDS = ("astra",)


class LLMBudgetError(LLMError):
    """Spend limit reached or the gateway reports no credit: the agent must switch itself off."""


class LLM:
    def __init__(self, base_url: str, cache_dir: str, timeout_s: float = 20, temperature: float | None = 0,
                 api_key: str | None = None, offline: bool = False, reasoning_effort: str | None = None,
                 budget_usd: float | None = None, price_in_per_m: float = 0.25, price_out_per_m: float = 2.0):
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.offline = offline  # replay-only: never call the API (backtests)
        self.calls = 0
        self.cache_hits = 0
        self.tokens_in = 0
        self.tokens_out = 0
        self.budget_usd = budget_usd
        self.price_in, self.price_out = price_in_per_m, price_out_per_m
        self._spend_path = self.cache_dir / "_spend.json"  # persists across restarts

    @property
    def spent_usd(self) -> float:
        try:
            return float(json.loads(self._spend_path.read_text())["usd"])
        except (OSError, ValueError, KeyError):
            return 0.0

    def _add_spend(self, usage: dict) -> None:
        if usage.get("cost") is not None:  # the gateway reports the real charge per call
            usd = float(usage["cost"])
        else:
            usd = (int(usage.get("prompt_tokens", 0)) * self.price_in + int(usage.get("completion_tokens", 0)) * self.price_out) / 1e6
        with _SPEND_LOCK:
            self._spend_path.write_text(json.dumps({"usd": round(self.spent_usd + usd, 6), "updated": time.time()}))

    def _key(self, model: str, system: str, user: str) -> str:
        return hashlib.sha256(json.dumps([model, system, user], ensure_ascii=False).encode()).hexdigest()

    def json_chat(self, model: str, system: str, user: str) -> tuple[dict, str, bool]:
        """Returns (parsed JSON, cache key, cached?)."""
        if any(w in model.lower() for w in FORBIDDEN_MODEL_WORDS):
            raise LLMError(f"model {model} is forbidden (never use astra models)")
        key = self._key(model, system, user)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"], key, True
        if self.offline:
            raise LLMError("offline mode and no cached response")
        if not self.api_key:
            raise LLMError("OPENAI_API_KEY not set")
        if self.budget_usd is not None and self.spent_usd >= self.budget_usd:
            raise LLMBudgetError(f"agent budget reached (${self.spent_usd:.3f} >= ${self.budget_usd:.2f})")
        body = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        t0 = time.monotonic()
        for attempt in range(3):  # the gateway sometimes drops connection attempts: retry those only
            try:
                r = httpx.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=self.timeout_s,
                )
                break
            except httpx.ConnectError as e:
                if attempt == 2:
                    raise LLMError(f"network: {e}") from e
                time.sleep(2 * (attempt + 1))
            except httpx.ConnectTimeout as e:
                if attempt == 2:
                    raise LLMError(f"network: {e}") from e
                time.sleep(2 * (attempt + 1))
            except httpx.HTTPError as e:
                raise LLMError(f"network: {e}") from e
        if r.status_code == 402 or (r.status_code >= 400 and any(w in r.text.lower() for w in ("insufficient", "quota", "credit", "balance"))):
            raise LLMBudgetError(f"HTTP {r.status_code}: {r.text[:300]}")
        if r.status_code >= 400:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        usage = j.get("usage", {})
        self.tokens_in += int(usage.get("prompt_tokens", 0))
        self.tokens_out += int(usage.get("completion_tokens", 0))
        self._add_spend(usage)
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            raise LLMError(f"invalid JSON from model: {content[:200]}") from e
        self.calls += 1
        path.write_text(
            json.dumps(
                {"model": model, "system": system, "user": user, "response": parsed, "usage": usage,
                 "latency_s": round(time.monotonic() - t0, 3), "ts": time.time()},
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
        return parsed, key, False


def from_config(cfg, offline: bool = False) -> LLM:
    a = cfg.agent
    return LLM(a.base_url, a.cache_dir, a.timeout_s, a.get("temperature"), offline=offline,
               reasoning_effort=a.get("reasoning_effort"), budget_usd=a.get("budget_usd"),
               price_in_per_m=a.get("price_in_per_m", 0.25), price_out_per_m=a.get("price_out_per_m", 2.0))
