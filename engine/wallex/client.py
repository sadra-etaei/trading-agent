"""Wallex REST client.

Public endpoints (verified against the live API):
  GET /v1/markets                     symbols, precision, minQty, minNotional, 24h stats
  GET /v1/depth?symbol=               order book
  GET /v1/trades?symbol=              recent public trades
  GET /v1/udf/history?symbol=&resolution=&from=&to=
      resolution "15" returns 1-minute candles (~4320 per request, >=500 days history)
      resolution "60" returns 1-hour candles (>=3 years history)

Private endpoints (auth header `x-api-key`; /v1/account/balances confirmed to exist, returns 401
without a key). Paths and payloads below follow the public Wallex API docs and MUST be verified
with `python cli.py verify-api` once the key is available.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Wallex UDF resolution codes (empirical, see module docstring).
RES_1M = "15"
RES_1H = "60"


class WallexError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, payload: Any = None):
        super().__init__(msg)
        self.status = status
        self.payload = payload


class WallexClient:
    def __init__(
        self,
        base_url: str = "https://api.wallex.ir",
        api_key: str | None = None,
        timeout_s: float = 15,
        max_retries: int = 4,
        min_request_interval_s: float = 0.25,
        transport: httpx.BaseTransport | None = None,  # tests inject a fake exchange here
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("WALLEX_API_KEY")
        self.max_retries = max_retries
        self.min_interval = min_request_interval_s
        self._last_req = 0.0
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=timeout_s, headers={"User-Agent": "wallex-ai-engine/1.0"}, transport=transport)

    @classmethod
    def from_config(cls, cfg) -> "WallexClient":
        w = cfg.wallex
        return cls(w.base_url, None, w.timeout_s, w.max_retries, w.min_request_interval_s)

    # ------------------------------------------------------------------ transport
    def _throttle(self) -> None:
        with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last_req)
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.monotonic()

    def request(self, method: str, path: str, *, params=None, json=None, private=False) -> Any:
        headers = {}
        if private:
            if not self.api_key:
                raise WallexError("WALLEX_API_KEY is not set")
            headers["x-api-key"] = self.api_key
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                r = self._http.request(method, url, params=params, json=json, headers=headers)
            except httpx.HTTPError as e:  # network problem -> retry
                last_exc = e
                log.warning("wallex %s %s network error (%s), attempt %d", method, path, e, attempt + 1)
                time.sleep(min(2 ** attempt, 20))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                last_exc = WallexError(f"HTTP {r.status_code}", r.status_code, r.text[:500])
                log.warning("wallex %s %s HTTP %d, attempt %d", method, path, r.status_code, attempt + 1)
                time.sleep(min(2 ** attempt, 20))
                continue
            try:
                data = r.json()
            except ValueError:
                raise WallexError(f"non-JSON response HTTP {r.status_code}", r.status_code, r.text[:500])
            if r.status_code >= 400:
                raise WallexError(f"HTTP {r.status_code}: {data}", r.status_code, data)
            return data
        raise WallexError(f"request failed after retries: {method} {path}: {last_exc}")

    # ------------------------------------------------------------------ public
    def markets(self) -> dict[str, dict]:
        return self.request("GET", "/v1/markets")["result"]["symbols"]

    def depth(self, symbol: str) -> dict[str, list[dict]]:
        res = self.request("GET", "/v1/depth", params={"symbol": symbol})["result"]
        return {"ask": res.get("ask", []), "bid": res.get("bid", [])}

    def trades(self, symbol: str) -> list[dict]:
        return self.request("GET", "/v1/trades", params={"symbol": symbol})["result"]["latestTrades"]

    def candles(self, symbol: str, resolution: str, start: int, end: int) -> dict:
        """Raw UDF candles: dict with t,o,h,l,c,v lists (strings) or s='no_data'."""
        return self.request(
            "GET",
            "/v1/udf/history",
            params={"symbol": symbol, "resolution": resolution, "from": int(start), "to": int(end)},
        )

    # ------------------------------------------------------------------ private (VERIFY)
    def balances(self) -> dict[str, dict]:
        return self.request("GET", "/v1/account/balances", private=True)["result"]["balances"]

    def place_order(
        self, symbol: str, side: str, price: str, quantity: str, client_id: str, order_type: str = "LIMIT"
    ) -> dict:
        body = {
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "price": price,
            "quantity": quantity,
            "client_id": client_id,
        }
        return self.request("POST", "/v1/account/orders", json=body, private=True)["result"]

    def get_order(self, client_id: str) -> dict:
        return self.request("GET", f"/v1/account/orders/{client_id}", private=True)["result"]

    def cancel_order(self, client_id: str) -> dict:
        return self.request("DELETE", "/v1/account/orders", json={"clientOrderId": client_id}, private=True)

    def open_orders(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else None
        res = self.request("GET", "/v1/account/openOrders", params=params, private=True)["result"]
        return res.get("orders", res) if isinstance(res, dict) else res

    def account_trades(self, symbol: str | None = None) -> list[dict]:
        params = {"symbol": symbol} if symbol else None
        res = self.request("GET", "/v1/account/trades", params=params, private=True)["result"]
        return res.get("AccountLatestTrades", res) if isinstance(res, dict) else res
