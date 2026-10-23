"""In-process fake of the Wallex private API (httpx.MockTransport) for testing WallexBroker.

Order status fields follow what the broker parses (status, executedQty, executedSum, executedPrice,
fee); confirm against the real API with `cli.py verify-api` once a key is available.
"""
from __future__ import annotations

import json

import httpx


class FakeWallex:
    def __init__(self, bid=100.0, ask=101.0, maker_fill_after=None, maker_partial=None):
        self.bid, self.ask = bid, ask
        self.maker_fill_after = maker_fill_after  # status checks before a resting order fills fully
        self.maker_partial = maker_partial        # fraction filled on first status check, then rests
        self.orders: dict[str, dict] = {}
        self.balances = {"TMN": 5_000_000.0}
        self.calls: list[tuple[str, str]] = []

    # ------------------------------------------------------------------ helpers
    def _fill(self, o, qty, price):
        qty = min(qty, o["qty"] - o["executedQty"])
        if qty <= 0:
            return
        o["executedSum"] += qty * price
        o["executedQty"] += qty
        o["status"] = "FILLED" if o["executedQty"] >= o["qty"] - 1e-12 else "PARTIALLY_FILLED"
        base = o["symbol"].removesuffix("TMN")
        if o["side"] == "BUY":
            self.balances["TMN"] -= qty * price
            self.balances[base] = self.balances.get(base, 0.0) + qty
        else:
            self.balances[base] = self.balances.get(base, 0.0) - qty
            self.balances["TMN"] += qty * price

    def _view(self, o):
        q = o["executedQty"]
        return {"clientOrderId": o["cid"], "symbol": o["symbol"], "side": o["side"], "price": str(o["price"]),
                "origQty": str(o["qty"]), "status": o["status"], "executedQty": str(q), "executedSum": str(o["executedSum"]),
                "executedPrice": str(o["executedSum"] / q if q else 0), "fee": "0"}

    # ------------------------------------------------------------------ transport
    def handler(self, req: httpx.Request) -> httpx.Response:
        path, m = req.url.path, req.method
        self.calls.append((m, path))
        if req.url.path.startswith("/v1/account") and "x-api-key" not in req.headers:
            return httpx.Response(401, json={"message": "Unauthorized"})
        if m == "GET" and path == "/v1/depth":
            return httpx.Response(200, json={"result": {"bid": [{"price": str(self.bid), "quantity": 10}],
                                                        "ask": [{"price": str(self.ask), "quantity": 10}]}})
        if m == "GET" and path == "/v1/account/balances":
            return httpx.Response(200, json={"result": {"balances": {
                a: {"asset": a, "value": str(v), "locked": "0"} for a, v in self.balances.items()}}})
        if m == "POST" and path == "/v1/account/orders":
            b = json.loads(req.content)
            o = {"cid": b["client_id"], "symbol": b["symbol"], "side": b["side"], "price": float(b["price"]),
                 "qty": float(b["quantity"]), "executedQty": 0.0, "executedSum": 0.0, "status": "NEW", "checks": 0}
            self.orders[o["cid"]] = o
            if o["side"] == "BUY" and o["price"] >= self.ask:
                self._fill(o, o["qty"], self.ask)       # marketable: fills at the ask
            elif o["side"] == "SELL" and o["price"] <= self.bid:
                self._fill(o, o["qty"], self.bid)
            return httpx.Response(200, json={"result": self._view(o)})
        if m == "GET" and path.startswith("/v1/account/orders/"):
            o = self.orders[path.rsplit("/", 1)[1]]
            if o["status"] in ("NEW", "PARTIALLY_FILLED"):
                o["checks"] += 1
                if self.maker_partial and o["checks"] == 1:
                    self._fill(o, o["qty"] * self.maker_partial, o["price"])
                elif self.maker_fill_after is not None and o["checks"] >= self.maker_fill_after:
                    self._fill(o, o["qty"], o["price"])
            return httpx.Response(200, json={"result": self._view(o)})
        if m == "DELETE" and path == "/v1/account/orders":
            o = self.orders[json.loads(req.content)["clientOrderId"]]
            if o["status"] in ("NEW", "PARTIALLY_FILLED"):
                o["status"] = "CANCELED"
            return httpx.Response(200, json={"result": self._view(o)})
        return httpx.Response(404, json={"message": "not found"})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)
