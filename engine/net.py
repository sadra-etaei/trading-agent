"""External endpoints that may need a different route when the bot runs on a server in Iran (Liara).

Binance geo-blocks Iranian IPs (HTTP 451). The base URLs can be changed with environment variables, and httpx
honours HTTPS_PROXY / NO_PROXY, so a proxy can carry the Binance traffic while Wallex stays direct
(e.g. NO_PROXY=api.wallex.ir,ai.boxapi.ir). `python cli.py check-net` tests every endpoint from where it runs.
"""
from __future__ import annotations

import os


def binance_spot() -> str:
    return os.environ.get("BINANCE_SPOT_URL", "https://api.binance.com").rstrip("/")


def binance_futures() -> str:
    return os.environ.get("BINANCE_FUTURES_URL", "https://fapi.binance.com").rstrip("/")
