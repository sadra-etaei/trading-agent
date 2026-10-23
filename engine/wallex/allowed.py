"""Challenge market blocklist (blocked_markets.txt, BASE/QUOTE per line -> Wallex symbol BASEQUOTE).
Checked when building the universe and again right before any order is placed."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


class BlockedMarketError(RuntimeError):
    pass


@lru_cache(maxsize=8)
def load_blocked(path: str = "blocked_markets.txt") -> frozenset[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"blocked market list not found: {p}")
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.replace("/", "").upper())
    return frozenset(out)


def is_allowed(symbol: str, path: str = "blocked_markets.txt") -> bool:
    return symbol.upper() not in load_blocked(path)


def assert_allowed(symbol: str, path: str = "blocked_markets.txt") -> None:
    if not is_allowed(symbol, path):
        raise BlockedMarketError(f"{symbol} is not an allowed challenge market")
