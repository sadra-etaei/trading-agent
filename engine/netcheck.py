"""`python cli.py check-net`: tests every outside service the bot uses, from wherever it runs (e.g. Liara in Iran,
where Binance answers HTTP 451 to Iranian IPs). Exit code 1 if a critical service fails."""
from __future__ import annotations

import os
import time

import httpx

from engine.net import binance_futures, binance_spot


def endpoints() -> list[tuple[str, str, bool, dict]]:
    gw = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY", "")
    out = [
        ("Wallex markets", "https://api.wallex.ir/v1/markets", True, {}),
        ("Wallex order book", "https://api.wallex.ir/v1/depth?symbol=BTCTMN", True, {}),
        ("Wallex candles", "https://api.wallex.ir/v1/udf/history?symbol=BTCTMN&resolution=60"
                           f"&from={int(time.time()) - 7200}&to={int(time.time())}", True, {}),
        ("Binance spot candles (model features, crash guard)", binance_spot() + "/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=2", True, {}),
        ("  alt: Binance mirror data-api.binance.vision", "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=2", False, {}),
        ("Binance futures funding (model feature)", binance_futures() + "/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1", False, {}),
        ("Fear & Greed index", "https://api.alternative.me/fng/?limit=1", False, {}),
        ("Trump posts (event gate)", "https://www.trumpstruth.org/feed", False, {}),
        ("Fed releases (event gate)", "https://www.federalreserve.gov/feeds/press_monetary.xml", False, {}),
        ("Fed FOMC calendar", "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm", False, {}),
        ("Binance announcements (event gate)", "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query?type=1&pageNo=1&pageSize=1", False, {}),
        ("CoinDesk RSS", "https://www.coindesk.com/arc/outboundfeeds/rss/", False, {}),
        ("Cointelegraph RSS", "https://cointelegraph.com/rss", False, {}),
        ("BBC World RSS", "https://feeds.bbci.co.uk/news/world/rss.xml", False, {}),
    ]
    if gw:
        out.insert(3, ("GPT gateway (agent)", gw + "/models", False, {"Authorization": f"Bearer {key}"} if key else {}))
    return out


def run() -> int:
    bad_critical = 0
    print(f"{'service':52s} {'result':10s} {'time':>7s}")
    with httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as c:
        for name, url, critical, headers in endpoints():
            t0 = time.time()
            for attempt in range(2):  # one retry: a single timeout is not an outage
                try:
                    r = c.get(url, headers=headers)
                    ok = r.status_code < 400
                    res = f"HTTP {r.status_code}"
                except httpx.HTTPError as e:
                    ok, res = False, type(e).__name__
                if ok:
                    break
            print(f"{name:52s} {('OK ' if ok else 'FAIL ') + res:10s} {time.time() - t0:6.1f}s{'  <- CRITICAL' if critical and not ok else ''}")
            bad_critical += int(critical and not ok)
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    print(f"\nproxy: {proxy or 'none'} | NO_PROXY: {os.environ.get('NO_PROXY') or os.environ.get('no_proxy') or 'none'}")
    if bad_critical:
        print(f"{bad_critical} critical service(s) unreachable: do not go live. If the 'alt' Binance mirror is OK, set "
              "BINANCE_SPOT_URL=https://data-api.binance.vision; otherwise set HTTPS_PROXY (a proxy outside Iran and "
              "the US) with NO_PROXY=api.wallex.ir,ai.boxapi.ir.")
    else:
        print("all critical services reachable")
    return 1 if bad_critical else 0
