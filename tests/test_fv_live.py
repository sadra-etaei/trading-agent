"""LiveFV executor against the fake exchange: maker entry -> real fill -> maker exit; flatten at end."""
import time

import pytest

from engine.config import load_config
from engine.fv.live import LiveFV
from engine.wallex.client import WallexClient
from tests.fake_wallex import FakeWallex

MARKETS = {"QQTMN": {"min_notional": 50_000, "qty_dec": 6, "price_dec": 0}}


def snap(t, bid, ask, F_usd=10.0, usdt=1000.0):
    # X/USDT mid = F_usd, USDT/TMN mid = usdt -> fair value F = F_usd * usdt
    return {"ts": t, "b": {"QQTMN": [[[bid, 5.0]], [[ask, 5.0]]],
                           "QQUSDT": [[[F_usd - 0.001, 5.0]], [[F_usd + 0.001, 5.0]]],
                           "USDTTMN": [[[usdt - 0.5, 9.0]], [[usdt + 0.5, 9.0]]]}}


def make(tmp_path, fake, end=None, now=None):
    now = now or time.time()
    cfg = load_config("config.yaml", {"challenge": {"end": end}, "fv": {"notional_tmn": 600_000, "min_profit": 0.0}})
    client = WallexClient(api_key="k", min_request_interval_s=0, transport=fake.transport(), max_retries=0)
    lv = LiveFV(cfg, client, ["QQ"], MARKETS, tmp_path)
    lv.broker.poll_s, lv.broker.taker_wait_s = 0.0, 0.05
    st = lv.states["QQ"]
    for i in range(40):  # warm the offset window at zero premium
        st.hist.append((now - 400 + i, 0.0))
    st.last_change = {"usd": now, "usdt": now}
    st.last_val = {"usd": 10.0, "usdt": 1000.0}
    return lv


def test_entry_fill_exit_cycle(tmp_path):
    fake = FakeWallex(bid=9900, ask=9950, maker_fill_after=1)
    now = time.time()
    lv = make(tmp_path, fake, now=now)
    lv.tick(snap(now, 9900, 9950), now)            # gap ~ -1% -> maker BUY placed
    assert lv.orders["QQ"]["side"] == "BUY"
    fake.bid, fake.ask = 9990, 10060
    lv.tick(snap(now + 2, 9990, 10060), now + 2)   # poll -> BUY filled -> SELL at fair value placed
    eng = lv.engines["QQ"]
    assert eng.pos is not None and lv.orders["QQ"]["side"] == "SELL"
    assert lv.orders["QQ"]["price"] >= 10000
    lv.tick(snap(now + 4, 9990, 10060), now + 4)   # poll -> SELL filled
    assert eng.pos is None and len(eng.trades) == 1 and eng.trades[0].exit_kind == "maker_target"
    assert fake.balances.get("QQ", 0) == pytest.approx(0, abs=1e-6)


def test_flatten_before_end_sells_to_tmn(tmp_path):
    fake = FakeWallex(bid=9900, ask=9950, maker_fill_after=1)
    now = time.time()
    end = __import__("pandas").Timestamp(now + 3600, unit="s", tz="UTC").isoformat()
    lv = make(tmp_path, fake, end=end, now=now)
    lv.flatten_s, lv.no_entry_s = 60, 120
    lv.tick(snap(now, 9900, 9950), now)
    lv.tick(snap(now + 2, 9900, 9950), now + 2)    # bought
    assert lv.engines["QQ"].pos is not None
    fake.maker_fill_after = None                    # resting exit never fills -> must be force-sold
    late = lv.end_ts - 30                           # inside the flatten window
    lv.tick(snap(late, 9900, 9950), late)
    assert lv.engines["QQ"].pos is None and not lv.orders
    assert fake.balances.get("QQ", 0) == pytest.approx(0, abs=1e-6)
    assert lv.engines["QQ"].trades[-1].exit_kind == "flatten_end"
