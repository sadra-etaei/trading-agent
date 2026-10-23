"""WallexBroker order flow against the fake exchange (no network)."""
import pytest

from engine.broker.brokers import WallexBroker
from engine.config import load_config
from engine.wallex.allowed import BlockedMarketError
from engine.wallex.client import WallexClient
from tests.fake_wallex import FakeWallex

MARKETS = {"BTCTMN": {"min_notional": 50_000, "qty_dec": 8, "price_dec": 0},
           "ZECTMN": {"min_notional": 50_000, "qty_dec": 8, "price_dec": 0}}


def make(fake: FakeWallex, entry_style: str = "maker_then_taker"):
    cfg = load_config("config.yaml", {"execution": {"maker_timeout_s": 0.05, "max_entry_slippage_pct": 0.003,
                                                    "entry_style": entry_style}})
    client = WallexClient(api_key="test", min_request_interval_s=0, transport=fake.transport(), max_retries=0)
    br = WallexBroker(client, cfg, MARKETS)
    br.poll_s, br.taker_wait_s = 0.0, 0.05
    return br


def test_maker_entry_fills_at_bid():
    fake = FakeWallex(maker_fill_after=1)
    f = make(fake).buy("BTCTMN", 1_000_000)
    assert f is not None and f.price == 100 and f.qty == pytest.approx(10_000, rel=1e-6)
    assert len(f.orders) == 1
    assert fake.balances["BTC"] == pytest.approx(f.qty)


def test_unfilled_maker_is_cancelled_then_crosses_spread():
    fake = FakeWallex(maker_fill_after=None)  # resting orders never fill
    f = make(fake).buy("BTCTMN", 1_000_000)
    assert f is not None and len(f.orders) == 2
    assert f.price == 101  # taker fill at the ask
    first = fake.orders[f.orders[0]]
    assert first["status"] == "CANCELED" and first["executedQty"] == 0


def test_partial_maker_then_taker_remainder():
    fake = FakeWallex(maker_partial=0.4)
    f = make(fake).buy("BTCTMN", 1_000_000)
    assert len(f.orders) == 2
    assert 100 < f.price < 101  # blended maker + taker
    assert f.qty * f.price == pytest.approx(1_000_000, rel=0.01)


def test_sell_crosses_spread_and_balance_updates():
    fake = FakeWallex(maker_fill_after=1)
    br = make(fake)
    b = br.buy("BTCTMN", 1_000_000)
    s = br.sell("BTCTMN", b.qty)
    assert s is not None and s.price == 100 and s.qty == pytest.approx(b.qty)
    assert fake.balances["BTC"] == pytest.approx(0, abs=1e-9)
    assert br.free_balance("TMN") == pytest.approx(5_000_000)


def test_blocked_market_never_reaches_exchange():
    fake = FakeWallex(maker_fill_after=1)
    with pytest.raises(BlockedMarketError):
        make(fake).buy("ZECTMN", 1_000_000)
    assert not any(p == "/v1/account/orders" for _, p in fake.calls)


def test_taker_entry_fills_at_ask_without_maker_wait():
    fake = FakeWallex(maker_fill_after=1)
    f = make(fake, entry_style="taker").buy("BTCTMN", 1_000_000)
    assert f is not None and len(f.orders) == 1 and f.price == 101  # crosses the spread at once


def test_parse_real_wallex_fill_uses_average_not_limit_price():
    """Real USDTTMN BUY from the smoke test (2026-10-23): executedPrice is the limit price."""
    from engine.broker.brokers import WallexBroker

    o = {"status": "FILLED", "price": "273247.0", "origQty": "0.4", "executedSum": "109080.8", "executedQty": "0.4",
         "executedPrice": "273247.0", "fee": "0.0012",
         "fills": [{"price": "272702.0", "quantity": "0.4", "fee": "0.0012", "feeAsset": "USDT"}]}
    status, q, price, fee = WallexBroker._parse(o)
    assert (status, q, fee) == ("FILLED", 0.4, 0.0012)
    assert abs(price - 272702.0) < 1e-6
    sell = {"status": "FILLED", "executedSum": "106314.39", "executedQty": "0.39", "executedPrice": "272056.0",
            "fee": "318.94317", "fills": [{"price": "272601.0", "quantity": "0.39"}]}
    assert abs(WallexBroker._parse(sell)[2] - 272601.0) < 1e-6
