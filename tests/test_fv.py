import numpy as np
import pandas as pd

from engine.fv.sim import FVParams, simulate_pair

T0 = pd.Timestamp("2026-10-22 10:00", tz="UTC")


def frame(rows):
    """rows: (sec, bid, ask, F) -> fair-value frame as produced by fv.data (offset 0, fresh)."""
    idx = [T0 + pd.Timedelta(seconds=s) for s, *_ in rows]
    d = pd.DataFrame([r[1:] for r in rows], columns=["bid", "ask", "F"], index=idx)
    d["bid_qty"] = d["ask_qty"] = 1.0
    d["offset"] = 0.0
    d["gap_bid"] = np.log(d["bid"] / d["F"])
    d["usd_spread"] = 0.0
    d["stale_usd"] = d["stale_usdt"] = False
    return d


def trades(rows):
    if not rows:
        return pd.DataFrame(columns=["price", "qty", "buy"])
    idx = [T0 + pd.Timedelta(seconds=s) for s, *_ in rows]
    return pd.DataFrame([r[1:] for r in rows], columns=["price", "qty", "buy"], index=idx)


def test_trade_through_fill_then_maker_exit():
    # TMN bid 9900 vs fair 10000 -> gap -1%: buy posted at min(bid+1, 10000*e^-0.004) = 9901
    d = frame([(0, 9900, 9950, 10000), (5, 9900, 9950, 10000), (10, 9960, 10010, 10000), (15, 9990, 10050, 10000), (20, 9990, 10050, 10000)])
    tr = trades([(4, 9899, 1.0, False),      # trades through our 9901 bid after 2s latency -> filled
                 (17, 10051, 1.0, True)])    # trades through our exit ask -> filled
    p = FVParams(theta_in=0.004, maker_fee=0.001, taker_fee=0.002, min_profit=0.0)
    out, st = simulate_pair("XTMN", d, tr, p)
    assert st["fills"] == 1 and len(out) == 1
    t = out[0]
    assert t.entry_px == 9901 and t.exit_kind == "maker_target"
    assert t.exit_px >= 10000 and t.pnl_tmn > 0


def test_no_fill_without_trade_through_and_time_stop():
    d = frame([(0, 9900, 9950, 10000), (5, 9900, 9950, 10000)])
    out, st = simulate_pair("XTMN", d, trades([(3, 9901, 5.0, True)]), FVParams())  # buyer-initiated at our price: no fill
    assert st["fills"] == 0 and not out
    # filled, price never reaches target -> time stop at the bid (taker)
    d = frame([(0, 9900, 9950, 10000), (5, 9900, 9950, 10000), (4000, 9905, 9950, 10000)])
    out, _ = simulate_pair("XTMN", d, trades([(4, 9800, 1.0, False)]), FVParams(hold_max_s=1800))
    assert len(out) == 1 and out[0].exit_kind == "time_stop" and out[0].exit_px == 9905
