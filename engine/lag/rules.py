"""Entry and exit rules shared by the historical test, the gap logger and (later) live trading,
so every stage tests the same decisions."""
from __future__ import annotations


def bn_threshold(cfg_lag, symbol: str) -> float:
    m = cfg_lag["bn_move_3m"]
    return float(m.get(symbol, m["default"]))


def fair_price(bn_mid: float, basis: float) -> float:
    return bn_mid * (1 + basis)


def entry_signal(cfg_lag, symbol: str, wx_ask: float, fair: float, bn_ret_3m: float) -> bool:
    """Buy when Wallex still sells below where Binance says it should be, right after Binance rose."""
    return (wx_ask < fair * (1 - cfg_lag["entry_gap"])
            and bn_ret_3m >= bn_threshold(cfg_lag, symbol))


def exit_reason(cfg_lag, entry_px: float, bn_entry: float, bn_now: float, wx_high: float,
                minutes_held: float) -> str | None:
    """Checked once per minute (historical) or per snapshot (live). The worse outcome is checked
    first: a Binance reversal beats a take-profit touched in the same minute."""
    if bn_now <= bn_entry * (1 - cfg_lag["bn_reverse"]):
        return "bn_reverse"
    if wx_high > entry_px * (1 + cfg_lag["take_profit"]):
        return "take_profit"
    if minutes_held >= cfg_lag["time_stop_min"]:
        return "time_stop"
    return None
