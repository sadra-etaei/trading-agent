"""Deterministic risk manager. It can shrink or block trades, never enlarge them.
Every check result is returned so it can be written to the decision log."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.risk.portfolio import Portfolio, Position


@dataclass
class SizeDecision:
    allowed: bool
    notional: float
    checks: list[str]
    reason: str = ""


class RiskManager:
    def __init__(self, cfg, start_ts: pd.Timestamp | None = None):
        self.r = cfg.risk
        self.c = cfg.challenge
        self.bar_hours = {"15m": 0.25, "30m": 0.5, "1h": 1.0, "4h": 4.0}.get(cfg.data.bar, 1.0)
        self.start_ts = start_ts
        self.end_ts = pd.Timestamp(self.c.end) if self.c.end else None
        self.halted = False
        self.halt_reason = ""
        self.day_halt = ""  # Tehran day on which the daily halt fired (flat until the next day)

    def set_window(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp | None) -> None:
        self.start_ts = start_ts
        self.end_ts = end_ts

    # ---------------------------------------------------------------- account state
    def update(self, pf: Portfolio, equity: float) -> list[str]:
        """Called every mark. Returns events (e.g. kill switch)."""
        events = []
        dd = 1 - equity / pf.peak_equity if pf.peak_equity else 0.0
        limit = self.r.get("daily_halt")
        if limit and self.day_halt != pf.current_day and self.daily_loss(pf, equity) >= limit:
            self.day_halt = pf.current_day
            events.append("daily_halt")
        if not self.halted and dd >= self.r.max_drawdown_kill:
            self.halted = True
            self.halt_reason = f"max drawdown {dd:.2%} >= {self.r.max_drawdown_kill:.0%}"
            events.append("kill_switch")
        return events

    def daily_loss(self, pf: Portfolio, equity: float) -> float:
        return 1 - equity / pf.day_start_equity if pf.day_start_equity else 0.0

    def entries_blocked(self, ts: pd.Timestamp, pf: Portfolio, equity: float) -> str | None:
        if self.halted:
            return f"halted: {self.halt_reason}"
        if self.day_halt and self.day_halt == _day(ts):
            return f"daily halt: day loss >= {self.r.daily_halt:.1%}"
        dl = self.daily_loss(pf, equity)
        if dl >= self.r.daily_loss_limit:
            return f"daily loss {dl:.2%} >= {self.r.daily_loss_limit:.0%}"
        if self.end_ts is not None and ts >= self.end_ts - pd.Timedelta(minutes=self.c.no_new_entries_before_end_min):
            return "no new entries near challenge end"
        return None

    def day_halted(self, ts: pd.Timestamp) -> bool:
        return bool(self.day_halt) and self.day_halt == _day(ts)

    def flatten_reason(self, ts: pd.Timestamp) -> str:
        return "kill_switch" if self.halted else ("daily_halt" if self.day_halted(ts) else "flatten_end")

    def must_flatten(self, ts: pd.Timestamp) -> bool:
        if self.halted or (self.day_halt and self.day_halt == _day(ts)):
            return True
        return self.end_ts is not None and ts >= self.end_ts - pd.Timedelta(minutes=self.c.flatten_before_end_min)

    # ---------------------------------------------------------------- sizing
    def size(
        self,
        ts: pd.Timestamp,
        symbol: str,
        sl_pct: float,
        size_scale: float,
        pf: Portfolio,
        prices: dict[str, float],
        min_notional: float,
        pending_notional: float = 0.0,
        pending_count: int = 0,
    ) -> SizeDecision:
        checks = []
        eq = pf.equity(prices)
        if symbol in pf.positions:
            return SizeDecision(False, 0, checks, "already in position (no averaging)")
        if len(pf.positions) + pending_count >= self.r.max_positions:
            return SizeDecision(False, 0, checks, f"max positions {self.r.max_positions}")
        last_loss = pf.last_loss_ts.get(symbol)
        if last_loss is not None and ts - last_loss < pd.Timedelta(hours=self.r.cooldown_bars_after_loss * self.bar_hours):
            return SizeDecision(False, 0, checks, "cooldown after loss")

        frac = min(self.r.max_position_frac, self.r.risk_per_trade / max(sl_pct, 1e-6))
        checks.append(f"base frac {frac:.3f} = min(max_position {self.r.max_position_frac}, risk/trade {self.r.risk_per_trade}/sl {sl_pct:.4f})")
        scale = max(0.0, min(1.0, size_scale))
        if self.start_ts is not None and ts - self.start_ts < pd.Timedelta(hours=self.r.ramp_up_hours):
            scale *= self.r.ramp_up_size
            checks.append(f"ramp-up x{self.r.ramp_up_size}")
        notional = eq * frac * scale
        checks.append(f"scale {scale:.2f} -> {notional:,.0f} TMN")

        room = eq * self.r.max_total_exposure_frac - pf.exposure(prices) - pending_notional
        if notional > room:
            notional = max(0.0, room)
            checks.append(f"capped by total exposure -> {notional:,.0f}")
        cash_room = (pf.available() - pending_notional) * 0.99
        if notional > cash_room:
            notional = max(0.0, cash_room)
            checks.append(f"capped by cash -> {notional:,.0f}")
        if notional < min_notional:
            return SizeDecision(False, 0, checks, f"below min notional {min_notional:,.0f}")
        return SizeDecision(True, notional, checks)


def _day(ts: pd.Timestamp, tz_offset_h: float = 3.5) -> str:
    """Tehran calendar day, the same day key Portfolio.mark uses."""
    return (ts + pd.Timedelta(hours=tz_offset_h)).strftime("%Y-%m-%d")


def exit_check(pos: Position, high: float, low: float, ts: pd.Timestamp, trailing_atr: float) -> tuple[str, float] | None:
    """Intrabar exit logic on one price bar (1-minute in backtest, a poll in live).
    Order: stop-loss first (conservative), then take-profit, then trailing stop."""
    if low <= pos.sl_price:
        return ("trailing_stop" if pos.trailing_active else "stop_loss", pos.sl_price)
    if high >= pos.tp_price:
        return ("take_profit", pos.tp_price)
    # trailing: after +1 ATR, stop trails `trailing_atr` ATR below the peak
    pos.peak_price = max(pos.peak_price, high)
    atr_abs = pos.atr_pct * pos.entry_price
    if pos.peak_price >= pos.entry_price + atr_abs:
        pos.trailing_active = True
        pos.sl_price = max(pos.sl_price, pos.peak_price - trailing_atr * atr_abs, pos.entry_price * 1.001)
    if ts >= pos.time_stop_ts:
        return ("time_stop", None)  # exit at current price
    return None


def binance_guard(pos: Position, ret_3m: float | None, price: float, cfg_guard) -> str | None:
    """Binance crash guard (STRATEGY.md §7.1f). Wallex follows Binance by ~1-2 minutes, so a sharp
    Binance drop predicts a Wallex drop. Returns "exit" to sell now, or tightens the stop in place
    ("tighten" mode, once per position) and returns None."""
    if ret_3m is None or ret_3m != ret_3m or ret_3m > -cfg_guard["drop_3m"]:
        return None
    if cfg_guard["mode"] == "exit":
        return "exit"
    if not pos.guard_fired:
        pos.sl_price = max(pos.sl_price, price * (1 - cfg_guard["tighten_to"]))
        pos.guard_fired = True
    return None
