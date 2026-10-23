"""Portfolio / ledger: cash, positions, realized P&L, fees, equity curve. Shared by all modes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_ts: pd.Timestamp
    tp_price: float
    sl_price: float
    atr_pct: float
    time_stop_ts: pd.Timestamp
    decision_id: str
    entry_fee: float
    peak_price: float = 0.0
    trailing_active: bool = False
    guard_fired: bool = False   # Binance crash guard already tightened this position's stop
    invalidation: str = ""      # agent's exit-plan condition, written at entry (early exit only when met)

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price


@dataclass
class Trade:
    symbol: str
    decision_id: str
    entry_ts: str
    exit_ts: str
    entry_price: float
    exit_price: float
    qty: float
    notional_tmn: float
    fees_tmn: float
    pnl_tmn: float
    ret_pct: float
    exit_reason: str
    hold_hours: float


@dataclass
class Portfolio:
    cash: float
    initial: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    fees_paid: float = 0.0
    peak_equity: float = 0.0
    day_start_equity: float = 0.0
    current_day: str = ""
    last_loss_ts: dict[str, pd.Timestamp] = field(default_factory=dict)
    # Idle cash in USDT (user rule): Toman needed for a trade is converted from USDT on entry and back
    # on exit through USDT/TMN; everything returns to Toman at the challenge end.
    usdt_mode: bool = False
    usdt: float = 0.0
    rate: float = 0.0            # USDT/TMN mid used for conversions and marking
    conv_fee: float = 0.0031     # USDT/TMN taker fee + half spread
    conv_fees_paid: float = 0.0

    def __post_init__(self):
        self.peak_equity = self.peak_equity or self.initial
        self.day_start_equity = self.day_start_equity or self.initial

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.usdt * self.rate + sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())

    def available(self) -> float:
        """Toman that can be spent on a new position (idle USDT counts at its conversion value)."""
        return self.cash + self.usdt * self.rate * (1 - self.conv_fee)

    def set_rate(self, rate: float | None) -> None:
        if rate and rate > 0:
            self.rate = float(rate)

    def to_usdt_all(self) -> None:
        if self.usdt_mode and self.cash > 0 and self.rate > 0:
            fee = self.cash * self.conv_fee
            self.usdt += (self.cash - fee) / self.rate
            self.conv_fees_paid += fee
            self.fees_paid += fee
            self.cash = 0.0

    def to_tmn_all(self) -> None:
        if self.usdt > 0 and self.rate > 0:
            gross = self.usdt * self.rate
            fee = gross * self.conv_fee
            self.cash += gross - fee
            self.conv_fees_paid += fee
            self.fees_paid += fee
            self.usdt = 0.0

    def _ensure_tmn(self, amount: float) -> None:
        """Convert just enough USDT to Toman to pay `amount`."""
        need = amount - self.cash
        if self.usdt_mode and need > 0 and self.rate > 0:
            usdt_sold = min(self.usdt, need / (self.rate * (1 - self.conv_fee)))
            gross = usdt_sold * self.rate
            fee = gross * self.conv_fee
            self.usdt -= usdt_sold
            self.cash += gross - fee
            self.conv_fees_paid += fee
            self.fees_paid += fee

    def exposure(self, prices: dict[str, float]) -> float:
        return sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())

    def open(self, pos: Position) -> None:
        cost = pos.qty * pos.entry_price + pos.entry_fee
        self._ensure_tmn(cost)
        self.cash -= cost
        self.fees_paid += pos.entry_fee
        pos.peak_price = pos.entry_price
        self.positions[pos.symbol] = pos

    def close(self, symbol: str, price: float, fee: float, ts: pd.Timestamp, reason: str) -> Trade:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * price - fee
        self.cash += proceeds
        self.fees_paid += fee
        pnl = proceeds - (pos.qty * pos.entry_price + pos.entry_fee)
        t = Trade(
            symbol=symbol,
            decision_id=pos.decision_id,
            entry_ts=pos.entry_ts.isoformat(),
            exit_ts=ts.isoformat(),
            entry_price=pos.entry_price,
            exit_price=price,
            qty=pos.qty,
            notional_tmn=pos.notional,
            fees_tmn=pos.entry_fee + fee,
            pnl_tmn=pnl,
            ret_pct=pnl / pos.notional if pos.notional else 0.0,
            exit_reason=reason,
            hold_hours=(ts - pos.entry_ts).total_seconds() / 3600,
        )
        self.trades.append(t)
        if pnl < 0:
            self.last_loss_ts[symbol] = ts
        if self.usdt_mode and not self.positions:
            self.to_usdt_all()  # flat again -> back to USDT
        return t

    def mark(self, ts: pd.Timestamp, prices: dict[str, float], tz_offset_h: float = 3.5) -> float:
        """Update peak and day-start equity. Day boundary in Tehran time."""
        eq = self.equity(prices)
        day = (ts + pd.Timedelta(hours=tz_offset_h)).strftime("%Y-%m-%d")
        if day != self.current_day:
            self.current_day = day
            self.day_start_equity = eq
        self.peak_equity = max(self.peak_equity, eq)
        return eq

    def snapshot(self) -> dict:
        return {
            "cash": self.cash,
            "positions": {s: {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in asdict(p).items()} for s, p in self.positions.items()},
            "fees_paid": self.fees_paid,
            "peak_equity": self.peak_equity,
            "trades": len(self.trades),
        }
