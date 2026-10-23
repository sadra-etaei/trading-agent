"""Builds the pooled training table (all pairs) and the time splits."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.config import universe
from engine.data import store
from engine.features.build import build_features, usdt_pairs
from engine.model.labels import triple_barrier
from engine.sentiment import fear_greed, funding


def load_inputs(cfg, symbols: list[str] | None = None):
    symbols = symbols or universe(cfg)
    syms = sorted(set(symbols) | set(cfg.universe.context_pairs))
    if cfg.universe.get("usdt_premium"):
        syms = sorted(set(syms) | set(usdt_pairs(symbols)))
    if cfg.data.get("feature_source") == "binance":  # candidate C: dense Binance candles, same symbol names
        from engine.data import binance_ohlc

        bars = binance_ohlc.load_bars(cfg.data.dir, [s for s in syms if s != "USDTTMN"])
        if "USDTTMN" in syms:
            bars["USDTTMN"] = store.load_bars(cfg.data.dir, ["USDTTMN"], cfg.data.bar)["USDTTMN"]
    elif cfg.data.get("feature_source") == "wallex_1h":  # candidate D: Wallex hourly history back to 2021
        from engine.data import wallex_hourly

        bars = wallex_hourly.load_bars(cfg.data.dir, syms)
    else:
        bars = store.load_bars(cfg.data.dir, syms, cfg.data.bar)
    bars = {s: b for s, b in bars.items() if len(b)}
    fng = fear_greed.load(cfg.data.dir) if cfg.sentiment.fear_greed else None
    fund = {s: funding.load(cfg.data.dir, s) for s in syms} if cfg.sentiment.funding else None
    bn = None
    if (cfg.get("outside") or {}).get("binance"):
        from engine.data import binance_hourly

        if cfg.data.get("feature_source") == "binance":  # order flow from the same Binance candles
            bn = {s: bars[s] for s in symbols if s in bars}
        elif cfg.data.get("feature_source") == "wallex_1h":  # long Binance history (2017+) for the outside features
            from engine.data import binance_ohlc

            bn = binance_ohlc.load_bars(cfg.data.dir, [s for s in symbols if s not in ("XAUTUSDT", "USDTTMN")])
        else:
            bn = {s: binance_hourly.load(cfg.data.dir, s, cfg.data.bar) for s in symbols}
    return bars, fng, fund, bn


def build_table(cfg, symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or cfg.data.get("train_pool") or universe(cfg)  # train_pool: pooled training coins (C)
    bars, fng, fund, bn = load_inputs(cfg, symbols)
    feats = build_features(bars, fng, fund, bn)
    rows = []
    for sym in symbols:
        if sym not in feats:
            continue
        f = feats[sym]
        lab = triple_barrier(bars[sym], f["atr_pct"], cfg.labels)
        t = f.join(lab)
        t["symbol"] = sym
        rows.append(t)
    table = pd.concat(rows).sort_index()
    table = table[table["dist_ema200"].notna()]  # warm-up period
    return table


@dataclass
class Splits:
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    data_end: pd.Timestamp

    def as_dict(self) -> dict:
        return {k: v.isoformat() for k, v in self.__dict__.items()}


def make_splits(index: pd.DatetimeIndex, cfg) -> Splits:
    data_end = index.max()
    purge = pd.Timedelta(hours=int(cfg.labels.horizon_bars) + 1) * _bar_hours(cfg)
    test_start = data_end - pd.Timedelta(days=cfg.model.test_days)
    val_end = test_start - purge
    val_start = val_end - pd.Timedelta(days=cfg.model.val_days)
    train_end = val_start - purge
    return Splits(train_end, val_start, val_end, test_start, data_end)


def _bar_hours(cfg) -> float:
    return {"15m": 0.25, "30m": 0.5, "1h": 1.0, "4h": 4.0}.get(cfg.data.bar, 1.0)


def net_return(outcome, cfg):
    """Trade return after fees and slippage on both sides."""
    return outcome - 2 * cfg.costs.fee_rate - 2 * cfg.costs.slippage_bps / 1e4
