"""Feature matrix for the entry model.

Row = one pair at one bar. Index = bar open time (UTC); every feature uses only data up to
that bar's close, so a decision at the close can use the row and enter at the next bar's open.
The same function is used in training, backtest, shadow and live.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine.features import indicators as ind

REGIME_RANGING, REGIME_TRENDING, REGIME_HIGHVOL = 0, 1, 2
REGIME_NAMES = {0: "ranging", 1: "trending", 2: "high_vol"}

FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72",
    "dist_ema20", "dist_ema50", "dist_ema200", "slope_ema20", "slope_ema50",
    "atr_pct", "rvol_24", "atr_rank", "bb_width", "bb_z",
    "rsi", "macd_h", "stoch", "adx", "range_pos_24",
    "vol_z", "flow_24", "active_frac", "stale",
    "btc_ret_1", "btc_ret_6", "btc_ret_24", "btc_rvol_24",
    "usdt_ret_1", "usdt_ret_24",
    "rel_btc_24", "rel_btc_72",
    "hour_sin", "hour_cos", "dow",
    "regime", "sig_pullback", "sig_meanrev",
    "fng", "fng_chg_7",
]
FUNDING_FEATURES = ["funding", "funding_avg_3d"]
# Outside data (Binance spot 1h, STRATEGY.md §7.1i): global price lead, order flow, activity, local premium.
OUTSIDE_FEATURES = ["bn_ret_1", "bn_ret_6", "bn_ret_24", "bn_flow_1", "bn_flow_6", "bn_vol_z",
                    "glob_prem", "glob_prem_chg_6", "lag_gap_1", "lag_gap_6"]
# Extra features (PLAN.md §15c, from the research report): multi-hour Binance taker flow and activity, short-horizon
# moves relative to BTC, volatility regime, cross-sectional momentum ranks, market breadth, funding change.
EXTRA_FEATURES = ["bn_flow_4", "bn_flow_8", "bn_flow_12", "bn_trades_z", "rel_btc_1", "rel_btc_6", "rvol_ratio",
                  "ret_rank_6", "ret_rank_24", "mkt_breadth_ema50", "mkt_ret_24", "funding_chg_24"]
# Wallex-internal fair value: BTCTMN vs BTCUSDT x USDTTMN (all from Wallex, no external data).
PREMIUM_FEATURES = ["prem", "prem_z", "prem_chg_3"]

TEHRAN_OFFSET_H = 3.5


def _bars_per_day(index: pd.DatetimeIndex) -> int:
    step = (index[1] - index[0]).total_seconds() if len(index) > 1 else 3600
    return max(1, int(round(86400 / step)))


def pair_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars
    c = df["close"]
    out = pd.DataFrame(index=df.index)
    lc = np.log(c)
    for k in (1, 3, 6, 12, 24, 72):
        out[f"ret_{k}"] = lc.diff(k)
    e20, e50, e200 = ind.ema(c, 20), ind.ema(c, 50), ind.ema(c, 200)
    out["dist_ema20"] = c / e20 - 1
    out["dist_ema50"] = c / e50 - 1
    out["dist_ema200"] = c / e200 - 1
    out["slope_ema20"] = e20.pct_change(3)
    out["slope_ema50"] = e50.pct_change(6)
    a = ind.atr(df, 14)
    out["atr_pct"] = a / c
    out["rvol_24"] = out["ret_1"].rolling(24).std()
    bpd = _bars_per_day(df.index)
    out["atr_rank"] = out["atr_pct"].rolling(30 * bpd, min_periods=5 * bpd).rank(pct=True)
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    out["bb_width"] = 4 * sd20 / ma20
    out["bb_z"] = (c - ma20) / sd20.replace(0, np.nan)
    out["rsi"] = ind.rsi(c, 14)
    out["macd_h"] = ind.macd_hist(c) / c
    out["stoch"] = ind.stoch_k(df, 14)
    out["adx"] = ind.adx(df, 14)
    lo24, hi24 = df["low"].rolling(24).min(), df["high"].rolling(24).max()
    out["range_pos_24"] = ((c - lo24) / (hi24 - lo24).replace(0, np.nan)).fillna(0.5)
    tmn_vol = np.log1p(df["volume"] * c)
    out["vol_z"] = ind.zscore(tmn_vol, 7 * bpd)
    signed = np.sign(out["ret_1"].fillna(0)) * df["volume"]
    out["flow_24"] = signed.rolling(24).sum() / df["volume"].rolling(24).sum().replace(0, np.nan)
    minutes_per_bar = 1440 / bpd
    out["active_frac"] = df.get("active_min", pd.Series(minutes_per_bar, index=df.index)) / minutes_per_bar
    out["stale"] = (out["active_frac"] < 0.1).astype(int)

    # regime + rule signals (used as features and as sanity checks, not as gates)
    regime = np.where(
        out["atr_rank"] > 0.85, REGIME_HIGHVOL, np.where(out["adx"] > 25, REGIME_TRENDING, REGIME_RANGING)
    )
    out["regime"] = regime
    out["sig_pullback"] = (
        (e20 > e50) & (e50 > e200) & (out["dist_ema20"].abs() < 0.5 * out["atr_pct"]) & (out["rsi"] < 55)
    ).astype(int)
    out["sig_meanrev"] = ((out["bb_z"] < -2) & (out["rsi"] < 30) & (regime == REGIME_RANGING)).astype(int)

    hour = (df.index.hour + df.index.minute / 60 + TEHRAN_OFFSET_H) % 24
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow"] = ((df.index + pd.Timedelta(hours=TEHRAN_OFFSET_H)).dayofweek).astype(int)
    return out


def btc_key(bars) -> str | None:
    """BTC leg used as market context: BTCTMN for the Toman universe, BTCUSDT for a USDT-only universe."""
    return "BTCTMN" if "BTCTMN" in bars else ("BTCUSDT" if "BTCUSDT" in bars else None)


def _context(bars: dict[str, pd.DataFrame]) -> pd.DataFrame:
    ctx = {}
    btc = btc_key(bars)
    if btc:
        lc = np.log(bars[btc]["close"])
        ctx["btc_ret_1"] = lc.diff(1)
        ctx["btc_ret_6"] = lc.diff(6)
        ctx["btc_ret_24"] = lc.diff(24)
        ctx["btc_ret_72"] = lc.diff(72)
        ctx["btc_rvol_24"] = lc.diff().rolling(24).std()
        # market-trend context (used by the trend filter, not model features)
        c = bars[btc]["close"]
        e50, e200 = ind.ema(c, 50), ind.ema(c, 200)
        ctx["btc_dist_ema50"] = c / e50 - 1
        ctx["btc_dist_ema200"] = c / e200 - 1
        ctx["btc_slope_ema50"] = e50.pct_change(6)
    if "USDTTMN" in bars:
        lc = np.log(bars["USDTTMN"]["close"])
        ctx["usdt_ret_1"] = lc.diff(1)
        ctx["usdt_ret_24"] = lc.diff(24)
    return pd.DataFrame(ctx)


def _asof_daily(index: pd.DatetimeIndex, daily: pd.DataFrame, col: str, lag_days: int = 1) -> pd.Series:
    """Daily value known at each bar. Lagged one day so we never use a value published later."""
    d = daily[["ts", col]].copy()
    d["ts"] = (d["ts"] + pd.Timedelta(days=lag_days)).astype("datetime64[ns, UTC]")
    left = pd.DataFrame({"t": index.astype("datetime64[ns, UTC]")})
    m = pd.merge_asof(left.sort_values("t"), d.rename(columns={"ts": "t"}).sort_values("t"), on="t")
    return pd.Series(m[col].to_numpy(), index=index)


def _outside(f: pd.DataFrame, b: pd.DataFrame, bn: pd.DataFrame, usdt: pd.DataFrame | None, bpd: int,
             usd_quoted: bool = False) -> None:
    """Adds Binance-based columns to the feature frame f (in place). Everything is aligned on the bar
    open time and uses only data up to that bar's close."""
    bn = bn.reindex(f.index)
    lc = np.log(bn["close"])
    for k in (1, 6, 24):
        f[f"bn_ret_{k}"] = lc.diff(k)
    f["bn_flow_1"] = bn["taker_buy"] / bn["volume"].replace(0, np.nan) - 0.5
    f["bn_flow_6"] = bn["taker_buy"].rolling(6).sum() / bn["volume"].rolling(6).sum().replace(0, np.nan) - 0.5
    f["bn_vol_z"] = ind.zscore(np.log1p(bn["quote_volume"]), 7 * bpd)
    for k in (4, 8, 12):
        f[f"bn_flow_{k}"] = bn["taker_buy"].rolling(k).sum() / bn["volume"].rolling(k).sum().replace(0, np.nan) - 0.5
    f["bn_trades_z"] = ind.zscore(np.log1p(bn["trades"]), 7 * bpd) if "trades" in bn else np.nan
    if usd_quoted or usdt is not None:
        ref = bn["close"] if usd_quoted else bn["close"] * usdt["close"].reindex(f.index)
        raw = np.log(b["close"] / ref)
        f["glob_prem"] = raw - raw.rolling(7 * bpd, min_periods=2 * bpd).median().shift(1)
        f["glob_prem_chg_6"] = f["glob_prem"].diff(6)
    else:
        f["glob_prem"] = f["glob_prem_chg_6"] = np.nan
    f["lag_gap_1"] = f["bn_ret_1"] - f["ret_1"]
    f["lag_gap_6"] = f["bn_ret_6"] - f["ret_6"]


def build_features(
    bars: dict[str, pd.DataFrame],
    fng: pd.DataFrame | None = None,
    funding: dict[str, pd.DataFrame] | None = None,
    binance1h: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    ctx = _context(bars)
    btc = btc_key(bars)
    out = {}
    for sym, b in bars.items():
        if len(b) < 50:
            continue
        f = pair_features(b)
        f = f.join(ctx, how="left")
        f["rvol_ratio"] = f["ret_1"].rolling(6).std() / f["ret_1"].rolling(72).std().replace(0, np.nan)
        for col in ("btc_ret_1", "btc_ret_6", "btc_ret_24", "btc_rvol_24", "usdt_ret_1", "usdt_ret_24"):
            if col not in f:
                f[col] = 0.0
        if sym == btc:
            f["rel_btc_24"] = 0.0
            f["rel_btc_72"] = 0.0
            f["rel_btc_1"] = f["rel_btc_6"] = 0.0
        else:
            f["rel_btc_1"] = f["ret_1"] - f["btc_ret_1"]
            f["rel_btc_6"] = f["ret_6"] - f["btc_ret_6"]
            f["rel_btc_24"] = f["ret_24"] - f["btc_ret_24"]
            f["rel_btc_72"] = f["ret_72"] - f.get("btc_ret_72", 0.0)
        f = f.drop(columns=[c for c in ("btc_ret_72",) if c in f])
        if fng is not None and len(fng):
            f["fng"] = _asof_daily(f.index, fng, "fng") / 100.0
            f["fng_chg_7"] = f["fng"] - f["fng"].shift(7 * _bars_per_day(f.index))
        else:
            f["fng"] = np.nan
            f["fng_chg_7"] = np.nan
        if funding and sym in funding and funding[sym] is not None and len(funding[sym]):
            fr = funding[sym].sort_values("ts")
            fr["ts"] = fr["ts"].astype("datetime64[ns, UTC]")
            left = pd.DataFrame({"t": f.index.astype("datetime64[ns, UTC]")})
            m = pd.merge_asof(left, fr.rename(columns={"ts": "t"}), on="t")
            f["funding"] = m["funding"].to_numpy()
            f["funding_avg_3d"] = f["funding"].rolling(3 * _bars_per_day(f.index), min_periods=1).mean()
            f["funding_chg_24"] = f["funding"] - f["funding"].shift(_bars_per_day(f.index))
        elif funding is not None:  # no futures market for this coin (e.g. XAUT): missing, not a crash
            f["funding"] = f["funding_avg_3d"] = f["funding_chg_24"] = np.nan
        if binance1h is not None:
            if sym in binance1h and binance1h[sym] is not None and len(binance1h[sym]):
                _outside(f, b, binance1h[sym], bars.get("USDTTMN"), _bars_per_day(f.index), sym.endswith("USDT"))
            else:
                for c in OUTSIDE_FEATURES + ["bn_flow_4", "bn_flow_8", "bn_flow_12", "bn_trades_z"]:
                    f[c] = np.nan
        base = sym.removesuffix("TMN")
        usdt_pair = f"{base}USDT"
        if usdt_pair in bars and "USDTTMN" in bars and sym != "USDTTMN":
            fair = bars[usdt_pair]["close"] * bars["USDTTMN"]["close"]
            prem = (b["close"] / fair.reindex(b.index) - 1).replace([np.inf, -np.inf], np.nan)
            f["prem"] = prem
            f["prem_z"] = ind.zscore(prem, 72)
            f["prem_chg_3"] = prem - prem.shift(3)
        else:
            for c in PREMIUM_FEATURES:
                f[c] = np.nan
        f["close"] = b["close"]
        out[sym] = f
    # market breadth across all pairs (trend filter context): share above their EMA50, median 24h return
    alts = [f for s_, f in out.items() if s_ != "USDTTMN"]
    if alts:
        above = pd.concat([(f["dist_ema50"] > 0).astype(float).rename(i) for i, f in enumerate(alts)], axis=1)
        r24 = pd.concat([f["ret_24"].rename(i) for i, f in enumerate(alts)], axis=1)
        breadth, med24 = above.mean(axis=1), r24.median(axis=1)
        for f in out.values():
            f["mkt_breadth_ema50"] = breadth.reindex(f.index)
            f["mkt_ret_24"] = med24.reindex(f.index)
        # cross-sectional momentum: each coin's rank (0..1) among all coins at the same hour
        for k in (6, 24):
            r = pd.concat({s_: f[f"ret_{k}"] for s_, f in out.items() if s_ != "USDTTMN"}, axis=1)
            rank = r.rank(axis=1, pct=True)
            for s_, f in out.items():
                f[f"ret_rank_{k}"] = rank[s_].reindex(f.index) if s_ in rank else np.nan
    return out


def feature_list(use_funding: bool, use_premium: bool = False, exclude: list[str] | None = None,
                 use_outside: bool = False, use_extra: bool = False) -> list[str]:
    feats = (FEATURES + (FUNDING_FEATURES if use_funding else []) + (PREMIUM_FEATURES if use_premium else [])
             + (OUTSIDE_FEATURES if use_outside else []) + (EXTRA_FEATURES if use_extra else []))
    return [f for f in feats if f not in set(exclude or [])]


def usdt_pairs(symbols: list[str]) -> list[str]:
    """Wallex USDT-quoted counterparts used for the premium feature."""
    return [s.removesuffix("TMN") + "USDT" for s in symbols if s.endswith("TMN") and s not in ("USDTTMN",)]
