"""Shadow and live trading loop (same code; only the broker differs).

Every `execution.poll_interval_s`:
  - read bid/ask for all pairs (one /v1/markets call), check exits (SL/TP/trailing/time), flatten rules
At every bar close (+20s):
  - update 1-minute candles incrementally, rebuild features exactly as in training
  - strategy: entry model -> EV ranking -> GPT overlay -> risk sizing -> broker
  - GPT exit review on open positions (news), refresh headlines
State is saved to state.json after every change so a restart resumes open positions.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor

import httpx
import pandas as pd

from engine.agent.llm import from_config as llm_from_config
from engine.agent.overlay import AgentOverlay
from engine.broker.brokers import ShadowBroker, WallexBroker
from engine.config import config_hash, universe
from engine.data import store
from engine.data import binance
from engine.data.download import download_symbol
from engine.decision_log import DecisionLog
from engine.features.build import build_features, usdt_pairs
from engine.model.predict import EntryModel
from engine.risk.manager import RiskManager, binance_guard, exit_check
from engine.risk.portfolio import Portfolio, Position
from engine.sentiment import fear_greed, funding
from engine.sentiment.events import EventGate, base_of
from engine.sentiment.news import NewsMonitor
from engine.strategy.core import Strategy
from engine.wallex import markets as mk
from engine.wallex.client import WallexClient, WallexError

log = logging.getLogger(__name__)
BAR = {"15m": pd.Timedelta(minutes=15), "30m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}
LIVE_HISTORY_DAYS = 45


class LiveRunner:
    def __init__(self, cfg, mode: str, resume: str | None = None):
        assert mode in ("shadow", "live")
        self.cfg, self.mode = cfg, mode
        self.client = WallexClient.from_config(cfg)
        self.markets = mk.load(cfg, self.client, refresh=True)
        self.model = EntryModel.load(Path(cfg.model.dir) / "latest.joblib")
        self.symbols = [s for s in universe(cfg) if s in self.model.bundle["symbols"]]
        self.data_syms = sorted(set(self.symbols) | set(cfg.universe.context_pairs))
        if cfg.universe.get("usdt_premium"):
            self.data_syms = sorted(set(self.data_syms) | set(usdt_pairs(self.symbols)))
        self.bar = BAR[cfg.data.bar]

        if resume:
            self.run_dir = Path(resume)
            self.run_id = self.run_dir.name
        else:
            self.run_id = f"{mode}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            self.run_dir = Path(cfg.run.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.dlog = DecisionLog(self.run_dir, self.run_id, mode)

        llm = None
        if cfg.agent.enabled:
            llm = llm_from_config(cfg)
            if not llm.api_key:
                log.warning("agent.enabled but OPENAI_API_KEY missing -> agent disabled")
                llm = None
        # event gate (Trump / Fed / Binance / news, risk-only) replaces the plain headline monitor when enabled
        ev = (cfg.sentiment.get("events") or {})
        self.news = EventGate(cfg, llm, self.dlog, self.symbols) if ev.get("enabled") else NewsMonitor(cfg, llm, self.dlog)
        self.agent = AgentOverlay(cfg, llm, self.news)

        now = pd.Timestamp.now(tz="UTC")
        self.risk = RiskManager(cfg)
        start = pd.Timestamp(cfg.challenge.start) if cfg.challenge.start else now
        end = pd.Timestamp(cfg.challenge.end) if cfg.challenge.end else None
        self.risk.set_window(start, end)
        self.strategy = Strategy(cfg, self.model, self.risk, self.agent, self.dlog, self.markets, self.symbols)
        self.broker = WallexBroker(self.client, cfg, self.markets) if mode == "live" else ShadowBroker(self.client, cfg, self.markets)

        self.pf = Portfolio(cash=float(cfg.risk.initial_equity_tmn), initial=float(cfg.risk.initial_equity_tmn))
        # USDT-pair mode (user rule): trade XUSDT pairs, hold USDT when idle, Toman only at start/end
        self.usd_quoted = bool(self.symbols) and all(s.endswith("USDT") for s in self.symbols)
        self.converted_back = False
        if mode == "live":
            tmn = self.broker.free_balance("TMN")
            if tmn is None:
                raise RuntimeError("cannot read balances: check WALLEX_API_KEY and verify-api")
            if self.usd_quoted:
                if tmn >= 150_000 and not resume:
                    self._convert_start(tmn)
                usdt = self.broker.free_balance("USDT") or 0.0
                self.pf.cash = usdt
                if not resume:
                    self.pf.initial = self.pf.peak_equity = self.pf.day_start_equity = usdt
                log.info(f"live USDT balance: {usdt:,.2f} (Toman left: {self.broker.free_balance('TMN') or 0:,.0f})")
            else:
                self.pf.cash = tmn
                log.info(f"live TMN balance: {tmn:,.0f}")
        elif self.usd_quoted and not resume:
            self._shadow_convert(start=True)
        self._bhttp = httpx.Client(timeout=8, limits=httpx.Limits(max_connections=12, max_keepalive_connections=12))
        self._fast = WallexClient(cfg.wallex.base_url, None, cfg.wallex.timeout_s, cfg.wallex.max_retries, 0.0)  # parallel public reads
        self._pool = ThreadPoolExecutor(8)
        self._bn, self._fund, self._prefetched_at, self._feature_latency = {}, {}, 0.0, 0.0
        self._bno: dict[str, pd.DataFrame] = {}  # Binance hourly OHLCV (data.feature_source: binance)
        self.last_bar: pd.Timestamp | None = None
        self.errors = 0
        self.fng: pd.DataFrame | None = None
        self.fng_at = 0.0
        if resume:
            self._load_state()
        meta = {"run_id": self.run_id, "mode": mode, "model_version": self.model.version,
                "symbols": self.symbols, "config_hash": config_hash(cfg), "config": cfg,
                "agent": self.agent.enabled, "window": [start.isoformat(), end.isoformat() if end else None],
                "started_at": now.isoformat()}
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
        self._save_state()  # a restart right after start-up resumes this run (deploy/start.py)

    # ---------------------------------------------------------------- state
    def _save_state(self) -> None:
        state = {
            "cash": self.pf.cash, "initial": self.pf.initial, "fees_paid": self.pf.fees_paid,
            "peak_equity": self.pf.peak_equity, "day_start_equity": self.pf.day_start_equity,
            "current_day": self.pf.current_day, "halted": self.risk.halted, "halt_reason": self.risk.halt_reason,
            "day_halt": self.risk.day_halt,
            "positions": {s: {k: (v.isoformat() if isinstance(v, pd.Timestamp) else v) for k, v in asdict(p).items()}
                          for s, p in self.pf.positions.items()},
            "last_loss_ts": {s: t.isoformat() for s, t in self.pf.last_loss_ts.items()},
        }
        tmp = self.run_dir / "state.json.tmp"
        tmp.write_text(json.dumps(state, indent=1))
        tmp.replace(self.run_dir / "state.json")
        pd.DataFrame([asdict(t) for t in self.pf.trades]).to_csv(self.run_dir / "trades.csv", index=False,
                                                                 mode="w")

    def _load_state(self) -> None:
        p = self.run_dir / "state.json"
        if not p.exists():
            return
        s = json.loads(p.read_text())
        self.pf.cash, self.pf.initial, self.pf.fees_paid = s["cash"], s["initial"], s["fees_paid"]
        self.pf.peak_equity, self.pf.day_start_equity, self.pf.current_day = s["peak_equity"], s["day_start_equity"], s["current_day"]
        self.risk.halted, self.risk.halt_reason = s["halted"], s["halt_reason"]
        self.risk.day_halt = s.get("day_halt", "")
        for sym, d in s["positions"].items():
            for k in ("entry_ts", "time_stop_ts"):
                d[k] = pd.Timestamp(d[k])
            self.pf.positions[sym] = Position(**d)
        self.pf.last_loss_ts = {k: pd.Timestamp(v) for k, v in s.get("last_loss_ts", {}).items()}
        tp = self.run_dir / "trades.csv"
        if tp.exists():
            from engine.risk.portfolio import Trade
            try:
                self.pf.trades = [Trade(**r) for r in pd.read_csv(tp).to_dict("records")]
            except pd.errors.EmptyDataError:  # run without closed trades yet
                self.pf.trades = []
        log.info(f"resumed: {len(self.pf.positions)} open positions, cash {self.pf.cash:,.0f}")

    def _outside_missing(self, bar_open: pd.Timestamp) -> list[str]:
        """Pairs whose Binance candle for `bar_open` is missing, when the model needs outside data and more than
        half of the pairs lack it (a few missing coins are normal: e.g. USDT has no Binance pair)."""
        if not (self.cfg.get("outside") or {}).get("binance") or self.cfg.data.get("feature_source") == "binance":
            return []
        from engine.data import binance_hourly

        need = [s for s in self.symbols if binance_hourly.spot_symbol(s)]
        miss = [s for s in need if self._bn.get(s) is None or bar_open not in self._bn[s].index]
        return miss if need and len(miss) > len(need) / 2 else []

    def _entry_reviews(self) -> dict[str, dict]:
        """Agent review recorded with each entry signal of this run (for reflection)."""
        out = {}
        p = self.run_dir / "decisions.jsonl"
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if '"entry_signal"' in line:
                    d = json.loads(line)
                    out[d["decision_id"]] = d.get("agent") or {}
        return out

    # ---------------------------------------------------------------- market data
    def _quotes(self, symbols) -> dict[str, dict]:
        """Best bid/ask from the REAL order book (/v1/depth). /v1/markets bid/ask is a cached summary
        that can lag the book by minutes (it once triggered a false stop-loss), so it is never used."""
        out = {}
        for s in symbols:
            try:
                book = self.client.depth(s)
                out[s] = {"bid": float(book["bid"][0]["price"]), "ask": float(book["ask"][0]["price"])}
            except (WallexError, KeyError, IndexError, TypeError, ValueError) as e:
                log.warning("depth %s failed: %s", s, e)
        return out

    # ---------------------------------------------------------------- fast data path
    # The outside-data model's edge decays within minutes of the hour close (STRATEGY.md §7.1i:
    # +13.3% at 0 min, +11.4% at 1 min, +6.3% at 3 min delay on the held-out month). So almost
    # everything is fetched ~90s BEFORE the close, and only the newest minutes right after it, in parallel.
    def _prefetch(self) -> None:
        t0 = time.time()
        list(self._pool.map(lambda s: download_symbol(self._fast, self.cfg.data.dir, s, LIVE_HISTORY_DAYS), self.data_syms))
        if self.cfg.sentiment.fear_greed and time.time() - self.fng_at > 3600:
            try:
                self.fng = fear_greed.fetch(60)
                self.fng_at = time.time()
            except Exception as e:
                log.warning("fear&greed fetch failed: %s", e)
        if (self.cfg.get("outside") or {}).get("binance"):
            from engine.data import binance_hourly

            def one(s):
                try:
                    return s, binance_hourly.recent(self._bhttp, s, LIVE_HISTORY_DAYS * 24, self.cfg.data.bar)
                except httpx.HTTPError as e:
                    log.warning("binance 1h %s failed: %s", s, e)
                    return s, self._bn.get(s)
            self._bn = dict(self._pool.map(one, self.symbols))
        if self.cfg.sentiment.funding:
            end = int(time.time() * 1000)

            def onef(s):
                fs = funding.binance_symbol(s)
                try:
                    return s, funding.fetch(fs, end - 5 * 86_400_000, end) if fs else None
                except httpx.HTTPError as e:
                    log.warning("funding %s failed: %s", fs, e)
                    return s, self._fund.get(s)
            self._fund = dict(self._pool.map(onef, self.symbols))
        if self.cfg.data.get("feature_source") == "binance":  # candidate C: the model reads Binance candles
            from engine.data import binance_ohlc

            def oneb(s):
                try:
                    return s, binance_ohlc.recent(self._bhttp, s, LIVE_HISTORY_DAYS * 24)
                except httpx.HTTPError as e:
                    log.warning("binance ohlc %s failed: %s", s, e)
                    return s, self._bno.get(s)
            self._bno = dict(self._pool.map(oneb, [x for x in self.data_syms if x != "USDTTMN"]))
        self._prefetched_at = time.time()
        log.info("prefetch done in %.1fs", time.time() - t0)

    def _features_now(self, bar_open: pd.Timestamp) -> pd.DataFrame:
        t0 = time.time()
        if time.time() - self._prefetched_at > 600:  # missed the prefetch (start-up / errors)
            self._prefetch()
        # newest Wallex minutes (incremental: one request per pair), in parallel
        list(self._pool.map(lambda s: download_symbol(self._fast, self.cfg.data.dir, s, LIVE_HISTORY_DAYS), self.data_syms))
        if (self.cfg.get("outside") or {}).get("binance"):
            from engine.data import binance_hourly

            def newest(s):
                b = binance_hourly.spot_symbol(s)
                old = self._bn.get(s)
                if not b or old is None:
                    return s, old
                try:
                    new = binance_hourly.fetch(self._bhttp, b, int((time.time() - 3 * 3600) * 1000), interval=self.cfg.data.bar)
                except httpx.HTTPError:
                    return s, old
                new.index = pd.to_datetime(new["ts"], unit="s", utc=True)
                return s, pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            self._bn = dict(self._pool.map(newest, self.symbols))
        cutoff = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=LIVE_HISTORY_DAYS)).timestamp())
        bars = {}
        for s in self.data_syms:
            m = store.load_1m(self.cfg.data.dir, s)
            bars[s] = store.resample(m[m["ts"] >= cutoff], self.cfg.data.bar)
        bn = self._bn if (self.cfg.get("outside") or {}).get("binance") else None
        if self.cfg.data.get("feature_source") == "binance":
            from engine.data import binance_ohlc

            def newest_ohlc(s):
                old = self._bno.get(s)
                try:
                    new = binance_ohlc.to_bars(binance_ohlc.fetch(self._bhttp, s, int((time.time() - 3 * 3600) * 1000)))
                except httpx.HTTPError:
                    return s, old
                return s, new if old is None or old.empty else pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            self._bno = dict(self._pool.map(newest_ohlc, list(self._bno)))
            bars = {**{s: b for s, b in self._bno.items() if b is not None and len(b)},
                    **({"USDTTMN": bars["USDTTMN"]} if "USDTTMN" in bars else {})}
            bn = {s: bars[s] for s in self.symbols if s in bars} if bn is not None else None
        fund = self._fund if self.cfg.sentiment.funding else None
        feats = build_features(bars, self.fng, fund, bn)
        if self.model.bundle.get("sequence"):  # GRU only: PyTorch is loaded only when needed
            from engine.model.seq import add_seq_proba

            add_seq_proba(self.model.bundle, feats)
        rows = {s: f.loc[bar_open] for s, f in feats.items() if bar_open in f.index}
        self._feature_latency = time.time() - t0
        return pd.DataFrame(rows).T.apply(pd.to_numeric, errors="coerce")

    # ---------------------------------------------------------------- actions
    def _exit(self, sym: str, reason: str, ts: pd.Timestamp) -> None:
        pos = self.pf.positions[sym]
        qty = pos.qty
        if self.mode == "live":
            free = self.broker.free_balance(base_of(sym))  # BTCUSDT -> BTC, BTCTMN -> BTC
            if free is not None:
                qty = min(qty, free)  # buy fees may be taken in the base asset
        fill = self.broker.sell(sym, qty)
        if fill is None:
            self.dlog.write("exit_failed", ts, entry_decision_id=pos.decision_id, symbol=sym, reason=reason)
            self.errors += 1
            return
        if fill.qty < pos.qty * 0.999:
            log.warning("%s: sold %s of %s (rest is dust or failed)", sym, fill.qty, pos.qty)
            pos.qty = fill.qty
        t = self.pf.close(sym, fill.price, fill.fee_tmn, fill.ts, reason)
        self.dlog.write("exit", fill.ts, decision_id=pos.decision_id + "-x", entry_decision_id=pos.decision_id,
                        symbol=sym, reason=reason, exit_price=fill.price, qty=fill.qty, orders=fill.orders,
                        pnl_tmn=t.pnl_tmn, ret_pct=t.ret_pct, hold_hours=t.hold_hours)
        self._save_state()

    # ---------------------------------------------------------------- USDT <-> Toman (USDT-pair mode)
    def _convert_start(self, tmn: float) -> None:
        """Start of the run: Toman -> USDT through USDT/TMN (one conversion for the whole challenge)."""
        fill = self.broker.buy("USDTTMN", tmn * 0.995)
        self.dlog.write("convert", pd.Timestamp.now(tz="UTC"), direction="TMN->USDT", tmn=tmn,
                        filled_usdt=fill.qty if fill else 0.0, price=fill.price if fill else None, orders=fill.orders if fill else [])
        log.info("start conversion TMN->USDT: %s", f"{fill.qty:,.2f} USDT @ {fill.price:,.0f}" if fill else "FAILED")

    def _shadow_convert(self, start: bool) -> None:
        """Shadow USDT-pair mode: simulate the Toman <-> USDT conversions at the live USDT/TMN book
        (taker fee of a Toman market), so shadow results include their cost and the rial move."""
        q = self._quotes(["USDTTMN"]).get("USDTTMN")
        fee = float(self.cfg.costs.tmn_taker_fee)
        now = pd.Timestamp.now(tz="UTC")
        if q is None:
            log.warning("shadow conversion: USDT/TMN book unavailable")
            return
        if start:
            tmn = float(self.cfg.risk.initial_equity_tmn)
            usdt = tmn * (1 - fee) / q["ask"]
            self.pf.cash = self.pf.initial = self.pf.peak_equity = self.pf.day_start_equity = usdt
            self.dlog.write("convert", now, direction="TMN->USDT", simulated=True, tmn=tmn, filled_usdt=usdt, price=q["ask"])
            log.info(f"shadow start: {tmn:,.0f} TMN -> {usdt:,.4f} USDT @ {q['ask']:,.0f}")
        else:
            usdt = self.pf.cash
            tmn = usdt * q["bid"] * (1 - fee)
            self.dlog.write("convert", now, direction="USDT->TMN", simulated=True, usdt=usdt, tmn=tmn, price=q["bid"],
                            ret_tmn_pct=round(100 * (tmn / float(self.cfg.risk.initial_equity_tmn) - 1), 3))
            log.info(f"shadow end: {usdt:,.4f} USDT -> {tmn:,.0f} TMN @ {q['bid']:,.0f}")

    def _convert_back(self, now: pd.Timestamp) -> None:
        """Challenge end: all USDT -> Toman. If USDT/TMN does not fill (it froze for ~14h on 2026-10-21),
        route through BTC: buy BTC with USDT on BTCUSDT, then sell that BTC on BTCTMN."""
        if self.converted_back:
            return
        if self.mode != "live":
            self._shadow_convert(start=False)
            self.converted_back = True
            return
        usdt = self.broker.free_balance("USDT") or 0.0
        if usdt * 270_000 < 100_000:  # dust: nothing meaningful left to convert
            self.converted_back = True
            return
        fill = self.broker.sell("USDTTMN", usdt)
        self.dlog.write("convert", now, direction="USDT->TMN", usdt=usdt, filled=fill.qty if fill else 0.0,
                        price=fill.price if fill else None, orders=fill.orders if fill else [])
        left = self.broker.free_balance("USDT") or 0.0
        if left * 270_000 >= 100_000:
            log.warning("USDT/TMN did not fill %.2f USDT -> fallback route via BTC", left)
            b = self.broker.buy("BTCUSDT", left * 0.995)
            if b is not None:
                btc = self.broker.free_balance("BTC") or b.qty
                s = self.broker.sell("BTCTMN", btc)
                self.dlog.write("convert", now, direction="USDT->BTC->TMN", usdt=left, btc=btc,
                                tmn_price=s.price if s else None, orders=(b.orders + (s.orders if s else [])))
            left = self.broker.free_balance("USDT") or 0.0
        self.converted_back = left * 270_000 < 100_000
        log.info("end conversion done: %s USDT left, Toman %s", f"{left:,.2f}", f"{self.broker.free_balance('TMN') or 0:,.0f}")

    def _check_exits(self, quotes: dict, now: pd.Timestamp) -> None:
        flatten = self.risk.must_flatten(now)
        at_end = self.risk.end_ts is not None and now >= self.risk.end_ts - pd.Timedelta(minutes=self.cfg.challenge.flatten_before_end_min)
        if (at_end or self.risk.halted) and self.usd_quoted and not self.pf.positions and not self.converted_back:
            self._convert_back(now)
        for sym in list(self.pf.positions):
            q = quotes.get(sym)
            if q is None:
                continue
            if flatten:
                self._exit(sym, self.risk.flatten_reason(now), now)
                continue
            pos = self.pf.positions[sym]
            g = self.cfg.risk.get("binance_guard") or {}
            if g.get("enabled"):
                bsym = binance.binance_symbol(sym)
                r3 = binance.live_ret_3m(self._bhttp, bsym) if bsym else None
                sl_before = pos.sl_price
                if binance_guard(pos, r3, q["bid"], g) == "exit":
                    self.dlog.write("binance_guard", now, symbol=sym, binance_symbol=bsym, ret_3m=r3, action="exit")
                    self._exit(sym, "binance_guard", now)
                    continue
                if pos.sl_price != sl_before:
                    self.dlog.write("binance_guard", now, symbol=sym, binance_symbol=bsym, ret_3m=r3, action="tighten",
                                    stop_before=sl_before, stop_after=pos.sl_price)
                    self._save_state()
            hit = exit_check(pos, q["bid"], q["bid"], now, self.cfg.risk.trailing_atr)
            if hit:
                self._exit(sym, hit[0], now)

    def _on_bar(self, bar_open: pd.Timestamp, quotes: dict) -> None:
        close_ts = bar_open + self.bar
        rows = self._features_now(bar_open)
        prices = {s: q["bid"] for s, q in quotes.items()}
        eq = self.pf.mark(close_ts, prices)
        for ev in self.risk.update(self.pf, eq):
            self.dlog.write("risk_event", close_ts, event=ev, reason=self.risk.halt_reason, equity=eq)
        if rows.empty:
            log.warning("no feature rows for %s", bar_open)
            return
        market = {
            "btc_ret_24h_pct": next((round(float(rows.loc[b, "ret_24"]) * 100, 2) for b in ("BTCTMN", "BTCUSDT")
                                     if b in rows.index), None),
            "usdt_tmn_ret_24h_pct": round(float(rows.loc["USDTTMN", "ret_24"]) * 100, 2) if "USDTTMN" in rows.index else None,
            "fear_greed": float(rows["fng"].iloc[0] * 100) if pd.notna(rows["fng"].iloc[0]) else None,
            "news_market_score": round(self.news.market_score(), 2),
        }
        if isinstance(self.news, EventGate):
            market["event_gate"] = self.news.status()
        intents = self.strategy.decide_entries(close_ts, rows, self.pf, prices, market)
        late = pd.Timestamp.now(tz="UTC") - close_ts
        if intents and late > pd.Timedelta(minutes=10):  # e.g. started mid-bar: signal is stale
            self.dlog.write("stale_signal_skipped", close_ts, symbols=[i.symbol for i in intents],
                            late_min=round(late.total_seconds() / 60, 1))
            intents = []
        missing = self._outside_missing(bar_open)
        if intents and missing:  # S1 without Binance data: held-out +2.33% vs +20.47% (STRATEGY.md §7.1t)
            self.dlog.write("outside_data_missing", close_ts, symbols=[i.symbol for i in intents], missing=missing)
            log.warning("Binance data missing for %d pairs -> no new entries this bar", len(missing))
            intents = []
        blocked, why = self.news.entries_blocked() if isinstance(self.news, EventGate) else (False, "")
        if intents and blocked:
            self.dlog.write("event_gate_skip", close_ts, symbols=[i.symbol for i in intents],
                            entry_decision_ids=[i.decision_id for i in intents], reason=why)
            intents = []
        for it in intents:
            fill = self.broker.buy(it.symbol, it.notional)
            if fill is None or fill.qty <= 0:
                self.dlog.write("fill_failed", pd.Timestamp.now(tz="UTC"), entry_decision_id=it.decision_id, symbol=it.symbol)
                continue
            pos = Position(it.symbol, fill.qty, fill.price, fill.ts, fill.price * (1 + it.tp_pct),
                           fill.price * (1 - it.sl_pct), it.atr_pct,
                           fill.ts + self.bar * int(self.cfg.labels.horizon_bars), it.decision_id, fill.fee_tmn,
                           invalidation=it.invalidation)
            self.pf.open(pos)
            self.dlog.write("fill", fill.ts, decision_id=it.decision_id + "-f", entry_decision_id=it.decision_id,
                            symbol=it.symbol, side="BUY", price=fill.price, qty=fill.qty,
                            notional_tmn=fill.qty * fill.price, orders=fill.orders,
                            seconds_after_close=round((fill.ts - close_ts).total_seconds(), 1))
            self._save_state()
        self.dlog.write("bar_latency", close_ts, features_s=round(self._feature_latency, 1),
                        decided_s=round((pd.Timestamp.now(tz="UTC") - close_ts).total_seconds(), 1), entries=len(intents))
        # slower housekeeping AFTER the time-critical entries: news scoring and the agent's exit review
        self.news.refresh()
        if self.pf.positions:
            held = [{"symbol": s, "entry": p.entry_price, "now": prices.get(s),
                     "pnl_pct": round((prices.get(s, p.entry_price) / p.entry_price - 1) * 100, 2),
                     "held_hours": round((close_ts - p.entry_ts).total_seconds() / 3600, 1),
                     "invalidation": p.invalidation or "(none recorded)"}
                    for s, p in self.pf.positions.items()]
            rev = self.agent.review_exits(close_ts, held)
            for sym in rev.get("exit", []):
                if sym in self.pf.positions:
                    self.dlog.write("agent_exit", close_ts, symbol=sym, reasoning=rev.get("reasoning"),
                                    invalidation=self.pf.positions[sym].invalidation,
                                    evidence=(rev.get("evidence") or {}).get(sym))
                    self._exit(sym, "agent_exit", close_ts)
        refl = self.agent.maybe_reflect(close_ts, [asdict(t) for t in self.pf.trades], self._entry_reviews())
        if refl:
            self.dlog.write("agent_reflection", close_ts, **refl)
        eq = self.pf.equity(prices)
        with open(self.run_dir / "equity.csv", "a", encoding="utf-8") as f:
            if f.tell() == 0:
                f.write("ts,equity,cash,exposure,positions,drawdown\n")
            nd = 4 if self.usd_quoted else 0  # USDT amounts need decimals (the account is ~18 USDT)
            f.write(f"{close_ts.isoformat()},{eq:.{nd}f},{self.pf.cash:.{nd}f},{self.pf.exposure(prices):.{nd}f},"
                    f"{len(self.pf.positions)},{1 - eq / self.pf.peak_equity:.6f}\n")
        log.info(f"bar {close_ts} equity {eq:,.{4 if self.usd_quoted else 0}f} ({(eq / self.pf.initial - 1) * 100:+.2f}%) "
                 f"positions {list(self.pf.positions)}")
        self._save_state()

    # ---------------------------------------------------------------- loop
    def run(self) -> None:
        log.info("%s run %s: %s, model %s, agent %s", self.mode, self.run_id, self.symbols, self.model.version,
                 self.agent.enabled)
        if isinstance(self.news, EventGate):
            self.news.start()  # background thread: polls its sources on their own schedule
        else:
            self.news.refresh(force=True)
        while True:
            t0 = time.time()
            try:
                now = pd.Timestamp.now(tz="UTC")
                quotes = self._quotes(list(self.pf.positions))  # held pairs every poll
                self._check_exits(quotes, now)
                if isinstance(self.news, EventGate):
                    for sym in self.news.take_exits(list(self.pf.positions)):
                        self._exit(sym, "event_gate", now)
                next_close = now.floor(self.bar) + self.bar
                if (next_close - now).total_seconds() < 120 and time.time() - self._prefetched_at > 300:
                    self._prefetch()  # heavy downloads before the close
                bar_open = (now - pd.Timedelta(seconds=3)).floor(self.bar) - self.bar  # last fully closed bar
                if self.last_bar is None or bar_open > self.last_bar:
                    self.last_bar = bar_open
                    self._on_bar(bar_open, self._quotes(self.data_syms))
                if (self.risk.end_ts is not None and now >= self.risk.end_ts and not self.pf.positions
                        and (not self.usd_quoted or self.converted_back)):
                    log.info("challenge window over and flat in TMN -> stopping")
                    break
                self.errors = 0
            except (WallexError, OSError, KeyError, ValueError) as e:
                self.errors += 1
                log.exception("loop error #%d: %s", self.errors, e)
                self.dlog.write("error", pd.Timestamp.now(tz="UTC"), error=str(e)[:500])
                time.sleep(min(60, 5 * self.errors))
            wait = max(1.0, self.cfg.execution.poll_interval_s - (time.time() - t0))
            to_close = ((pd.Timestamp.now(tz="UTC").floor(self.bar) + self.bar) - pd.Timestamp.now(tz="UTC")).total_seconds()
            time.sleep(min(wait, max(0.5, to_close + 3)))  # wake 3s after the hour close, not up to a poll later


def run_live(cfg, mode: str, resume: str | None = None) -> None:
    LiveRunner(cfg, mode, resume).run()
