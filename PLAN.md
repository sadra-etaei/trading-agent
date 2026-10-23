# Wallex AI Trading Engine — Plan

Plan for the Wallex challenge (Starcoach Agentic AI Hackathon): an automated, AI-driven
spot trading engine that trades a 5,000,000 TMN account for 3 days and finishes fully in Toman.

Status (Thu 2026-10-22 02:10): **Strategy 1 improved** to 16 pairs with an ensemble model (held-out month
+1.40%, 104 trades, max DD 4.15%; STRATEGY.md §7.1b). **Running overnight:** recorder (real books),
Strategy 2 shadow (8 variants), Strategy 1 shadow (16 pairs). The Strategy 2 live executor is built
and tested against a fake exchange. Waiting for the API key (morning).

Earlier status (2026-10-21): **engine built and tested end to end** (download → train → backtest → shadow).
Backtest of record is safe but nearly inactive (+0.01%, 3 trades in 35 days; see STRATEGY.md §7).
The hourly-candle edge is too thin for ~0.6% round-trip costs. Next research options:
1. Confirm the real fee on the hackathon account (it changes everything; see §12).
2. Minute-level strategy on the Wallex-internal fair value (TMN price vs USDT pair × USDTTMN) with
   maker limit orders, using the recorded order books.
3. Keep the current engine as the safe baseline and agent/risk shell either way.

Core idea (v2): **ML-first.** A calibrated ML entry model is the main decision maker
(what to trade, when, how much via expected value). A GPT agent is an overlay that reads
news/sentiment (incl. Persian/Iranian macro news) and may veto, reduce, exit early or pause —
it never opens trades on its own. A deterministic risk manager has the final word.

---

## 1. Goals and what the judges reward

| Area | Points | What it means for us |
|---|---|---|
| Net return % (rank) | 90 | Must finish **positive**; a losing team gets 0 here |
| Loss control (drawdown) | 30 | Low intra-challenge drawdown → steady, small edges over big bets |
| Algorithm activity | 15 | Regular, sensible trading over all 3 days (not 1 lucky trade, not churn) |
| Execution stability | 15 | Few errors, correct orders, no crashes |
| Strategy logic | 40 | Every buy/sell explainable from defined rules |
| Real use of AI | 35 | AI actually decides; not a label on if/else |
| Risk & money management | 30 | Order size caps, stop-loss, never all-in |
| Reproducibility | 25 | Clean code, run guide, judges can re-run it |
| Backtest + shadow test | 20 | Done and results shown |

Hard rules we must respect:
- Spot crypto only. Spot = **long-only** (buy, then sell to exit; no shorting).
- Any spot pair is allowed **except the organizers' blocklist** (`blocked_markets.txt`, 345 pairs).
  Enforced when building the universe and again in the broker before every order.
- All trades placed by the algorithm; no manual trades.
- No self-trading, wash trading, or coordinated trades.
- **Final value must be realized in TMN** — all coin positions closed before the end.
- Every trade must have a stored decision log (signal seen, why bought/sold).
- Submitted source code must be exactly the code that traded.

Design consequence: fees (~0.4–0.5% round trip, **to be confirmed**) are the main enemy.
Prefer limit (maker) orders; only take trades whose expected move clearly exceeds fees.

---

## 2. Data

### 2.1 Verified facts about Wallex public data (no API key needed)
- `GET /v1/markets` — all symbols with precision, `minQty`, `minNotional`, 24h stats.
  187 TMN-quoted pairs, 186 USDT-quoted pairs.
- `GET /v1/depth?symbol=X` — order book (ask/bid levels with price, quantity).
- `GET /v1/udf/history?symbol=X&resolution=R&from=T&to=T` — OHLCV candles.
  - `resolution=60` → 1h candles, history available **≥ 3 years**. ~480–700 bars per request.
  - `resolution=15` → actually returns **1-minute** candles (60s spacing), history **≥ 500 days**.
    ~4320 bars (3 days) per request.
  - `resolution=1` → returned no data.
  - → Download in paged chunks; build our own 15m / 1h candles by resampling 1-minute data.
- Some candles have volume 0 with a repeated price (quiet hours) → must be flagged.

### 2.2 Training / backtest universe
Chosen by volume and spread (snapshot; re-rank on 30-day average before freezing):

| Role | Pairs |
|---|---|
| Core (train + trade) | BTCTMN, ETHTMN, SOLTMN, XRPTMN, BNBTMN, DOGETMN, TRXTMN |
| Context feature + tradeable | USDTTMN (moves every TMN price) |
| Optional | XAUTTMN (gold token, diversifier) |
| Excluded | PEPE, PUMP, USOON, ARB, ZEC and other recent hype pairs (short history, wide/odd spreads) |

### 2.3 Datasets
1. **Main:** last ~12 months of Wallex 1-minute candles for the universe (~4–5M rows),
   resampled to 15m and 1h. Stored locally (Parquet/SQLite) with a manifest (pairs, date range, hash).
2. **Robustness (optional):** earlier 2–3 years of Wallex 1h candles for bull/bear/flat checks.
3. **Self-recorded:** order book + trade snapshots from now until the challenge.
   Used for entry timing and shadow testing, not for the main model.
4. **Fallback only:** Binance public USDT klines converted with USDTTMN — training augmentation
   only, never the backtest of record.
5. **Sentiment (with history → ML features):**
   - Crypto Fear & Greed Index (alternative.me, daily since 2018) — verified reachable.
   - Binance futures funding rates (long history) — if reachable from the run location.
   - Binance long/short ratio + open interest (~30 days free history) → shadow/live only.
6. **Sentiment (no history → agent overlay only):** crypto news headlines (RSS / CryptoPanic)
   and Persian/Iranian news (rial, sanctions, political shocks) scored by GPT.

### 2.4 Splits (no look-ahead)
- Months 1–9: walk-forward training.
- Months 10–11: validation / threshold tuning.
- Last 4–6 weeks: **final held-out test**, run once → official backtest report.
- Purge gap between train and test = label horizon (labels overlap).

---

## 3. ML models

### 3.1 Entry model (primary decision maker)

It scores every pair at every bar; candidates are ranked by expected value:

`EV = p_win × TP% − (1 − p_win) × SL% − fees%`

Top-EV candidates above threshold are traded (up to the position cap); size grows with EV/p_win
within risk limits. Rule signals (trend / mean-reversion) are **features and sanity checks**, not gates.
If walk-forward shows ML does not beat plain rules after fees, fall back to rules + ML filter.

**Question it answers:** "If we buy this pair at the next bar's open, will it hit take-profit
before stop-loss, net of fees, within N bars?"

- **Row:** one pair at one 15m/1h bar close; all pairs pooled in one model.
- **Label (triple barrier):** 1 if +TP (e.g. 1.5×ATR, and > round-trip fee) is hit first;
  0 if −SL (1×ATR) is hit first or time limit (e.g. 12h) expires. Same TP/SL as the live risk manager.
- **Features (past data only):**
  - returns over 1/3/6/12/24 bars
  - trend: distance from EMA20/50/200, EMA slopes, ADX
  - volatility: ATR %, realized vol, Bollinger width, vol percentile
  - momentum: RSI, MACD histogram, stochastic
  - volume: z-score, TMN volume, OBV slope, zero-volume flag
  - market context: BTC return/vol, **USDTTMN return** (Toman devaluation effect)
  - time: hour of day, day of week (Tehran time, Thu/Fri weekend)
  - regime tag (trending / ranging / high-vol)
  - rule-signal flags (EMA pullback, RSI/z-score extreme)
  - sentiment: Fear & Greed level + change, funding rate (if available)
- **Models:** logistic regression baseline → LightGBM (small, regularized), probability-calibrated.
- **Evaluation:** net profit per trade after fees in backtest vs. rules-without-ML.
  If ML doesn't beat the plain rules, drop it.
- **Versioning:** every model has a version id; each decision log entry records it.
- **Retraining:** once before the challenge; optionally daily on latest data.

### 3.2 Selection model (experiment)

Two-stage design, kept **only if** walk-forward shows it beats EV ranking after fees
(return and drawdown). Otherwise dropped and documented as tested-and-rejected.

- Wider universe (~20–30 liquid TMN pairs), runs every ~4h.
- Target: next-24h return **relative to other pairs**, net of spread (cross-sectional momentum).
- Features: relative strength vs BTC, 7/30-day momentum, predicted volatility, liquidity/spread
  trend, sentiment.
- Output: top 4–6 pairs → the entry model trades only those.

---

## 4. Strategy pipeline

Per pair, at each bar close:

1. **Features** — technicals, regime tag, rule-signal flags, market context, sentiment.
2. **(Optional) selection** — restrict to top pairs from the selection model.
3. **Entry model** — p_win → EV; rank candidates; keep EV > 0 and p_win ≥ threshold.
4. **GPT overlay** — veto / reduce / pause pair based on news and anomalies (see §5).
5. **Risk manager** — hard caps; can shrink or block, never enlarge.
6. **Entry timing** — order-book imbalance + spread check; place limit order near best bid.
7. **Exit** — TP / SL / trailing stop / time stop / agent-requested exit / end-of-challenge flatten.

---

## 5. AI / agent layer

- **Models:** OpenAI GPT models, configurable in `config.yaml`
  (cheaper/faster model for screening, stronger model for final decisions).
- **Role: overlay, not originator.** It can veto, reduce size, request early exits, and pause a
  pair or the whole market ("risk off"). It never opens a trade by itself.
- **Input to agent:** candidate trade (pair, p_win, EV, features summary), open positions, equity,
  recent P&L, market context (BTC, USDTTMN moves), recent crypto headlines and Persian/Iranian
  macro headlines.
- **Output:** strict JSON — `{action: approve|veto|reduce, size_multiplier (0–1), risk_off, confidence, reasoning}`.
- **News scoring:** headlines are periodically scored by GPT (−1…+1, with category) and logged;
  scores feed the overlay. Shadow test compares outcomes with vs. without the overlay.
- **Guardrails:** cannot exceed risk limits; invalid/timeout responses → safe default (no trade).
- **Wallex MCP:** gives the agent tools for market data, balances, and order status.
  Order *placement* goes through deterministic execution code (speed, safety, auditability).
- **Reproducibility:** every prompt + response is stored (keyed by input hash) so decisions can be
  replayed exactly. `temperature=0`.

---

## 6. Risk and money management

| Rule | Value (initial, tunable) |
|---|---|
| Max position size | 20% of equity |
| Max concurrent positions | 3 |
| Min order | respect `minNotional` / `minQty` per market |
| Stop-loss | ATR-based, placed on entry |
| Take-profit / trailing | ATR-based |
| No averaging down | always |
| Daily loss limit | −3% → no new entries for rest of day |
| Max drawdown kill switch | −6% from peak → flatten and halt |
| Ramp-up | day 1 starts at half size |
| End of challenge | no new entries in last ~3h; force-flatten to TMN in last ~45 min; verify zero coin balance |

---

## 7. Architecture

```
data/       paged history downloader, live REST poller, order book/trade recorder, local store
sentiment/  Fear & Greed, funding rates, news feeds + GPT headline scoring
features/   indicators, regime detection
signals/    rule-based candidate generators
model/      dataset building, labeling, training, calibration, inference
agent/      GPT decision layer, prompts, MCP tools, response cache
risk/       sizing, limits, kill switch, end-of-challenge flatten
execution/  order state machine, client order ids, retries, partial fills, cancel/replace
broker/     interface → BacktestBroker | ShadowBroker | WallexBroker
ledger/     positions, fills, fees, P&L, equity curve
logs/       decision log (JSONL + SQLite)
reports/    metrics + charts
cli.py      download | train | backtest | shadow | live | report
config.yaml all parameters; secrets only from env vars
```

**One code path for all modes** — only the broker changes. This is the basis of reproducibility.

Tech: Python, httpx, pandas, numpy, scikit-learn, LightGBM, pydantic, SQLite/Parquet, OpenAI SDK.
Secrets: `WALLEX_API_KEY`, `OPENAI_API_KEY` (env vars, never committed).

---

## 8. Decision log (deliverable)

One record per decision (including vetoed signals), e.g.:

```json
{
  "ts": "...", "mode": "live", "pair": "ETHTMN", "bar": "1h",
  "regime": "trending", "signal": "ema_pullback_long",
  "features": {"rsi": 41.2, "atr_pct": 1.1, "...": "..."},
  "ml": {"model_version": "lgbm-2026-10-20", "p_win": 0.63},
  "agent": {"model": "gpt-...", "action": "approve", "size_fraction": 0.6, "reasoning": "..."},
  "risk": {"approved_size_tmn": 600000, "sl": 0.0, "tp": 0.0, "checks": ["..."]},
  "order": {"client_id": "...", "type": "LIMIT", "side": "BUY", "price": 0, "qty": 0},
  "result": {"filled": true, "exit_reason": "tp", "pnl_tmn": 0, "fees_tmn": 0}
}
```

---

## 9. Testing

- **Unit tests:** indicators, labeling, risk rules, order state machine, fee math.
- **Backtest:** event-driven over 1-minute data, decisions on 15m/1h closes, fees + slippage +
  `minNotional` modeled. Reports: return, max drawdown, trade count, volume, avg trade size,
  win rate, profit factor, Sharpe; per-pair and per-regime breakdown.
  The LLM agent is run on a sample (cached) in backtest; full agent behavior is evaluated in shadow.
- **Shadow test:** ≥ 24h on live data with simulated fills, full pipeline incl. GPT agent.
  Report compares shadow vs. backtest expectations.
- **Live smoke test:** smallest allowed orders to verify order placement/cancel before full run.

---

## 10. Deliverables

- GitHub repo (exact code used in trading, tagged commit).
- `README.md` — setup and run guide (download → train → backtest → shadow → live → report).
- `STRATEGY.md` — architecture, AI usage, Wallex tools usage, buy/sell conditions,
  risk management, how to re-run and evaluate (English).
- Decision logs, trade history, backtest + shadow reports with charts.
- ~5-minute demo video (script prepared from the reports).

---

## 11. Timeline

| Phase | Work |
|---|---|
| Pre-challenge 1 | Data downloader + store, start order book recorder, features, labeling |
| Pre-challenge 2 | ML training + walk-forward evaluation, backtester, reports |
| Pre-challenge 3 | Agent layer, risk manager, execution + Wallex broker, shadow run |
| Day 1 | Live smoke test → live at half size, monitor |
| Day 2 | Full size if day 1 is healthy |
| Day 3 | Trading; stop entries ~3h before end, flatten to TMN, freeze logs and code tag |
| After | Final reports, docs, demo video |

---

## 12. Open items

- [ ] Wallex API key (user will provide)
- [ ] Wallex MCP endpoint/config (user will provide)
- [ ] OpenAI API key + chosen GPT model names (user will provide)
- [ ] Confirm private endpoints: place/cancel order, open orders, balances, order status, trade history
- [x] Fees confirmed (2026-10-22): **Toman markets maker 0.25% / taker 0.30%; USDT markets maker 0.10% / taker 0.13%.**
- [ ] Rate limits — public sources say 0.10–0.35% per side on
      TMN markets depending on user level; engine assumes 0.25%/side + 5 bps slippage. Critical:
      at ~0.6% round trip, hourly edges are thin.
- [ ] Wire Wallex MCP into the agent as a tool source (waiting for MCP endpoint/config)
- [ ] Confirm whether Wallex requires an Iranian IP → where the bot runs
- [ ] Exact challenge start/end time (for the flatten schedule)
- [ ] Re-rank pair universe on 30-day average volume

## 13. Known risks

- Candle-based prediction edge in crypto is small → ML is a filter, not an oracle.
- Thin liquidity on alt TMN pairs → slippage; universe restricted to liquid pairs.
- API outages / rate limits → retries, backoff, safe "no new trade" mode, alerts.
- LLM latency/errors → timeouts with safe default; deterministic risk layer always applies.

---

## 14. Strategy 2: minute-level fair-value gap with maker orders — **DROPPED (Thu 2026-10-22 11:50)**

**Decision:** dropped for live trading after ~10h of real order books. It stays in the repo as a
tested-and-rejected idea (`engine/fv/`, `cli.py fv-research | fv-backtest | fv-shadow`).
- **Gaps:** real gaps are 0.02–0.06% median, smaller than the spread. The 1-minute-candle "gaps"
  (§14.2) were mostly stale last-trade prices.
- **Lead-lag:** corr(TMN move at t+k, fair-value move at t) is only 0.02–0.07, so the TMN book
  does not follow the USDT books in a tradeable way.
- **Replay:** 2 trades at θ=0.25%, both losses. Shadow: 10 paper orders in 9.5h, 0 fills.
- **USDT/TMN:** no trades since 2026-10-21 23:30 Tehran, with the book frozen at 266,720/266,758,
  so fair value is unreliable.

Original status: plan only — not built. Recorder started 2026-10-21 23:38 (`python cli.py record`).
**Deadline: Friday 2026-10-23 23:00 (Tehran) submission; trading ends 21:00.** See §14.10 for the compressed schedule.

### 14.1 Idea

On Wallex every coin trades in two books: `XTMN` and `XUSDT`, and USDT itself trades in `USDTTMN`.
So the Toman price has a fair value implied by the other two books:

`F = mid(XUSDT) × mid(USDTTMN)`

The books have different participants and the TMN books are thinner, so the TMN price often lags
or overshoots `F` for a few minutes. When `XTMN` is **cheaper** than fair value we post a passive
(maker) buy below fair value. Once it fills, we post a maker sell back at fair value. This is
long-only, every position ends in TMN, and it uses only Wallex data, so it works from Iran.

### 14.2 Evidence so far (1-minute candles, 1 year)

`gap = log(close_TMN / (close_USDT_pair × close_USDTTMN))`, measured against its own 24h median
(a structural offset of about −0.06% to +0.08%).

| pair | typical gap | 90th pct | gaps > 0.6% per day | >0.6% gaps gone after 15 min | after 60 min |
|---|---|---|---|---|---|
| BTC | 0.12% | 0.34% | 7.9 | 50% | 76% |
| ETH | 0.15% | 0.42% | 7.3 | 58% | 87% |
| SOL | 0.16% | 0.45% | 5.8 | 51% | 89% |
| XRP | 0.15% | 0.43% | 4.8 | 45% | 89% |
| BNB | 0.17% | 0.52% | 2.5 | 15% | 74% |
| DOGE | 0.18% | 0.56% | 6.4 | 32% | 79% |
| TRX | 0.23% | 0.68% | 5.6 | 19% | 67% |
| XAUT | 0.16% | 0.45% | 8.8 | 46% | 80% |

**Caveat:** both legs traded in the same minute only 2–20% of the time, so candle closes are often
stale and part of this "gap" is a measurement artifact. The real numbers must come from recorded
bid/ask (§14.5). Treat the table as a reason to look closer, not as proof.

### 14.3 Trading rules (first version)

- **Fair value:** `F` from the USDT pair and USDTTMN mids, plus the rolling 24h median offset of the
  TMN price vs `F`. Skip a pair when either USDT-side book is stale (no change for N seconds) or
  its spread is wide.
- **Entry:** when `gap = log(bid_TMN / F) − offset ≤ −θ_in`, post a maker BUY at
  `min(best_bid + 1 tick, F·e^(offset − θ_in))`: join or improve the bid, never pay above target.
  Re-quote when `F` moves. Cancel when the gap closes before a fill. Rate-limit re-quotes.
- **Exit:** right after a fill, post a maker SELL at `F·e^(offset)` (fair value), re-quoted as `F` moves.
- **Protection:**
  - **time stop:** after H minutes (start with 30–60), switch to an aggressive limit sell;
  - **stop:** if `F` itself falls by S%, or the gap widens past `θ_stop`;
  - **disconnect:** cancel all orders on any error, disconnect or kill switch.
- **Only the "TMN is cheap" side trades.** Spot is long-only and we may not hold coins at the end. A
  "TMN is expensive" gap is used only to exit an open position early.
- **Variant (later):** the consensus USDT rate implied by BTCTMN/BTCUSDT, ETHTMN/ETHUSDT and so on.
  When USDTTMN trades below it, buy USDTTMN with maker orders. This is the same mechanics on the
  most liquid TMN book.

### 14.4 Economics

Break-even entry gap ≈ 2 × maker fee + expected adverse drift (unknown until we measure it).

| maker fee / side | round trip | entry gap needed (approx.) | gaps per day per pair (history) |
|---|---|---|---|
| 0.10% | 0.20% | ~0.35% | ~15–30 |
| 0.20% | 0.40% | ~0.55% | ~5–10 |
| 0.25% | 0.50% | ~0.70% | ~3–8 |

Across 8 pairs that is enough activity for "regular trading" even at the higher fee. **The maker
fee on the hackathon account is the single most important open number.**

### 14.5 Data and validation (no live money until each gate passes)

1. **Recording (running now):** best bid/ask/last for 15 pairs from one `/v1/markets` call every
   ~4–5s. Order book (top 10) and public trades round-robin, so each pair is refreshed about
   every 75s. Goal: 3–5+ days before the challenge.
2. **Research on recorded data:**
   - true gap distribution on bid/ask;
   - lead-lag, i.e. which leg moves first (if the TMN book leads, gaps are informed flow and the
     idea fails);
   - how often a gap closes by the TMN price moving vs by `F` moving.
3. **Event backtester:** replays the recordings with a **conservative maker fill rule**. Our buy at
   P fills only if a recorded trade prints strictly below P, or at P with more volume than the
   estimated queue ahead of us from the book. It adds 1–2s action latency, fees and min notional.
4. **Shadow:** live quotes, paper maker orders, same fill rule. It reports fill rate,
   time-to-fill, edge per fill and adverse selection.
5. **Live smoke:** minimum-size real orders (50k TMN) to measure true fill rate and latency, then
   compare them with shadow.

**Go/no-go after step 3 (Thursday ~18:00, ~18h of recordings):** positive net edge per trade after
fees under the conservative fill rule in both halves of the recording (walk-forward by ~6h blocks,
not by day), with ≥ 10 simulated trades per day across pairs. Otherwise drop Strategy 2 and run
Strategy 1 live.

### 14.5a Findings while building steps 1–2 (Thu 01:30)

- **`/v1/markets` bid/ask is a cached summary, not the live book.** It matched the real order
  book in under 6% of checks and often did not change for minutes. The first ~2h of recordings
  (`data/live_v1_markets_cached/`) are therefore set aside.
- **Recorder v2 (running since Thu 01:31 Tehran)** fetches the real `/v1/depth` book of all 15 pairs
  in parallel over a keep-alive pool: about 1s per sweep, a snapshot every ~2–3s, no rate-limit
  errors at 5 workers. It writes top 5 levels plus trades to `data/live/<date>/books.jsonl.gz`.
- `/v2/depth/all` returns every market's book in one call (1.1 MB), but takes 9–32s from here.
  It may be usable from a server inside Iran.
- **Queue competition is real.** Bots move the BTCTMN best bid by +1 TMN every few seconds, so
  "best bid + 1 tick" maker orders are quickly outbid. The replay fill rule (trade-through or
  queue-ahead volume) accounts for this.
- **Steps 1–2 are built:** `cli.py fv-research` (gap stats, entry events and how they close,
  lead-lag) and `cli.py fv-backtest` (replay with the conservative maker fill rule, θ grid × two
  fee scenarios, first-half vs second-half check). Both are tested on synthetic cases, and the
  end-to-end run on real recordings works.

### 14.6 AI role

- **Gap-close model (ML).** For each gap event it predicts P(TMN mid moves ≥ X% toward `F` within
  H minutes while `F` does not move away) and P(fill). Features:
  - gap size and how long it has lasted;
  - which leg moved last, and how much, over the last 10–60s;
  - spreads and imbalance of all three books, depth at our price;
  - recent trade flow (buy/sell);
  - volatility of `F` and USDTTMN moves;
  - hour of day.
- **Decision:** entries and size go through EV after fees, as in Strategy 1. It is trained on the
  recorded events, with walk-forward by day.
- **GPT overlay:** unchanged. News and Iranian macro risk-off (a rial shock makes USDTTMN jump
  and breaks `F`), plus veto or reduce on abnormal conditions.
- **Risk manager:** reused, with new limits: max inventory per pair, max open maker orders per pair,
  max total exposure, and cancel-all on kill switch or end of challenge.

### 14.7 Speed and API budget

- **Polling cost:** REST latency is about 1s per request. The quote loop costs 1 request
  (`/v1/markets` covers all pairs), and book and order calls only happen when acting.
- **Websocket:** check whether Wallex offers one. It would cut reaction time from seconds to sub-second.
- **Rate limits:** to be confirmed with Wallex. Recorder and trader must share the budget, or the
  recorder pauses during live trading.

### 14.8 Build steps (after approval)

| step | output |
|---|---|
| 1 | `engine/fv/` research notebook-style script: gap stats + lead-lag on recorded data → report |
| 2 | `engine/fv/sim.py`: event replay with maker fill model; `cli.py fv-backtest` |
| 3 | `engine/fv/model.py`: gap-close + fill models, walk-forward by day |
| 4 | `engine/fv/runner.py`: fast quote loop + order manager (post / re-quote / cancel, unique client ids); `fv-shadow` |
| 5 | reuse risk manager, decision log, agent overlay; add maker-order limits |
| 6 | live smoke test with minimum size, then a decision on how to split capital between Strategy 1 and 2 |

### 14.10 Compressed schedule (deadline Friday 23:00 Tehran)

| when (Tehran) | work | gate / output |
|---|---|---|
| Wed night (now) | recorder running; **you send Wallex API key, OpenAI key, MCP config, fee tier** | keys in env vars |
| Thu 00:00–10:00 | build research script + event replay backtester (steps 1–2) on the first hours of recordings; `verify-api` once the key arrives | replay runs end to end |
| Thu 10:00–16:00 | fill model + simple gap-close model (step 3); fast quote loop + order manager in shadow (step 4) | shadow running |
| Thu ~18:00 | **go/no-go** on ~18h of recordings (§14.5) | decide Strategy 2 vs Strategy 1 |
| Thu evening | live smoke test with 50k TMN orders; then live at reduced size (ramp-up) | first real trades + decision logs |
| Fri 00:00–20:00 | live trading, monitoring; docs (README/STRATEGY), GitHub repo, demo video script in parallel | reports from live run |
| Fri 20:00 | no new entries (1h before trading end) | — |
| Fri 20:30 | flatten everything to TMN, verify zero coin balance | — |
| **Fri 21:00** | **trading ends** (`challenge.end`); freeze code tag + logs | final account value in TMN |
| Fri 21:00–23:00 | final report, push repo, record/submit video | submission by 23:00 |

Fallback at every gate: Strategy 1 (already built and tested) runs live so the account trades by
algorithm regardless. GPT overlay and risk manager apply to both.

### 14.9 Open questions for Wallex / organizers

- Maker vs taker fee on the challenge account.
- Websocket availability and rate limits.
- Order status fields and whether partial fills are reported (needed for the order manager).

## 15. Last-day plan: trade the Binance lag on USDT pairs (written Thu 2026-10-22 21:45 Tehran)

Source: `reports/Trading agent ideas review.md`. Rules: trade USDT pairs, hold USDT when idle, convert to Toman only at the end.
Time left: about 23 h of trading (ends Fri 21:00). XAUT has no Binance reference, so it is excluded from items 1-3.

### What we build, in order

| # | Item | What it does | Effort | Go rule (fixed now, before any result) |
|---|---|---|---|---|
| 1 | **Gap logger** | Every 5 s: Binance book for BTC, ETH, XRP, SOL, PEPE, BNB vs the Wallex `/v1/depth` book. Fair = Binance mid × (1 + basis), basis = rolling 45 min median of Wallex/Binance − 1. Logs gap of the Wallex ask and bid to fair, spread, snapshot age, and Wallex `/v1/trades` flow. | 1 h | None; it only measures. Runs until the end. |
| 2 | **Lag-buy** | Buy (marketable limit at the ask) when the Wallex ask < fair × (1 − 0.35%) and Binance rose ≥ 0.4% (BTC/ETH) or ≥ 0.6% (others) over 3 min. Sell with a maker order at entry × 1.003; time stop 5 min (taker exit); exit early if Binance falls 0.15% from entry. One position per pair. | 2 h | **Historical test on 1-minute candles** (Wallex + Binance, validation and held-out windows, entry one minute late, taker exit, 0.26% round trip): go if net > 0 in **both** windows, profit factor ≥ 1.2 and ≥ 30 trades. Then **shadow** overnight: go live if shadow net ≥ 0 with ≥ 10 signals. |
| 3 | **Binance-priced maker quotes** | Bid at fair × (1 − h), h = 0.20% BTC/ETH, 0.30% XRP/SOL/BNB, 0.5% PEPE. After a fill, rest the sell at fill × 1.003. Requote when fair moves 0.1% or after 180 s. Inventory cap 40% of the pair's allocation; no bids while Binance's 15 min return < −0.7%. | 3-4 h | Only on pairs where the logger shows median spread ≥ 0.30% and ≥ 20 Wallex trades/hour. Shadow fills counted when Wallex trades print through our quote price. Go if shadow net ≥ 0 over ≥ 2 h. |
| 4 | **Hard risk layer** (built with 2 and 3) | Per-pair stop at −2% of that pair's allocation; stop new buys at −1.5% on the day; halt everything at −3%. Sizes set in code, never by GPT. Existing Binance crash guard stays. | 1 h | Always on. |
| 5 | **Trump risk gate** | Poll `www.trumpstruth.org/feed` every 90 s. Keyword filter, then one gpt-6-luna call returns risk −2..+2. Risk ≤ −1 on tariff/geopolitical news blocks new buys for 3 h; −2 sells alts to USDT. | 1 h | Always on once built. Under $0.05 of GPT. |
| 5b | **Fed module** (permanent part of the strategy; no Fed event falls before this deadline, but the bot is meant to run afterwards) | **Calendar:** `data/fed_calendar.yaml` lists FOMC decision times, minutes releases, chair speeches/testimony and CPI/payrolls days. It is refreshed from the federalreserve.gov calendar pages, with a manual file as fallback. **Before an event:** no new entries from 2 h before an FOMC decision or chair speech (30 min before CPI/payrolls); open positions tighten stops to 0.5 ATR. **On release:** poll `press_monetary.xml`, `speeches.xml` and `press_all.xml` every 60 s (every 10 s during the event window). gpt-6.1-sol scores the statement against the previous one and against expectations: hawkish/dovish −2..+2, surprise 0-1, key changes in one line. **After:** surprise ≥ 0.5 hawkish → no buys for 4 h; −2 → sell alts to USDT. Dovish → normal trading resumes after a 15 min wait (no chasing the first spike, since FOMC moves often reverse). Neutral → resume after 30 min. **Model features (later):** `hours_to_fomc`, `hours_since_fomc`, last Fed score; added to the hourly model at its next retrain, with event-window backtests over the past FOMC dates in our data (2025-26). | 2 h (calendar + gate + scorer); features at next retrain | Gate always on. Tested now by replaying the last 8 FOMC statements through the scorer and checking BTC's move in the 4 h after each; the features need a walk-forward test before use. |
| 6 | **GPT overlay rework** | One batched gpt-6.1-sol call per hour for all signals; veto only with a reason code; exit plan recorded at entry; reflection every 8 h writing 3 lessons into the prompt; every veto logged with what would have happened. | 1-2 h | Always on. Budget stays under the $1.50 cap. |

Skipped (reasons in the report): X/Twitter scraping or API, multi-agent debate, RL, retraining the hourly model, static grids, the hour-of-day tilt.

### Money split (one USDT pool)
Hourly model (U1, USDT pairs) up to 50% of equity; lag-buy up to 25% per trade; maker quotes up to 30% across pairs only if item 3 passes. No more than 100% of the pool is ever committed.

### Timeline (Tehran time)
- **Thu 21:45-23:30:** build item 1 and start it; historical test of item 2.
- **Thu 23:30-01:30:** build items 2 and 4 into the live runner; start the overnight shadow (lag-buy + logger).
- **Before sleeping (user):** run `python cli.py smoke-order --yes` and share the output; go live with the hourly U1 model in USDT mode if the smoke test is clean.
- **Fri 08:00-10:00:** read the overnight shadow and logger; apply the go rules for item 2; build items 5 and 5b.
- **Fri 10:00-13:00:** build item 3 only if the logger passes its rule; shadow it ≥ 2 h; build item 6.
- **Fri 13:00-20:00:** everything that passed runs live.
- **Fri 20:00:** no new entries. **20:30:** flatten and convert all USDT to Toman (BTC route if USDT/TMN is frozen). **21:00:** trading ends.
- **Fri 21:00-23:00:** final logs, README/STRATEGY results, demo video, GitHub repo.

### What could stop us
- The smoke test has not been run, so order-status parsing is still unproven.
- Wallex fill rates and rate limits are unmeasured; the logger answers both.
- The lag may be stale quotes rather than real tradable prices; the historical and shadow rules catch that.
- The internet outage seen Thursday 15:20-15:41: stops run client-side, so a long outage while holding is a real risk.

#### §15 result 1 (Thu 22:10): lag-buy as written is NO-GO
Strict fills: validation 116 trades, PF 0.29, sum −20.5%; held-out 64 trades, PF 0.08. Diagnostic: after a signal the
next Wallex trades print +0.4-0.65% (median) higher within 2-10 min on every pair while Binance is flat, so the lag is
real; the 0.15% Binance-reversal exit fired on noise before Wallex caught up.
**Retest rule (fixed before running):** grid bn_reverse ∈ {0.15%, 0.5%, off} × take_profit ∈ {0.3%, 0.4%, 0.5%} ×
time_stop ∈ {5, 10} min, entry rule unchanged, strict fills. Pick the highest validation sum among settings with
PF ≥ 1.2 and ≥ 30 trades; run it ONCE on the held-out window; go only if it also passes the go rule there.

#### §15 status (Thu 22:20)
- Item 1 gap logger: running since 21:48 (`lag-log`). Item 2 lag-buy: NO-GO on candles; decided by `lag-replay` on the logged books Friday morning.
- Items 5 + 5b event gate (Trump, Fed, Binance, news) and item 4 hard limits: built, tested, in the shadow.
- Hourly model switched to U1 on USDT pairs; shadow restarted 22:19 in USDT mode with the event gate (STRATEGY.md §7.1n).
- Open: smoke test by the user, then live; item 3 maker quotes (only if the logger passes its rule); item 6 GPT overlay rework
  (its prompts are still written for Toman pairs).
- Item 6 GPT overlay rework: built (Thu 22:45): batched review, reason-coded vetoes, exit plans, reflection, veto-report. Shadow restarted with it.

#### §15b Fixing U1 (rule fixed Thu 23:50, before any result)
Candidates, all with **per-pair costs measured by the gap logger** (half the median Wallex spread, replacing the flat 0.15%):
- **A** = U1 model as is, 7 pairs.
- **B** = U1 model without BNBUSDT (0.9% spread) and XAUTUSDT (no Binance data): 5 pairs.
- **C** = new model trained on Binance 1h candles pooled over ~30 liquid coins (2 years), no Wallex data in training;
  traded on the same 5 Wallex pairs as B with Wallex 1-minute fills.
Same windows for all: validation 2026-08-01..2026-09-15, held-out 2026-09-17..2026-10-21, same risk settings.
**Pick:** among candidates with validation profit factor ≥ 1.2 and ≥ 10 trades, the highest validation return.
**Confirm once** on the held-out window: the pick goes live only if its held-out return beats U1's −0.95%;
otherwise A (U1 with measured costs) runs.
- §15b result (Fri 00:20): C made 0 validation trades (ineligible); A and B tied on validation (+4.45%); B held-out
  −1.41% does not beat −0.95%, so A (U1 with measured costs) runs. Details: STRATEGY.md §7.1p.

#### §15c Longer history (rule fixed Fri 00:30, before any result)
- **D** = U1 design (Wallex USDT pairs + Binance outside features), trained on Wallex hourly history back to 2021
  (Wallex `udf/history` resolution 60) with Binance hourly candles back to 2017, instead of one year.
  `active_frac` is left out (old hourly candles have no per-minute activity); live features are unchanged otherwise.
- **E** = candidate C design (Binance only, 30 coins pooled), trained on all Binance history (2017 onward).
Same windows, per-pair costs and risk settings as §15b. **Pick:** highest validation return among those with
PF ≥ 1.2 and ≥ 10 trades. **Confirm once** on the held-out window: adopt only if it beats A's held-out −1.41%
**and** is ≥ 0%; otherwise A keeps running.
- Added before any result (user request "add more features"): **D+** and **E+** = D and E plus 12 extra features
  (`model.extra_features`): Binance taker flow over 4/8/12 h, Binance trade-count z-score, 1 h and 6 h return vs BTC,
  short/long volatility ratio, cross-sectional 6 h / 24 h return ranks, market breadth, 24 h funding change.
  The pick rule above applies to all four candidates (D, D+, E, E+).
- Added before any result (user asked about deep learning): **F** = GRU sequence model (PyTorch, GPU) on the D+
  data and features: input = the last 72 hourly feature rows of a pair, output = probability of the triple-barrier
  win label; trained on the same train split, isotonic-calibrated and EV-threshold-tuned on the same rows as the
  others, early stopping on the walk-forward fold before validation. Same pick rule (D, D+, E, E+, F).
- §15c result (Fri 01:40): D, D+, E, E+ all made 0 validation trades -> ineligible; A keeps running. F (GRU) not trained yet (user decides).

#### §16 Trend following on daily / 4-hour bars (rule fixed Fri 02:00, before any result)
For use after the challenge (target: ≥ 4% a month in US dollars). Long-only spot, idle cash in USDT.
- **Coins:** BTC, ETH, XRP, SOL, BNB, PEPE (the Wallex USDT pairs with Binance history); Binance hourly candles
  resampled to the bar size; a coin joins when it has enough history for its lookback.
- **Signal per coin at each bar close:** in trend if close > SMA(N) **and** the N-bar return > 0; trade at the
  next bar's open (no look-ahead).
- **Portfolio:** equal weight 1/6 of equity per coin in trend, the rest in USDT; rebalance only when a coin's signal
  changes. Costs per side: taker fee 0.13% + the pair's measured Wallex half-spread (PEPE 0.23%, BNB 0.35%, ...).
- **Grid (6 variants):** bar ∈ {1d, 4h} × N ∈ {20, 50, 100} days (4h bars use 6×N bars).
- **Development window** 2018-01-01 .. 2024-12-31: pick the variant with the highest mean monthly return among
  those with max drawdown ≤ 35%. **Held-out** 2025-01-01 .. 2026-10-22: run the pick once. It "meets the target"
  only if its held-out mean monthly return is ≥ 4%. Benchmarks: equal-weight buy-and-hold, BTC buy-and-hold.
- §16 result (Fri 02:10): no variant had max drawdown ≤ 35% on 2018-2024 (47-58%) -> nothing confirmed. STRATEGY.md §7.2a.

#### §17 Wallex-vs-Binance premium mean reversion, hourly (rule fixed Fri 02:20, before any result)
- **Pairs:** BTC, ETH, XRP, SOL, BNB, PEPE (Wallex hourly USDT candles 2021+, Binance hourly candles).
- **Premium** p = log(Wallex close / Binance close) at each hour close; deviation d = p − median of p over the
  previous 168 hours; z = d / std of d over the previous 168 hours. Only hours in which Wallex traded count.
- **Entry:** z ≤ −k at an hour close -> buy at the next hour's Wallex open. **Exit:** at the next open after d ≥ 0
  (premium back to normal) or after H hours, whichever comes first. One position per pair, 1/6 of equity each.
- **Costs per side:** taker 0.13% + the pair's measured Wallex half-spread.
- **Grid (12):** k ∈ {1.5, 2, 2.5, 3} × H ∈ {1, 3, 6}.
- **Development** 2021-01-01..2024-12-31: among variants with ≥ 100 trades and profit factor ≥ 1.2, pick the highest
  total return. **Held-out** 2025-01-01..2026-10-22, once: passes if profit factor ≥ 1.2 and net return > 0;
  "meets 4%/month" only if its mean monthly return is ≥ 4%.
- §17 result (Fri 02:25): dev 2021-24 +1,111% (PF 2.48) but held-out 2025-26 −41.8% (PF 0.74) -> fails. STRATEGY.md §7.2b.
- Item 3 maker quotes (Fri 01:55): no pair meets spread ≥ 0.30% and ≥ 20 trades/h (4.1 h of logger data); replay on PEPE/BNB: 0 fills -> not built. STRATEGY.md §7.1r.
- Fri 02:15: user decision -> live strategy S1 (Toman pairs, idle Toman, original limits); shadow restarted with S1. Lag-buy replay: 0 signals in 4.3 h -> no-go. STRATEGY.md §7.1s.
- §15c F (GRU, Fri 02:40): validation +9.46% (PF 1.92, 35 trades; calibration tuned on it) -> held-out once: −5.98% (PF 0.36) -> fails. STRATEGY.md §7.1q.

#### §18 Improved GRU, judged by a 12-month walk-forward (rule fixed Fri 02:50, before any result)
The Sep 17 - Oct 22 window has been used by many candidates, so a single held-out month can no longer judge a new
version. **G** = improved GRU on the D+ data (7 Wallex USDT pairs, Wallex 2021+, Binance 2017+, extra features):
window 48 h, hidden 32, dropout 0.3, at most 3 epochs, **average of 3 seeds**; per fold, early stopping on the 60
days before the last 60 days of training and **isotonic calibration + EV threshold on the last 60 days of training**
(predictions out-of-sample for those rows; nothing tuned on the test month). PEPE stays in (removing it would use
the held-out result). Reference **T** = the tree ensemble (logreg + LightGBM) on the same data and folds with the
same calibration scheme.
**Folds:** 12 test months, 2025-11 .. 2026-10 (last one to 10-22); each trained on all data before the month
(13 h purge); traded with the normal backtest (Wallex 1-minute fills, per-pair costs, 18.3 USDT, same risk rules).
**G passes** if, over the 12 months: total compounded return > 0, profit factor of all trades ≥ 1.2, ≥ 30 trades,
and ≥ 8 of 12 months with return ≥ 0 (a month without trades counts as 0). It is used only after the challenge;
the challenge keeps S1.

#### §19 S1 with at least 2 trades a day (rule fixed Fri 02:55, before any result)
User goal: ≥ 2 trades/day. Same S1 model, Toman pairs, original limits; only position and filter settings change:
- **M1** = up to 3 positions at 30% of equity each (max_positions 3, max_position_frac 0.30), tuned EV threshold.
- **M2** = M1 + take every trade the model rates positive after costs (EV > 0; tuned threshold off).
- **M3** = M2 + min_p_win 0.35 (instead of 0.40).
Validation 2026-08-01..09-15: pick the highest return among variants with ≥ 2.0 trades/day and PF ≥ 1.2.
Held-out 2026-09-17..10-22, once: passes if ≥ 2.0 trades/day, PF ≥ 1.2 and return > 0. The user then chooses
between it and S1 (held-out +20.47%, 1.2 trades/day).
- **Amended before any result was seen** (user: "I am okay with more risk, but I want more daily trades, the more
  the better"): added **M4** = M3 + up to 5 positions at 18% each and up to 4 new entries per bar. **New pick:** the
  variant with the most trades/day among those with validation return > 0 and PF ≥ 1.1. Held-out, once: passes if
  return > 0 and PF ≥ 1.1. Trades the model itself expects to lose on (EV < 0) stay excluded.
- §18 result (Fri 03:15): G +8.87% over 12 months, PF 2.72, 11/12 months ≥ 0, but 13 trades -> fails (≥ 30). Trees +7.99%, 18 trades.
- §19 result (Fri 03:15): M1-M3 identical (39 validation trades); pick M4 held-out +4.24%, 59 trades (1.7/day), PF 2.18 -> passes, but < 2/day. STRATEGY.md §7.2c-d.

#### §20 Deadline extended (user, Fri 2026-10-23 ~12:30)
Submission Monday 2026-10-26 23:00 Tehran. Trading continues until **Monday 20:00**: no new entries from 19:00,
all positions sold to Toman from 19:45 (`challenge.end`, `no_new_entries_before_end_min: 60`,
`flatten_before_end_min: 15`). Strategy unchanged: S1. Next: smoke test (user), Liara deploy (shadow, check-net),
then live as early as possible to get ~3 days of live trading.
- Smoke test (user, Fri 2026-10-23 13:51 Tehran): USDTTMN BUY 0.4 and SELL 0.39 both FILLED at once, no open orders left.
  Confirmed: status `FILLED`, `executedQty`, `executedSum`; BUY fee charged in the base asset (USDT), SELL fee in TMN;
  taker fee 0.30%. **Bug fixed:** `executedPrice` is the limit price, not the fill price -> the broker now uses
  executedSum / executedQty (then the fills). Balance 5,000,000 -> 4,996,914 TMN plus 0.0088 USDT dust (about 2,400 TMN:
  the buy fee in USDT left 0.3988, the sell is floored to the 0.01 step), so the real cost was about 700 TMN.
