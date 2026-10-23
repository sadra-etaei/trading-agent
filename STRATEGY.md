# Strategy and Architecture

This document explains how the engine decides, how it uses AI and the Wallex tools, how it
manages money and risk, and how to re-run and evaluate it. How to install and run it:
[README.md](README.md).

## 1. Architecture

```
            Wallex public API                      news RSS (crypto + Persian)
   /v1/udf/history  /v1/markets  /v1/depth                 │
            │               │          │                    ▼
            ▼               │          │           GPT headline scoring
   1-min candle store ──► features ──► ENTRY MODEL (LightGBM, calibrated)
   (Parquet, resampled)     │            │ p_win per pair
                            │            ▼
   Fear & Greed index ──────┘     EV ranking: EV = p·TP − (1−p)·SL − costs
                                         │ top candidates above threshold
                                         ▼
                                  GPT OVERLAY AGENT ◄── scored headlines, market context
                                  approve / reduce / veto / risk-off / early exit
                                         │
                                         ▼
                                  RISK MANAGER (deterministic, final word)
                                  size caps, stops, daily loss, kill switch, end-of-challenge flatten
                                         │
                                         ▼
                      BROKER:  Backtest (1-min simulation) | Shadow (live book) | Wallex (real orders)
                                         │
                                         ▼
                        decision log (JSONL) · trades · equity · reports
```

The same `Strategy`, `RiskManager`, `Portfolio` and exit logic objects run in all three modes.
Only the broker changes. Features are computed by one function for training and for live trading.

## 2. Data

- **Candles:** Wallex `/v1/udf/history`. On Wallex, `resolution=15` returns 1-minute candles,
  with at least 500 days of history. One year of 1-minute candles is stored per pair and
  resampled to 1-hour decision bars, with a count of active (traded) minutes per bar.
- **Universe:** the most liquid TMN pairs by volume and spread: BTC, ETH, SOL, XRP, BNB, DOGE, TRX
  and USDT (XAUT optional). Recently hyped or wide-spread pairs are excluded.
- **Sentiment with history:** the Crypto Fear & Greed Index (daily since 2018, lagged one day to
  avoid look-ahead). It is used as an ML feature. Binance funding rates are optional.
- **Sentiment without history:** news headlines. They go to the GPT agent only, never into the ML model.
- **Order books:** there is no history, so `cli.py record` records them from now on. The live broker
  uses the order book for fills.

## 3. The entry model: when and what to buy

Spot only, so every trade is a long: buy, then sell back to TMN.

**Question the model answers.** If we buy this pair at the next bar's open, will price reach
take-profit before stop-loss within 12 hours?

- Take-profit = max(1.5 × ATR%, 1.2%), so a winning trade always clears fees.
- Stop-loss = 1.0 × ATR%, clipped to the range 0.6%–4%.
- A bar where both levels are touched counts as a loss, which keeps the labels conservative.

These are exactly the levels the risk manager uses in live trading.

**Features (41).** All of them use past data only:

| group | features |
|---|---|
| returns | log returns over 1, 3, 6, 12, 24 and 72 hours |
| trend | distance from EMA20/50/200, EMA slopes, ADX |
| volatility | ATR%, realized volatility, ATR percentile over 30 days, Bollinger width and z-score |
| momentum | RSI, MACD histogram, stochastic, position in the 24h range |
| volume | volume z-score, signed volume flow, share of active minutes, stale-market flag |
| market context | BTC returns and volatility, **USDT/TMN returns** (rial moves shift every TMN price), strength relative to BTC |
| time | hour of day (Tehran time), day of week |
| regime | ranging / trending / high-volatility |
| rule signals | trend pullback and mean-reversion flags (inputs to the model, not gates) |
| sentiment | Fear & Greed level and 7-day change |

**Model.** One LightGBM classifier pooled over all pairs, kept small and regularized. Its
probabilities are calibrated with isotonic regression on a separate validation window, because
the probability drives both the trade decision and the position size.

**Decision rule.** At every hourly close, each pair gets

`EV = p_win × TP% − (1 − p_win) × SL% − round-trip costs (fees + slippage)`

Pairs with EV above the threshold tuned on validation are ranked by EV. The best one or two are
passed to the agent and the risk manager. Position size grows with how far EV exceeds the threshold.

**Validation without look-ahead.**
- **Walk-forward folds:** each fold trains only on the past, with a purge gap the length of the label window.
- **Validation window:** used only for calibration and the EV threshold.
- **Final held-out test window:** used once, for the backtest of record.
- **Benchmark:** the model is compared against the plain rule signals after fees, and it is used
  only because it beats them (see §7).

**Selection model (experiment).** A second model can rank pairs by predicted 24-hour return
relative to the other pairs, so the entry model trades only the top-k. `cli.py experiment-selection`
compares it against plain EV ranking on the held-out window and keeps it only if it wins (§7).

## 4. AI usage

There are two AI components, each with a real decision role.

1. **ML entry model.** It decides what to buy, when, and how much (through EV). Rules are only features.
2. **GPT overlay agent** (OpenAI models, configurable). It works on what price history cannot show:
   - **Headline scoring:** every 15 minutes, crypto and Persian/Iranian news headlines are scored
     by a cheap model for direction, category, affected assets and severity, then logged.
   - **Trade review:** a stronger model sees each candidate trade (p_win, EV, regime, key features),
     the portfolio state, market context and the scored headlines. It returns strict JSON:
     approve, reduce (size × 0.25–0.75) or veto, an optional market-wide `risk_off` (no entries
     for 4 hours), a confidence and a short reason.
   - **Exit review:** after significant news (severity ≥ 2), it can close open positions early.
   - **Guardrails:** the agent can never open a trade and can never exceed risk limits. If it fails
     or times out, the default is veto (configurable).
   - **Reproducibility:** `temperature=0`, and every prompt and response is stored under a hash of the
     input in `agent_cache/`, so a re-run replays the same decisions.

**Wallex MCP.** The organizers' MCP server will be wired in as a tool source for the agent, for
market data, balances and order status, once its endpoint and configuration are available.
Order placement stays in deterministic code for speed, safety and auditability.

## 5. Selling conditions

| exit | rule |
|---|---|
| take-profit | price reaches entry × (1 + TP%) |
| stop-loss | price reaches entry × (1 − SL%) |
| trailing stop | after +1 ATR, the stop trails 1 ATR below the peak and never below break-even |
| time stop | 12 hours after entry |
| agent exit | GPT exit review on significant negative news |
| kill switch | drawdown from peak ≥ 6%: sell everything and stop |
| end of challenge | no entries in the last 3 hours, sell everything to TMN in the last 45 minutes |

In the backtest, exits are checked on every 1-minute candle. In shadow and live they are checked on
every poll (15 seconds) against the best bid.

## 6. Money and risk management

The hackathon ranks teams mainly on return (90 points, and 0 for a team that loses money), so
the limits below are set for profit first. They still keep the controls the judges ask for: a
stop-loss on every trade, a cap on order size, never all-in, and a kill switch. The values were
chosen on the validation window (§7.1e).

| rule | value |
|---|---|
| position size | equity × min(90%, 5% ÷ SL%) × EV scale (0.5–1) × agent multiplier (S1, §7.1k) |
| max equity lost if a stop is hit | about 1–3.5% (90% × the stop distance) |
| max open positions | 1 |
| max total exposure | 90% of equity (never all-in) |
| averaging down | never |
| cooldown after a losing trade | 1 hour on that pair |
| daily loss limit | −8% (Tehran day) → no new entries that day |
| kill switch | −30% from peak → sell everything and halt (never reached in any backtest) |
| exchange rules | min notional 50,000 TMN, price and quantity precision from `/v1/markets` |

**Execution.** Several TMN pairs have market orders disabled, so every order is a LIMIT order.
- **Entries:** posted at the best bid (maker) for 90 seconds. Anything unfilled is re-sent as a
  limit order that crosses the spread, capped at 0.3% slippage.
- **Exits:** limit orders that cross the spread, with a widening cap on each retry.
- **Order ids:** every order has a unique client id.
- **Restarts:** state is saved after every change, so a restart resumes open positions (`--resume`).

## 7. Evaluation results

**Data:** 8 TMN pairs, 1-minute candles from 2025-10-21 to 2026-10-21 (about 525k per pair), using
1-hour decision bars. **Costs:** 0.25% fee per side plus 5 bps slippage, about 0.6% per round trip.

### 7.1 Model selection (walk-forward + validation only)

Source: `models/experiments-*.md`. The rule was fixed before looking at results: highest total net
return on out-of-fold + validation trades, with at least 3 of 5 positive walk-forward folds.

| variant | WF AUC | OOF+val trades | mean net / trade | positive folds |
|---|---|---|---|---|
| LightGBM, all features | 0.593 | 1 | +0.71% | 2/5 |
| LightGBM, no calendar / F&G | 0.609 | 28 | +0.78% | 3/5 |
| LightGBM + Wallex price gap (premium) | 0.597 | 4 | +2.99% | 4/5 |
| logistic regression, all features | 0.576 | 85 | +1.24% | 3/5 |
| logistic regression + premium | 0.583 | 134 | +0.81% | 4/5 |
| **logistic regression, no calendar / F&G (chosen)** | 0.580 | **110** | **+1.13%** | 3/5 |
| wider targets (TP 2.5 ATR, 24h) | 0.531 | 0 | — | 0/5 |
| *plain rule signals (baseline)* | — | ~1000 per fold | **−0.51%** | 0/5 |

Findings:
- **Rules lose after fees.** The plain rule signals lose money after fees in every fold. The ML
  model's edge is weak but positive out of sample, and it trades rarely: a trade pays only when
  p_win exceeds about 60%, against a base win rate of about 30%.
- **Fear & Greed and calendar features did not help** out of sample. They were removed from the
  model, and Fear & Greed stays in the GPT agent's market context.

### 7.1b Round 2: more pairs, ensemble, 15-minute bars (walk-forward + validation only)

Source: `models/logs/exp_round2*.log`. Same selection rule as round 1.

| variant | WF AUC | OOF+val trades | mean net / trade | total | positive folds |
|---|---|---|---|---|---|
| round-1 choice (8 pairs, logistic) | 0.580 | 110 | +1.13% | +1.24 | 3/5 |
| ensemble (logistic + LightGBM), 8 pairs | 0.610 | 84 | +1.26% | +1.06 | 4/5 |
| 15-minute bars | 0.619 | 1,375 | +0.19% | +2.57 | 4/5 (latest validation −0.13%: rejected) |
| 16 pairs, logistic | 0.632 | 2,709 | +0.36% | +9.78 | 4/5 |
| **16 pairs + ensemble (chosen)** | **0.650** | **3,158** | **+0.34%** | **+10.83** | **5/5** |

The 8 added pairs are PEPE, SUI, LTC, ADA, LINK, AVAX, SHIB and UNI. They are allowed, liquid,
and each has at least a year of history. More pairs give the model far more opportunities.

**Backtest on the held-out month** (`runs/bt-*-round2`). The model was chosen without looking at
this window. Round 1 had used it once, so treat it as indicative.

| metric | value |
|---|---|
| net return (35 days) | **+1.40%** |
| max drawdown | 4.15% |
| trades | 104 (45 take-profit, 35 stop-loss, 21 trailing, 3 time) |
| profit factor | 1.12 |
| rolling 3-day windows | 33 windows, median 8 trades, 48% positive, mean +0.08%, worst −1.49%, best +2.76% |

Over a month the edge is positive. On a single 3-day window the result is close to a coin flip
with small losses. The live shadow run on unseen data is the confirmation.

### 7.1c Market-trend filter — tested, not used

After the 2026-10-22 overnight shadow run (5 longs, all stopped out while BTC fell about 2.5%), six
rules that block new longs in a weak market were tested on out-of-fold + validation predictions
(`engine/strategy/trend.py`, training report section "Market-trend filter").

| rule | trades | win rate | mean net | sum net | positive folds | worst fold |
|---|---|---|---|---|---|---|
| **none (used)** | 3064 | 67.2% | +0.350% | **+10.72** | 6/6 | **+0.16** |
| BTC above 50h EMA | 1749 | 70.6% | +0.401% | +7.02 | 6/6 | +0.04 |
| BTC above 200h EMA | 1662 | 67.6% | +0.363% | +6.03 | 6/6 | +0.13 |
| BTC 50h EMA rising | 1728 | 69.6% | +0.402% | +6.94 | 6/6 | +0.09 |
| BTC up over 24h | 1734 | 69.8% | +0.365% | +6.34 | 5/6 | −0.00 |
| ≥ half of pairs above EMA50 | 1325 | 69.8% | +0.419% | +5.55 | 4/6 | −0.03 |
| median pair up over 24h | 1347 | 71.0% | +0.457% | +6.16 | 5/6 | −0.03 |

The rule fixed before the test was: choose the highest total net result among rules that keep the
mean net per trade and the number of positive folds, and do not cut total profit without improving
the worst fold. Under that rule the choice is **no filter**.
- **Why:** the filters raise quality per trade slightly, but drop 43–57% of trades and lower total
  profit, with no better worst case. The model already sees BTC's trend through its features.
- **Overnight losses:** the filter would not have blocked them. BTC was above its 50h EMA and up
  2.4–4.1% over 24h at every entry; the losses came from a reversal after a rally.
- **Not used for the decision:** on the held-out month, the BTC-above-50h-EMA filter would have
  scored +4.63% with a 1.14% max drawdown (vs +1.40% / 4.15%). That month had already been
  looked at, so following the pre-registered rule keeps the selection honest.
  Source: `runs/bt-*-trend-ema50-info`.

### 7.1d Max open positions: 3 vs 5 vs 8 (validation window 2026-08-01 .. 09-15)

| max positions | net return | max drawdown | trades | profit factor |
|---|---|---|---|---|
| **3 (kept)** | **+0.42%** | **1.81%** | 58 | 1.07 |
| 5 | +0.37% | 1.85% | 59 | 1.06 |
| 8 | +0.37% | 1.85% | 59 | 1.06 |

**The cap almost never binds.** In 45 days, "max positions" blocked one entry, versus 9 blocks by
the cooldown after a loss. The bot held 0 positions in 86% of hours, and mean exposure was 2.1% of
equity. The limit on activity is how often the model finds a trade with positive EV, not the
number of slots.

**Correction to §7.1c:** the "3,064 trades" in the out-of-fold tables count every hourly row
whose signal passed. Consecutive hours of the same move are counted separately, so this
overstates how many distinct trades exist.

### 7.1e Position sizing and risk limits (validation window 2026-08-01 .. 09-15)

24 backtests: risk per trade {0.5, 0.75, 1.0, 1.5}% × max position {20, 30, 40}% of equity,
each run under the original tight limits and under relaxed limits. Relaxed means daily loss
8% instead of 3%, kill switch 15–30% instead of 6%, no half-size first day, exposure cap 90%
instead of 60%, and a 1h cooldown instead of 3h.

**Rule, fixed before the results:** highest mean return over the rolling 3-day challenge
windows (the return ranking matters most), provided the bot never halts itself. No run ever
reached a kill switch; the worst max drawdown was 5.58%.

| setting | net return | max DD | 3-day mean | 3-day worst / best | 3-day positive |
|---|---|---|---|---|---|
| old: 0.5% / 20%, tight limits | +0.42% | 1.81% | +0.01% | −0.93% / +1.07% | 47% |
| 1.0% / 40%, tight limits | +1.06% | 3.44% | +0.04% | −1.84% / +1.97% | 47% |
| **chosen: 1.5% / 40%, relaxed** | **+1.64%** | 5.58% | **+0.11%** | −4.02% / +5.13% | 47% |

Bigger positions scale both gains and losses. They do not change the chance of finishing a
3-day window positive, which stays at 47%. The ordering across the grid is noisy (some larger
settings did worse), as expected with a thin edge (profit factor about 1.1).

### 7.1f Binance crash guard (validation window, real fees)

**Research** (14 days of 1-minute data). Wallex Toman prices follow Binance by about 1–2 minutes:
corr(Wallex move at t+1 min, Binance move at t) is as high as at t (0.11–0.18). After a Binance
3-minute move of more than 0.5%, Wallex follows by 20–55 bp over the next 10 minutes. That is too
small to trade on directly after ~0.7% round-trip costs, but useful for exiting early.

**Rule, fixed before the test:** highest 3-day mean return, and only if it also beats no guard on
net return. While a position is open, the guard reads the last 3 *closed* Binance minutes and acts
at the next Wallex minute.

| variant | net | max DD | profit factor | 3-day mean | worst day | guard exits |
|---|---|---|---|---|---|---|
| no guard | +1.32% | 5.59% | 1.12 | +0.086% | −2.59% | — |
| exit at −0.5% / 3 min | −3.14% | 3.29% | 0.59 | −0.209% | −2.03% | 35 |
| **exit at −0.8% / 3 min (chosen)** | **+2.18%** | **3.14%** | **1.26** | **+0.179%** | −2.22% | 18 |
| tighten stop at −0.5% | −2.27% | 4.48% | 0.74 | −0.162% | −1.89% | — |
| tighten stop at −0.8% | +0.59% | 4.46% | 1.06 | +0.066% | −2.60% | — |

- **Why −0.8%:** at −0.5% the guard reacts to noise and sells good positions. At −0.8% it catches
  real drops, cutting stop-losses from 9 to 4.
- **Robustness:** it is the only setup still positive after removing its single best trade
  (+0.17%, vs −0.73% without the guard).
- **With aggressive mode A1** it lowers the result (+1.09% vs +3.86%). A1's profit depends on
  riding one big jump, and the guard exits on Binance dips along the way.

### 7.1i Outside data: Binance spot + futures funding (round 3)

New features from Binance (400 days of hourly data for all 15 coins):
- global price returns over 1/6/24h;
- order flow: the taker-buy share over 1h and 6h;
- the volume surge;
- the **global premium**: the Wallex Toman price vs Binance price × USDT/TMN;
- the **catch-up gap**: Binance's move minus Wallex's over 1h and 6h;
- futures funding rates.

Model comparison (walk-forward + validation, real 0.30% fee). The rule was the same as before.

| | current model | **+ outside data + funding (chosen)** | + outside data, no funding | funding only |
|---|---|---|---|---|
| walk-forward AUC | 0.649 | **0.696** | 0.696 | 0.651 |
| OOF+val trades × mean net | 1,720 × +0.41% | **4,455 × +0.39%** | 4,465 × +0.38% | 1,921 × +0.37% |
| validation AUC | 0.598 | **0.646** | 0.647 | 0.599 |
| validation-only mean net | −0.10% | **+0.21%** | +0.20% | +0.00% |

**Backtests with the crash guard on, at the real 0.30% fee.** The held-out month was not used for
this choice.

| entry delay after the hour close | validation net | held-out month net |
|---|---|---|
| 0 min (5 bp slippage) | +6.86% | +13.32% (PF 1.96, 76% of 3-day windows positive) |
| 1 min | +2.59% | +11.38% |
| 2 min | −0.40% | +7.86% |
| 3 min | −2.07% | +6.31% |
| **1 min, 15 bp slippage (live assumptions)** | **+1.04%** (46 trades) | **+11.84%** (54 trades, PF 3.11, max DD 1.66%, 3-day mean +0.97%, 79% positive) |

**The edge decays within minutes of the hour close.** Part of it is Wallex catching up with
Binance. Live execution was therefore rebuilt for speed:
- most data is prefetched ~90s before the close; the newest minutes are fetched right after it,
  in parallel;
- entries are marketable limit orders at once, with no 90-second maker wait;
- news scoring and the agent's exit review run after the entries.

Backtests now assume a 1-minute entry delay and 15 bp slippage per side.

**In dollar terms**, the held-out month was −2.76% (validation −9.74%): the account is in Toman
cash most of the time while the rial fell (USDT/TMN +35% in 60 days). The challenge is scored in
Toman.

### 7.1j More trades? Pairs, bar size and the profit filter (validation, live assumptions)

All runs use a 1-minute entry delay, 15 bp slippage, the 0.30% fee, the crash guard and the
outside-data model retrained per variant. V1/V3 add 20 more allowed liquid pairs: PUMP, XAUT, ARB,
S, APE, FIL, DYDX, HYPE, SAND, FLOKI, AAVE, HBAR, CAKE, GALA, ENA, SEI, BCH, CHZ, FORM, POL.

| | **V0: 16 pairs, 1h (live)** | V1: 36 pairs, 1h | V2: 16 pairs, 30m | V3: 36 pairs, 30m |
|---|---|---|---|---|
| trades per day | 0.9 | 2.9 | 2.3 | 6.8 |
| net (45 days) | **+4.12%** | +2.39% | −1.14% | −11.39% |
| max drawdown | **2.48%** | 10.75% | 5.11% | 12.82% |
| profit factor | **1.54** | 1.08 | 0.94 | 0.83 |

Loosening the profit filter on V0 (positions up to 5):

| filter | trades/day | net | max DD | win rate |
|---|---|---|---|---|
| **EV > 0 (live)** | 0.9 | **+4.12%** | 2.48% | 48.7% |
| EV > −0.2% | 1.9 | −1.83% | 7.28% | 40.0% |
| EV > −0.5% | 5.0 | −16.59% | 18.63% | 35.7% |
| EV > −1.0% | 4.7 | −29.70% (kill switch) | 30.17% | 17.1% |

**Conclusion:** extra trades come only from weaker signals or thin, wide-spread pairs, and each
round trip costs about 0.7%. 10–20 trades a day is not reachable with a positive edge here. The
live setup stays V0, which on the held-out month scored +9.87% (55 trades, 1.6/day, PF 2.42,
3-day mean +0.80%, 70% of 3-day windows positive).

### 7.1k Concentrated sizing with the outside-data model (S1, live)

Rule, fixed before the test: highest validation 3-day mean, provided the kill switch never fires.

| | 3 × 40% | **S1: 1 × 90%, 16 pairs (live)** | 2 × 45% | 3 × 60% | 1 × 90%, 7 volatile pairs |
|---|---|---|---|---|---|
| validation net / 3-day mean | +4.12% / +0.27% | **+5.41% / +0.40%** | +4.73% / +0.31% | +5.39% / +0.36% | +0.93% / +0.09% |
| validation max DD | 2.48% | 4.96% | 2.79% | 3.71% | 5.17% |
| held-out month net / 3-day mean | +9.87% / +0.80% | **+20.47% / +1.61%** | +11.70% / +0.94% | +13.46% / +1.09% | +11.60% / +0.95% |
| held-out 3-day positive / worst | 70% / −1.48% | **73% / −2.02%** | 73% / −1.27% | 73% / −2.05% | 67% / −2.02% |
| held-out net without best trade | +8.51% | **+17.46%** | +10.15% | +11.52% | +9.06% |

- **Why concentration works here:** the model trades about once a day, so the best signal each
  hour gets the whole position.
- **All 16 pairs beat the 7 volatile ones:** more pairs mean more good signals to choose from.
- **Risk:** the worst 3-day loss grows to about 2–2.5%, and on validation the result without the
  best trade is only +0.76%.

### 7.1l Strategy 3: trade the Binance lag on USDT pairs (PLAN.md §15)
Rule: buy when the Wallex ask is > 0.35% below fair (Binance mid x rolling 45 min basis) right after Binance rose
>= 0.4% (BTC/ETH) or 0.6% (others) in 3 min; sell at +0.3%, after 5 min, or when Binance falls 0.15% below entry.
Code: `engine/lag/` (`rules.py` shared by every stage). Commands: `lag-log`, `lag-backtest`, `lag-replay`.

**1-minute candle test (strict fills: a real Wallex trade at or below our limit in the next minute, taker fees):**

| window | trades | win rate | mean net | profit factor | sum |
|---|---|---|---|---|---|
| validation (45 d) | 116 (2.6/day) | 51% | −0.18% | 0.29 | −20.5% |
| held-out (35 d) | 64 (1.8/day) | 41% | −0.33% | 0.08 | −21.1% |

Pre-registered go rule failed. The retest grid (18 exit settings, validation only) had no setting with PF >= 1.2.
What we learned: after a signal, the next Wallex trades print +0.4-0.65% (median) higher within 2-10 min on every pair
while Binance stays flat, so the trade-price lag is real. But Wallex USDT books trade rarely (BTC in 25% of minutes,
BNB in 5%) and live books show quotes already close to Binance (BTC ask gap 5th percentile −0.003%, median spread
0.09%), so most of the "lag" in candles is a missing trade, not a stale quote anyone can hit. Candles cannot tell the
two apart; the gap logger's real books (`lag-replay`) decide. Lag-buy does not go live unless the replay passes.

### 7.1m Event gate: Trump, Fed, Binance and news (PLAN.md §15 items 5 + 5b)
Code: `engine/sentiment/events.py` (gate), `engine/sentiment/fed.py` (calendar + statement scorer); config
`sentiment.events`. It replaces the 30-minute headline monitor and only ever reduces risk.
- **Sources** (free, polled in a background thread): Trump's Truth Social posts (trumpstruth.org, 90 s), Fed
  monetary releases (60 s; every 10 s around a calendar event), Fed speeches and other releases, Binance
  announcements (listings, delistings, news), CoinDesk, Cointelegraph, Decrypt, The Block, Arzdigital, BBC World.
  Trump, BBC and minor Fed items are scored only when they match a keyword list (tariffs, China, Iran, oil, war, Fed, crypto, ...).
- **Scoring:** gpt-6-luna scores every item in US-dollar terms (score −1..+1, severity 0-3, category, assets,
  novelty). gpt-6.1-sol scores FOMC statements, minutes and chair speeches against the previous statement
  (stance −2 hawkish .. +2 dovish, surprise 0-1) and must confirm any sell decision.
- **Actions:** a fresh (≤ 60 min) item with severity ≥ 2, score ≤ −0.5, novelty ≥ 0.5 in a risk category blocks new
  entries for 3 h; a confirmed severity-3 shock (score ≤ −0.7), or a hack/delisting of a held coin, sells to USDT.
  Fed: no entries from 2 h before an FOMC decision or chair speech (30 min before minutes, CPI, payrolls) until
  15 min after; then hawkish surprise (stance ≤ −1, surprise ≥ 0.5) blocks 4 h (stance −2 also sells), dovish
  waits 15 min (no chasing), as-expected pauses 30 min. Positive news never triggers a buy.
- **Calendar:** FOMC decisions are parsed from federalreserve.gov (cached in `data/fed_calendar_auto.json`);
  chair speeches, CPI and payrolls go in `data/fed_calendar.yaml`. No Fed event falls inside the challenge window;
  the module is a permanent part of the strategy for use after the challenge.
- **Dataset:** every item, scored or not, is appended to `data/sentiment/events/<date>.jsonl` so sentiment can be
  tested as a model feature once enough history exists. Gate actions go to the decision log (`event_gate`,
  `event_gate_confirm`, `event_gate_skip`).
- **Checks:** a full first poll of all 11 sources costs about $0.009. Replay of the last 8 FOMC statements
  (`runs/fed_replay.csv`): the scorer rated the four Apr-Sep 2026 statements hawkish (stance −1) and the four before
  neutral; BTC's next-4 h move averaged −0.87% after the hawkish ones vs −0.22% after the neutral ones (largest
  −2.8% on 2026-06-17). No statement scored a surprise ≥ 0.5, so each would have paused entries 30 min. Eight
  events are too few to tune on, so the rules stay as written.

### 7.1n Live setup: USDT pairs, idle USDT, hard limits (Thu 2026-10-22 22:20 Tehran)
User rule: trade USDT pairs and hold USDT when idle. `config.yaml` now runs model **U1** (`models/u_U1`, USDT pairs
BTC, ETH, XRP, SOL, PEPE, XAUT, BNB with Binance outside data; context BTCUSDT + USDTTMN; fee 0.13% per side).
The account is 5,000,000 TMN ≈ **18.3 USDT** at 1 USDT ≈ 272,000 TMN; the Toman → USDT conversion happens once at
the start and back at the end (BTC route if USDT/TMN is frozen). The shadow simulates both conversions at the live
USDT/TMN book with the 0.30% Toman taker fee.

**Hard risk limits (PLAN.md §15 item 4):** no new entries after −1.5% on the Tehran day (`daily_loss_limit`);
sell everything and stay flat (in USDT) for the rest of the day at −3% (`daily_halt`); no position may lose more
than 2% (stop capped by `max_stop_pct`); the 30% drawdown kill switch and the Binance crash guard stay.

| U1 backtest (18.3 USDT, 0.13% fees) | return | trades | max DD | profit factor |
|---|---|---|---|---|
| validation 2026-08-01..09-15 | +3.74% | 15 | 1.37% | 3.14 |
| held-out 2026-09-17..10-22 | −0.95% | 32 | 3.56% | 0.92 |

Identical with and without the hard limits: no day lost 1.5%, and the four PEPE entries with stops wider than 2%
never reached their stop. The limits are a safety net for live trading, not a change to the tested strategy.

### 7.1o GPT overlay rework (PLAN.md §15 item 6)
Code: `engine/agent/overlay.py`. Built on what the published agent ablations support (report
`reports/Trading agent ideas review.md`): hard risk rules stay in code, the LLM filters and reflects, no debate.
- **One batched gpt-6.1-sol call per bar** reviews every candidate together (prompts now in US-dollar / USDT terms).
- **Asymmetric veto:** approve by default; a veto needs a reason code (`major_negative_news`, `market_shock`,
  `exchange_or_regulatory_risk`, `illiquid_or_stale_market`, `extreme_move_already_happened`), otherwise it is
  counted as an approval and the note is logged. This protects trade count from LLM over-caution.
- **Exit plan at entry:** take-profit and stop come from the model; the agent writes a one-sentence invalidation
  condition, stored with the position. The hourly exit review may close a position early only when that
  condition is met and it quotes the evidence (`agent_exit` records both).
- **Reflection** every 8 h (`agent.reflect_every_h`): closed trades plus the agent's own entry reviews become 3
  short lessons (`agent_cache/lessons.json`), added to later reviews as hints, never rules (`agent_reflection`).
- **Counterfactuals:** `python cli.py veto-report runs/<run>` replays every vetoed and event-gate-skipped entry
  on 1-minute candles with the model's take-profit / stop / time stop, so the filters' value is measurable.
- Checked with one real call (XRPUSDT candidate): approved with confidence 0.78 and the invalidation "Close early
  if confirmed adverse Ripple/XRP regulatory news emerges or Binance XRPUSDT falls 1.1% below the entry price."

### 7.1p Fixing U1: measured costs, fewer pairs, Binance-trained pooled model (PLAN.md §15b)
Rule fixed in advance: pick the best validation return among candidates with profit factor ≥ 1.2 and ≥ 10 trades,
run it once on the held-out month, and adopt it only if it beats U1's held-out −0.95%; otherwise run A.
All candidates use per-pair costs measured by the gap logger (half the median Wallex spread per side: BTC 0.03%,
ETH 0.09%, XRP 0.08%, SOL 0.10%, PEPE 0.23%, BNB 0.35%, XAUT 0.02%) instead of a flat 0.15%.

| candidate | validation (08-01..09-15) | held-out (09-17..10-22) |
|---|---|---|
| A: U1, 7 pairs | +4.45%, 13 trades, PF 5.03, DD 1.29% | −1.41%, 32 trades, PF 0.87, DD 4.42% (for the record) |
| B: U1 without BNB, XAUT | +4.45%, 13 trades, PF 5.03 (tie: BNB/XAUT never traded) | **−1.41%**, 32 trades, PF 0.87 |
| C: Binance 1h, 30 coins pooled, 2 years | **0 trades** (max EV −0.31%; validation AUC 0.562) | not run (ineligible) |

The tie between A and B was broken in favour of B (identical validation, less cost and no blank features). B's
held-out −1.41% does not beat −0.95%, so **A runs** (the live config: U1, 7 pairs, measured costs).
Findings: (1) a model on Binance data alone has almost no edge at 1 h (walk-forward AUC 0.50-0.55): U1's edge comes
from Wallex lagging Binance, which a Binance-only model cannot see; (2) with real costs U1's held-out result is worse
(−1.41% vs −0.95%) because PEPE, 20 of 32 held-out trades, costs 0.23% per side, not 0.15%. Dropping PEPE was not
pre-registered and is not applied (choosing it after seeing the held-out month would be overfitting).

### 7.1q Longer history and extra features (PLAN.md §15c)
Same windows, per-pair costs, risk settings and pick rule as §7.1p (rule fixed before any result).

| candidate | training data | validation AUC | walk-forward AUC | validation backtest |
|---|---|---|---|---|
| D: U1 design, long history | Wallex hourly 2021+ (7 pairs) + Binance 2017+ | 0.633 | 0.58-0.62 | **0 trades** (max EV −0.25%) |
| D+: D + 12 extra features | same | 0.636 | 0.58-0.62 | **0 trades** (max EV −0.25%) |
| E: Binance only, 30 coins | Binance hourly 2017+ (1.71M rows) | 0.594 | 0.54-0.56 | **0 trades** |
| E+: E + 12 extra features | same | 0.593 | 0.54-0.56 | **0 trades** |

None is eligible (rule: ≥ 10 validation trades, PF ≥ 1.2), so **A keeps running**. Longer history improves ranking
(AUC 0.63 vs 0.56 for two years of Binance data), but the models' win probabilities stay near the long-run
average (24-31%) and never clear the cost of a trade with the 1.2% minimum target: they rank better but are
never confident enough to trade. The extra features changed AUC by less than 0.005. Candidate F (GRU) was not
trained: the user decided to wait for these results first.

**Candidate F (GRU, trained after D-E+ on the user's request):** PyTorch GRU on the D+ data, input = the last 72
hourly feature rows (59 features), 250,035 training windows, early stopping on the 60 days before validation
(stopped after epoch 2; early-stop AUC 0.616), isotonic calibration and EV threshold on the validation window,
2 minutes on the RTX 3070. Validation AUC 0.650 (best USDT model) and validation backtest **+9.46%, 35 trades,
PF 1.92, DD 4.32%**, the only eligible candidate of §15c, so it was run once on the held-out window:
**−5.98%, 21 trades, PF 0.36, DD 7.22%** (PEPE 12 of 21 trades) → fails (needs ≥ 0% and > −1.41%). Its validation
score was inflated because calibration and the threshold were tuned on that same window; held-out it is worse
than U1. Not used.

### 7.1r Maker quotes priced from Binance: entry rule (PLAN.md §15 item 3)
Rule fixed in advance: quote only on pairs with median spread ≥ 0.30% **and** ≥ 20 Wallex trades per hour.
Gap logger, Thu 21:48 - Fri 01:52 Tehran (4.1 h, overnight):

| pair | median spread | trades / hour | USDT volume / hour |
|---|---|---|---|
| BNB | 0.93% | 6.1 | 1,268 |
| PEPE | 0.46% | 13.8 | 1,533 |
| SOL | 0.22% | 11.8 | 1,938 |
| XRP | 0.19% | 20.1 | 8,228 |
| ETH | 0.17% | 9.8 | 1,246 |
| BTC | 0.06% | 36.8 | 8,342 |

**No pair qualifies:** the wide-spread pairs barely trade and the active ones are too tight for two maker fees
(0.20%). For information, a replay on the logged trades of a bid at Binance-fair − 0.5% on PEPE and BNB (never
above the best bid) got **0 fills in 4.1 h**. Market making is not built. Caveat: an overnight sample; daytime flow
is likely higher, but PEPE would need about 1.5x more trades per hour to qualify.

### 7.1s Live strategy switched to S1, Toman pairs (user decision Fri 2026-10-23 02:00)
After every USDT-pair approach proved weak (§7.1l-§7.1r, §7.2a-b), the user chose the most profitable Toman model:
**S1** (16 Toman pairs, idle Toman, `models/latest.joblib`, one position at up to 90% of equity, §7.1k).
Re-run with today's code (real fees 0.30%, crash guard on):

| S1 | validation 08-01..09-15 | held-out 09-17..10-22 |
|---|---|---|
| original limits (8% daily stop, 30% kill switch) — **live** | +5.41%, 34 trades, PF 1.34, DD 4.96% | **+20.47%**, 42 trades, PF 2.95, DD 3.68% |
| with the U1 hard limits (1.5% / 3% daily, 2% stop cap) | +4.78%, 31 trades, PF 1.28, DD 5.95% | +17.37%, 41 trades, PF 2.68, DD 3.69% |

The hard limits are off for S1: they cost 0.6 and 3.1 points (Toman pairs swing more and the 2% cap cut trades
that recovered), and the user ranks profit above risk. In dollars S1 made −10.96% and +3.38% in the same windows:
most of the Toman gain is the rial falling. The GPT overlay and the event gate switch their prompts to Toman terms
automatically (USDT/TMN moves count). Lag-buy book replay for the record: 0 signals in 4.3 h of real books.

### 7.1t S1 without Binance data (Liara, Fri 2026-10-23)
On Liara every Binance request failed with "Connection refused" (network block in Iran). Held-out backtest of S1
with all Binance-based inputs blank (candles, funding, crash guard), i.e. what the bot would compute there:
**+2.33%, 14 trades, PF 1.30, DD 5.63%** vs **+20.47%, 42 trades, PF 2.95** with Binance. The live runner now skips
new entries (logged `outside_data_missing`) when more than half of the pairs lack their Binance candle; exits still
run. Fix on Liara: `BINANCE_SPOT_URL=https://data-api.binance.vision` (Binance's public market-data mirror) if
`check-net` shows it reachable, else `HTTPS_PROXY` through a proxy outside Iran with `NO_PROXY=api.wallex.ir,ai.boxapi.ir`.

### 7.1g Toman pairs vs USDT pairs (validation window, real fees)

Fees: Toman markets maker 0.25% / taker 0.30%; USDT markets maker 0.10% / taker 0.13%. Both
7-coin models were trained with the identical pipeline. The USDT results are converted to the
Toman score: one TMN→USDT conversion at the start and one back at the end through USDT/TMN
(0.30% taker each), plus the rial move during the window.

| | 16 Toman pairs (live setup) | 7 Toman pairs | 7 USDT pairs |
|---|---|---|---|
| trades (45 days) | 41 | 1 | 44 |
| net return in own quote | +1.32% | +0.56% | +0.55% (USDT) |
| 3-day mean, own quote | +0.086% | +0.04% | **+0.004%** |
| 3-day mean, **Toman score** | +0.086% | +0.04% | +0.66% |
| 3-day windows positive (Toman) | 37% | 7% | 58% |
| 3-day worst / best (Toman) | −3.31% / +4.87% | 0 / +0.56% | −3.17% / +6.38% |

**Reading.** The USDT model has a better walk-forward AUC (0.57–0.64) and pays lower fees, but its
trading edge in USDT terms is about zero (+0.004% per 3 days). Its better Toman score comes almost
entirely from holding USDT while the rial weakened: USDT/TMN rose 1.29% per 3-day window on
average in the validation period. That is a currency bet, not trading skill.
- **Size of the rial effect:** over the full year USDT/TMN rose 150%. Over 3 days it averaged
  +0.78% and rose in 61% of windows, but swung from −12.1% to +17.5%.
- **Freeze risk:** USDT/TMN had no trades for about 14 hours on the night of 2026-10-21. A freeze
  near the end of the challenge would leave the account outside the "end in Toman" rule.

Recommendation: keep trading Toman pairs, and use USDT and Binance prices only as
signals (crash guard).

### 7.1h Aggressive mode (validation window, real fees)

Same model, concentrated positions on the 7 most volatile allowed pairs (SUI, AVAX, PEPE, SHIB,
DOGE, LINK, UNI). The "let winners run" variants use exit take-profit 3 ATR and a 24h time stop.

| | current setup | **A1: 1 × 90%, volatile** | A2: A1 + TP 3 ATR | A3: 2 × 45%, TP 3 | A4: 2 × 45%, all 16, TP 3 |
|---|---|---|---|---|---|
| trades (45 days) | 41 | 25 | 23 | 34 | 37 |
| net return | +1.32% | **+3.86%** | −1.27% | −0.59% | +0.80% |
| max drawdown | 5.59% | 7.97% | 7.38% | 5.03% | 5.05% |
| mean per day | +0.03% | **+0.087%** | −0.03% | −0.01% | +0.02% |
| days ≥ +1% / ≥ +2% | 7% / 2% | 9% / **7%** | 7% / 4% | 9% / 4% | 7% / 4% |
| losing days | 18% | 20% | 20% | 20% | 20% |
| best / worst day | +4.45% / −2.59% | +6.63% / −4.11% | +4.43% / −4.12% | +2.22% / −3.13% | +2.85% / −3.13% |
| 3-day mean | +0.09% | **+0.28%** | −0.17% | −0.09% | −0.04% |
| 3-day windows ≥ +3% | 5% | **12%** | 5% | 0% | 2% |
| 3-day worst / best | −3.31% / +4.87% | −5.57% / +9.08% | −5.57% / +5.14% | −3.94% / +2.23% | −3.94% / +3.80% |

**Reading.**
- **A1 works best.** Concentrating on volatile pairs roughly triples the average return, and gives a
  real chance (7% of days) of +2% or more. Losses also grow: worst day −4.1%, worst 3 days −5.6%.
- **Letting winners run (TP 3 ATR) hurts.** Trailing stops close most trades before the far target,
  so the model's own target works better.
- **Caveat:** 25 trades is a small sample, and this window was also used for calibration. Treat the
  numbers as indicative.

No setting reaches 1–2% per day on average.

### 7.2a Trend following on daily / 4-hour bars (PLAN.md §16, for use after the challenge)
`python cli.py trend-backtest` (`engine/trend/backtest.py`). Long-only, idle USDT, 6 coins (BTC, ETH, XRP, SOL,
BNB, PEPE), each 1/6 of equity while close > SMA(N) and the N-bar return > 0; trades at the next open; taker fee
plus measured Wallex half-spread per side; Binance prices. Rule fixed in advance: pick the best mean month on
2018-2024 among variants with max drawdown ≤ 35%, then run once on 2025-01..2026-10.

| bar, N | mean month | median month | positive months | max DD | trades/month |
|---|---|---|---|---|---|
| 1d, 20 | +6.39% | −0.48% | 45% | 48.1% | 14.5 |
| 1d, 50 | +6.92% | 0.00% | 44% | 48.0% | 8.1 |
| 1d, 100 | +5.65% | 0.00% | 42% | 54.5% | 6.1 |
| 4h, 20 | +5.86% | −0.58% | 46% | 46.9% | 33.3 |
| 4h, 50 | +6.76% | 0.00% | 44% | 54.1% | 18.3 |
| 4h, 100 | +5.46% | −0.80% | 35% | 57.9% | 14.1 |

Benchmarks 2018-2024: equal-weight buy-and-hold +7.29% mean month with an 80.1% drawdown; BTC +4.49%, 81.2%.
**Result: no variant meets the 35% drawdown limit, so nothing is confirmed.** For information only, the best
development variant (1d, N = 50) on 2025-01..2026-10: +7.4% total, +0.57% mean month (median −0.9%, 27% positive
months), max drawdown 29.6%, 9.9 trades/month, while equal-weight buy-and-hold lost 30.7% and BTC 10.1%.
Trend following cut drawdowns from ~80% to ~50% and kept capital through the 2025-26 decline, but its average
month was driven by a few huge months (best +129%) in the 2018-2024 bull markets; it did not deliver 4% a month
recently. Not used in the challenge.

### 7.2b Wallex-vs-Binance premium mean reversion, hourly (PLAN.md §17)
`python cli.py premium-backtest` (`engine/stat/premium.py`). Buy at the next Wallex open when the Wallex/Binance
premium is k standard deviations below its 7-day median; sell when it is back to normal or after H hours; 1/6 of
equity per pair; taker fee + measured half-spread per side. Rule fixed in advance (pick on 2021-2024, confirm once).

| k, H | dev 2021-24 trades | dev PF | dev total | dev max DD |
|---|---|---|---|---|
| 2.0, 6 h (pick) | 2,212 | 2.48 | +1,111% | 15.0% |
| 2.5, 6 h | 1,034 | 3.87 | +660% | 5.6% |
| 3.0, 6 h | 535 | 7.17 | +516% | 2.7% |

**Held-out 2025-01..2026-10 (pick, once): 2,004 trades, profit factor 0.74, −41.8%, max drawdown 46.7% → fails.**
Every pair was flat or negative (BNB worst, −0.64% per trade). The edge was real in 2021-2024, when Wallex USDT
books were thin and slow, and has disappeared since: the gap logger shows today's Wallex quotes sitting close to
Binance (§7.1l), i.e. market makers now price off Binance. This is the clearest example in the project of why
results are confirmed on a later window: the best development result of all strategies is the worst held-out one.

### 7.2c Improved GRU, 12-month walk-forward (PLAN.md §18)
`engine/model/walkforward.py`. G = GRU (48 h window, hidden 32, dropout 0.3, 3 epochs, 3 seeds averaged) on the D+
data; per test month it is fitted on all earlier data, early-stopped on days −120..−60 and calibrated (isotonic +
EV threshold) on days −60..0, then the month is traded with the normal backtest. T = tree ensemble, same scheme.

| 12 months 2025-11..2026-10 | total | profit factor | trades | months ≥ 0 | passes |
|---|---|---|---|---|---|
| G (GRU) | **+8.87%** | 2.72 | 13 | 11 / 12 | no (needs ≥ 30 trades) |
| T (trees) | +7.99% | 1.84 | 18 | 12 / 12 | (reference) |

With honest calibration the GRU is profitable and steady over a year (only June 2026 lost, −2.55%), but it trades
about once a month: 8 of 12 months had no trade at all. It fails the pre-registered trade-count condition and is
far from the user's activity goal. Not used.

### 7.2d S1 with more trades per day (PLAN.md §19)
| variant (S1 model, Toman pairs) | validation return | trades (per day) | PF | held-out |
|---|---|---|---|---|
| S1 as live: 1 position at 90% | +5.41% | 34 (0.76) | 1.34 | **+20.47%**, 42 trades (1.2/day), PF 2.95 |
| M1: 3 positions at 30% | +3.16% | 39 (0.87) | 1.55 | not run |
| M2, M3: + EV > 0, p ≥ 0.35 | identical to M1 | | | not run |
| M4: 5 positions at 18%, 4 entries/bar (pick) | +1.69% | 40 (0.89) | 1.47 | +4.24%, 59 trades (1.7/day), PF 2.18, DD 1.0% |

Looser filters add almost no trades: of 205 coin-hours the model rated positive in validation, 142 were UNI and
SUI in consecutive hours while already held (the bot never adds to a held coin). The model's natural rate is about
one round trip a day; more positions spread the same signals over smaller sizes, so return falls (M4 held-out
+4.24% vs +20.47%). M4 passes its held-out check but no variant reaches 2 trades a day.

### 7.2 Round-1 backtest of record (held-out 2026-09-16 .. 2026-10-21)

Source: `runs/bt-20261021-172454/report.md`.

| metric | value |
|---|---|
| net return | +0.01% |
| max drawdown | 0.43% |
| trades | 3 (2 take-profit, 1 stop-loss) |
| rolling 3-day challenge windows | 33 windows, median 0 trades, worst −0.39%, best +0.40%, 100% end in TMN |

**Honest reading:** risk control works (tiny drawdown, always flat in TMN at the end), but at this
cost level the model has almost no trades it is willing to take in the test month. That would score
poorly on return and activity.

### 7.3 Selection model (two-stage) — rejected

The rank IC between predicted and realized relative 24h returns was +0.004 (positive in 48% of 4h
slots), which is no skill. With 8 pairs, the two-stage model made the same 3 trades as plain EV
ranking, so it is disabled (`strategy.selection.enabled: false`). Source: `runs/selection-*/report.md`.

### 7.4 Fee sensitivity

| fee per side | OOF+val trades (EV>0) | mean net | held-out backtest |
|---|---|---|---|
| 0.25% | 110 | +1.13% | +0.01%, 3 trades |
| 0.15% | 247 | +0.98% | — |
| 0.10% | 508 | +0.76% | −0.28%, 26 trades, win rate 46% (sensitivity run only, `runs/bt-*-fee10bps-sensitivity`) |

Lower fees produce many more trades, but in the held-out month the win rate fell from 55–67%
(earlier out-of-sample periods) to 46%. The hourly-candle edge is too small to rely on by itself.
The next research step is in PLAN.md (fee confirmation and a minute-level fair-value strategy).

## 8. Re-running and checking the results

```bash
python cli.py download      # same pairs, same window -> data/manifest.json lists row counts + hashes
python cli.py train         # deterministic (fixed seed); writes the walk-forward report
python cli.py backtest      # same model + data -> identical decisions.jsonl, trades.csv, report.md
python cli.py shadow        # same pipeline on live data without placing orders
```

- **What each run stores:** the model version, config hash and data hashes in `meta.json`.
- **Decision log:** every live trade can be traced in `decisions.jsonl`, from `scan`
  (all pairs' scores) to `entry_signal` (features, ML, agent and risk checks) to `fill` and `exit`,
  all joined by `decision_id`. A trade without such a chain did not come from the algorithm.
