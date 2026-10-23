# Market Making and Spread Capture on Thin Crypto Spot Books (small account, Wallex hackathon context)

Context assumed: Wallex USDT pairs (BTC, ETH, XRP, SOL, PEPE, BNB, XAUT), spot only, REST at about 1-2 s per call, maker 0.10% / taker 0.13%, minimum order about $1, account about $2,700, roughly one day of live trading left. Scoring rewards dollar return (a loss scores zero), drawdown, trade count and stability.

## Q1. How Hummingbot PMM, Avellaneda-Stoikov and grid bots set spreads, refresh, inventory and kill conditions, and what users report

### Takeaway
Hummingbot's pure market making (PMM) quotes are a fixed percentage offset from a reference price, either the local mid or an external exchange. A linear inventory skew scales order sizes between 0x and 2x. Orders are cancelled and replaced every `order_refresh_time` seconds, with a tolerance band that avoids needless replacement. The Avellaneda-Stoikov variant replaces the fixed offsets with a reservation price shifted by inventory times gamma times volatility, plus a spread term, floored at `min_spread`. Practitioners mostly report break-even to small profit, losses during sharp or trending moves, and a habit of starting wide and narrowing spreads by hand.

### Cited findings

**Hummingbot pure_market_making (v1), parameters and defaults**
- `bid_spread` and `ask_spread` are each side's percentage distance from the reference price. `order_levels` defaults to 1 and sets the number of orders per side. `max_order_age` is 1800 s: orders are replaced after that even if the price is unchanged. `order_refresh_tolerance_pct` defaults to 0 and sets how far price must move before orders are refreshed. `filled_order_delay` defaults to 60 s, the wait after a fill before new orders go out. — [Hummingbot PMM docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)
- Further PMM settings and defaults ([same docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)):
  - `inventory_skew_enabled` (False), `inventory_target_base_pct` (50) and `inventory_range_multiplier` (50, in multiples of total order size).
  - `hanging_orders_enabled` (False) keeps the unfilled side alive after a fill. `hanging_orders_cancel_pct` is 10%.
  - `price_ceiling` and `price_floor`: above the ceiling only sells are placed, below the floor only buys.
  - `minimum_spread` (-100) cancels orders whose spread falls below it.
  - `ping_pong_enabled` alternates buys and sells after fills.
  - `take_if_crossed`.
  - `price_source` can be `current_market`, `external_market` or `custom_api`, with `price_source_exchange` / `price_source_market` and `price_type` (default mid_price).
- Exact inventory skew algorithm, from `inventory_skew_calculator.pyx` ([source](https://raw.githubusercontent.com/hummingbot/hummingbot/master/hummingbot/strategy/pure_market_making/inventory_skew_calculator.pyx)):
  - total = base × price + quote; base_value = base × price.
  - range_value = min(range_in_base × price, 0.5 × total).
  - target = total × target_ratio; the band is [max(target − range_value, 0), target + range_value].
  - base_value maps linearly onto 0 to 1 across the band, with target at 0.5. Values beyond the band are pinned to the ends.
  - bid_ratio runs from 2 down to 1 when under-weight and from 1 down to 0 when over-weight. ask_ratio = 2 − bid_ratio.
  - The ratios multiply order **size**, not spread. At the band edge one side's size goes to 0.
- Hummingbot also offers a separate cross-exchange market making strategy. It places limit orders on a less liquid maker exchange, hedges fills with taker orders on a liquid exchange, and keeps quotes wider than taker prices by `min_profitability`. — [Hummingbot XEMM](https://www.hummingbot.org/strategies/cross-exchange-market-making)

**Hummingbot avellaneda_market_making**
- Parameters ([Hummingbot A-S docs](https://hummingbot.org/strategies/v1-strategies/avellaneda-market-making/)):
  - `risk_factor` (γ) and `order_amount_shape_factor` (η).
  - `min_spread` (default 0).
  - `inventory_target_base_pct` (50).
  - `volatility_buffer_size` and `trading_intensity_buffer_size` (200 ticks each).
  - `execution_timeframe`: `infinite`, `from_date_to_date` or `daily_between_times`.
- Exact implementation, from `avellaneda_market_making.pyx` ([source](https://raw.githubusercontent.com/hummingbot/hummingbot/master/hummingbot/strategy/avellaneda_market_making/avellaneda_market_making.pyx)):
  - q = (base_balance − q_target) / total_inventory_in_base, a normalized deviation.
  - reservation = mid − q·γ·σ·T_frac, where σ is absolute price volatility and T_frac = time_left/closing_time. In infinite mode T_frac is 1; the code comments call this a fix for errors in the paper's infinite-horizon equations.
  - optimal_spread = γ·σ·T_frac + (2/γ)·ln(1 + γ/κ). κ is estimated from trading intensity, meaning the rate at which order-book depth gets consumed.
  - min_spread% is converted to price. Quotes are clamped as ask = max(res + spread/2, mid + min_spread/2) and bid = min(res − spread/2, mid − min_spread/2).
  - η scales sizes: buy size × exp(−η·q) when long and sell size × exp(η·q) when short, so the bot shrinks the side that would add inventory.
- Hummingbot's guide: a higher γ moves the reservation price more aggressively toward the inventory target, and a γ near 1 is described as conservative. A high κ means a denser book and smaller spreads. An "easy mode" derives γ and κ from user min/max spread limits. — [Hummingbot A-S guide](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/)

**Grid bots**
- Spacing can be arithmetic (equal price steps) or geometric (equal percentage steps). Guides recommend geometric for volatile crypto. The most concrete sizing rule found: profit per grid should be at least 2 to 3 times the round-trip fee. — Aggregated guides via [HaasOnline simple grid bot](https://help.haasonline.com/docs/haasscript/community-bots/simple-grid-bot-spot) and [Altrady grid bot processing](https://support.altrady.com/en/article/grid-bot-processing-2agdtp/). These are vendor sources; treat the rule of thumb as practitioner convention, not evidence.
- The academic "Dynamic Grid" paper uses a geometric grid with ratio k and n+1 levels, and resets the grid around the current price when price breaks out of the range. — [Chen, Chen, Jang 2025, arXiv 2506.11921](https://arxiv.org/html/2506.11921v1)

**Reported user results (mostly Hummingbot-sponsored, likely biased upward)**
- Liquidity miner "MrBig" ran 20 PMM instances across all campaign pairs ([Hummingbot interview](https://hummingbot.org/blog/interview-with-liquidity-miner-mrbig/)):
  - He started with wide spreads and narrowed them until fills stopped losing money net of fees.
  - He lost money early, especially in sharp swings, and became profitable only with liquidity-mining rewards included. He claims about 0.6 to 0.7% per day, self-reported.
  - His rules: use `kill_switch_enable: True`, avoid very trendy markets, and do not quote at the top of the book ("the market will kill you").
- Other Hummingbot miners reported performance "around break-even, possibly slightly profitable". An academy post showed a downtrending pair at a net loss even after rewards, while an uptrending pair was profitable. — [Hummingbot interview](https://hummingbot.org/blog/interview-with-liquidity-miner-david-salter/), [Hummingbot academy](https://hummingbot.org/academy-content/sharing-is-caring-3-trading-pairs-i-picked/)
- In a 2022 FIRO liquidity-mining campaign, 90% of bots earned $0.03 or less in rewards in a week; capital losses were not measured. — [FIRO forum](https://forum.firo.org/t/2022-firo-hummingbot-campaign/2290?page=2)

### Inferences
- For a REST bot with 1-2 s latency, Hummingbot's defaults are not tuned for us: refresh about 30 s, tolerance 0, fill delay 60 s. A practical setup is a refresh every 15-30 s, a tolerance of about 0.05-0.1%, and a much shorter `filled_order_delay` (about 5-10 s) so fills repeat. Two cancel+place calls per side per cycle at 1-2 s each cap throughput at roughly one requote per pair every 5-10 s.
- The kill condition used in practice is a simple PnL kill switch (stop if PnL < −X%) plus price ceiling/floor bands. Both are easy to copy.
- With one day left, the A-S infinite-horizon form reduces to a fixed spread plus a linear inventory shift. Hand-rolled code is enough: reservation = ref_mid × (1 − k_inv·q) with k_inv about 0.2-0.5%, and half-spread = max(fee floor, c·σ).

### Gaps
- No independent, audited PnL statistics exist for Hummingbot PMM/A-S users; the available reports are self-reported and promotional.
- Binance, Pionex and Bybit do not publish statistics on the share of profitable grid-bot users. Leaderboard ROIs, such as a 7-day PEPE grid at 19.44%, are survivorship-selected ([CriptoNoticias](https://criptonoticias.com/finanzas/estrategias-mas-rentables-trading-bots-ia-binance-okx-bybit-okcoin)).
- I did not verify Freqtrade's grid-like strategies; Freqtrade is primarily a candle-based directional framework.

## Q2. Is grid / mean-reversion spread capture profitable after ~0.10% maker fees? Regimes and pausing

### Takeaway
The one rigorous study found shows that a static grid has zero expected value under a random walk before fees and negative expected value after fees. Profits come only from range-bound (mean-reverting) regimes. Practitioners stay out of, or pause in, trending and high-volatility moves. Adaptive resetting grids improved risk-adjusted results in backtests.

### Cited findings
- Under a 50/50 move-of-k random walk with no fees, the paper's verdict is "the expected value of a grid strategy is zero." Inventory loss in a decline is E(G) = −(M/n)(n²/8 − n/4), exactly offset in expectation by arbitrage cycles. After fees, small grid sizes lose. — [arXiv 2506.11921](https://arxiv.org/html/2506.11921v1)
- The same paper backtests minute-level BTC and ETH from January 2021 to July 2024 with a 0.08% per-side maker fee. Its Dynamic Grid (reset around the current price on breakout) reached IRR of about 60-70% across parameter sets with lower max drawdown than buy-and-hold. In an ~80% ETH decline its drawdown was still about 50%. — [arXiv 2506.11921](https://arxiv.org/html/2506.11921v1). This is a single paper without independent replication, and figures were read from text and plots.
- A Binance Square backtest of a BTC/USDT grid in April 2025 at a 0.1% fee returned 7.35% against 14.15% for buy-and-hold, an uptrend underperformance. — [Binance Square #gridbot](https://www.binance.com/square/hashtag/gridbot) (anecdotal)
- Practitioner regime rules: avoid "very trendy markets", treat stable markets as the market maker's best friend, and note that losses concentrate in sharp swings. — [Hummingbot MrBig interview](https://hummingbot.org/blog/interview-with-liquidity-miner-mrbig/)
- A symmetric strategy in a downtrend accumulates the base asset and loses value; A-S counters this through the reservation price. — [Hummingbot A-S guide](https://hummingbot.org/blog/guide-to-the-avellaneda--stoikov-strategy/)

### Inferences
- Fee math for Wallex USDT is my own derivation. A round trip of two maker fills costs 0.20%, so the full quoted spread (bid to ask) must exceed 0.20% plus expected adverse selection. A floor of about 0.30-0.40% total (0.15-0.20% per side from mid) is the minimum. On volatile names such as PEPE, SOL and XRP, use about 0.5-0.8%.
- If the bid fills while price moves against us and we must exit with a taker order, the cost is 0.10% + 0.13% = 0.23% plus the move. Avoid taker exits except on the kill switch.
- Cheap regime detection, implementable in minutes:
  - Pause the side that buys into a trend when the Binance 5-15 min return exceeds about ±0.5-1% (adjusted to the coin's volatility).
  - Also pause it when the 1-min EMA slope stays consistently signed.
  - Pause both sides when realized 1-min volatility exceeds about 2-3x its trailing 1-hour median.
  - Keep sell-only (unwind) mode active while inventory is above target.
- Because scoring zeroes a losing account, the daily objective is "positive, low drawdown, many trades". Spreads should err wide, inventory caps tight, and the stablest pairs (XAUT, BTC, ETH) get the most capital.

### Gaps
- I found no peer-reviewed measurement of grid profitability net of a 0.10% fee on thin, non-Binance venues. Thin books mean wider natural spreads, which helps capture, but fewer fills and more toxic flow, and the net effect is unmeasured.

## Q3. Using an external reference price (Binance) to avoid adverse selection on a slower exchange

### Takeaway
Hummingbot natively supports quoting off another exchange's mid (`price_source: external_market`). Its XEMM strategy formalizes making on the illiquid venue with a profitability margin over the liquid venue's price. Its rationale is that the more liquid market tends to lead price discovery. No published study quantifies the adverse-selection savings.

### Cited findings
- `price_source` = `external_market` with `price_source_exchange` and `price_source_market`; `price_type` can be mid, last, best bid or best ask. — [Hummingbot external price source](https://hummingbot.org/strategy-configs/external-price-source), [PMM docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)
- Hummingbot's price-source article argues the more liquid market leads price discovery, so quoting off it helps on a thinner venue. — [Hummingbot academy, price source](https://hummingbot.org/academy-content/commands-and-configs---price-source/)
- XEMM keeps maker quotes on the thin venue wider than the liquid venue's executable price by `min_profitability`. — [Hummingbot XEMM](https://www.hummingbot.org/strategies/cross-exchange-market-making)
- Lead-lag between major exchanges is not fixed; leadership shifts with liquidity, activity and time of day, based on 1-second Binance vs OKX cross-correlation. Centralized venues, Binance among them, lead DEXs in ETH price discovery. — [CoinAPI lead-lag tutorial](https://www.coinapi.io/tutorials/analyzing-lead-lag-effects-between-exchanges-flat-files), [arXiv 2506.08718](https://arxiv.org/html/2506.08718v1)

### Inferences (for Wallex)
- Fair value: fair_USDT = Binance mid for the same coin/USDT (XAUT is also on Binance spot or similar venues). If Wallex carries a persistent premium or discount, use fair = Binance_mid × (1 + basis), with basis an EMA (about 30-60 min) of (Wallex_mid / Binance_mid − 1).
- Quote bid = fair × (1 − h_b) and ask = fair × (1 + h_a). Never quote a bid above fair − fee, and never place a post-only order that crosses the Wallex book.
- Cancel and requote immediately when |fair_now − fair_at_quote| exceeds about 0.3 × half-spread. This is the stale-quote defense; with REST latency, poll Binance every 1-2 s, which is cheap.
- If a Wallex resting order sits on the wrong side of fair (for example, our bid is above the new fair after a Binance drop), cancel it first; this matters more than placing new orders.
- If the Wallex best bid is below fair by more than fee + margin, the Wallex book is stale, so do not step in front.
- MrBig's warning against quoting at the front of the book suggests sitting one tick behind or inside the spread, but never at the touch when the touch is near fair.

### Gaps
- No seconds-level study of Binance leading Iranian exchanges (Wallex, Nobitex) was found. The Wallex/Binance basis and lag must be measured from our own logs.

## Q4. Raising trade count without destroying returns

### Takeaway
Trade count comes from many small maker orders: multiple levels per side, ping-pong / hanging orders, short post-fill delays and several pairs in parallel. Every fill must still clear the 0.20% round-trip fee, so add levels wider rather than tighter, and keep order sizes near the minimum on outer levels.

### Cited findings
- Hummingbot's mechanisms for more fills:
  - `order_levels` with level spacing adds more orders per side.
  - `ping_pong_enabled` alternates sides after fills.
  - `hanging_orders_enabled` keeps the opposite order alive after a fill so the round trip completes.
  - `filled_order_delay` sets how quickly to requote after a fill.
  — [PMM docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)
- A-S η shrinks the size of the side that would increase inventory, which allows continuous quoting on both sides without accumulating inventory. — [Hummingbot A-S source](https://raw.githubusercontent.com/hummingbot/hummingbot/master/hummingbot/strategy/avellaneda_market_making/avellaneda_market_making.pyx)
- Grid profit per level should be at least 2 to 3 times the round-trip fee. — [Altrady](https://support.altrady.com/en/article/grid-bot-processing-2agdtp/) / guides, practitioner rule
- Small grids at 0.08% fees underperform in backtests, so tight spacing loses. — [arXiv 2506.11921](https://arxiv.org/html/2506.11921v1)

### Inferences: concrete starting parameters for Wallex, about $2,700
These are derived starting points, not sourced.
- **Pairs and capital.** Use 3-5 pairs. As a guide, put 25% in BTC, 20% in ETH and 15% in XAUT; split 20% across XRP and SOL; give PEPE 5% or less, or skip it if its book is too thin. Keep 15% or more in USDT as a buffer.
- **Inventory.** Start each pair at 0-30% base, never 50%. Idle money must be in USDT, and spot-only means no shorting, so start mostly in USDT. Buy first and sell what was filled, with a per-pair maximum base inventory of about 40-50% of the pair allocation.
- **Levels.** Use 2-3 levels per side.
  - Level 1 is about $10-30.
  - Further levels are 1.5x the size at +0.15% further out.
  - Level 1 half-spread from fair: BTC/ETH/XAUT 0.15-0.25%, XRP/SOL/BNB 0.25-0.40%, PEPE 0.5% or more.
  - Widen with volatility: half-spread = max(floor, 1.0-1.5 × σ_1min × √(refresh_min)).
- **Skew.** Use Hummingbot-style size multipliers (0-2x) plus a price shift: reservation = fair × (1 − 0.3% × q), with q in [−1, 1] normalized against the max-inventory cap. When inventory is at the cap, place no new bids.
- **Refresh.** Run a 10-20 s cycle per pair. Requote when fair moves more than 0.05-0.1%, and force a refresh after 120-300 s. Requote the filled side 3-5 s after a fill (a ping-pong exit at entry + 2 × half-spread).
- **Take-profit exits.** After a buy fill at p, place the sell at max(p × (1 + 0.25-0.35%), current ask quote). This guarantees each round trip clears fees, like a grid's "sell above buy". Keep it alive (hanging) unless inventory age exceeds about 1-2 h; then lower it to break-even + 0.2% or let skew handle it.
- **Kill and pause rules.**
  - Per-pair stop: cut and go flat (taker allowed) if unrealized loss on inventory exceeds 1.5-2% of the pair allocation, or Binance moves more than 2% against inventory in 15 min.
  - Global stop: stop all bidding if account equity is down 1% or more on the day, and halt entirely at −2%.
  - Trend pause: no bids while Binance 15-min return is below −0.7% (or −1.5% for PEPE and SOL).
  - Pause both sides if the Binance–Wallex basis jumps more than 0.5% (stale feed or outage).
- **Expected activity.** About 1-3 round trips per pair per hour on thin books gives roughly 50-300 fills per day. Each round trip nets about 0.1-0.3% of a $10-30 order, so the day's PnL is small (tens of dollars at most). The main value for scoring is trade count, stability and low drawdown. Do not size up to chase return; a modest positive PnL beats a negative one, which scores zero.

### Gaps
- Fill rates on Wallex USDT books are unknown and need measuring live. Whether resting orders get filled at all at 0.15-0.25% from fair depends on Wallex taker flow.

## GitHub repos

Star counts and last-push dates come from the GitHub API on 2026-10-22, except where marked.

| Repo | Stars | Last push | What it does | Files / modules worth copying |
|---|---|---|---|---|
| [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) | 20,630 | 2026-10-20 | The reference open-source market-making framework (PMM, A-S, XEMM, V2 controllers) | `hummingbot/strategy/pure_market_making/inventory_skew_calculator.pyx` (0-2x size skew); `pure_market_making.pyx` (refresh/tolerance/hanging/ping-pong loop); `avellaneda_market_making/avellaneda_market_making.pyx` (reservation price, spread, min_spread clamp, η sizing); `cross_exchange_market_making/` (maker on thin venue vs reference) |
| [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) | 4,952 | 2025-12-23 | HFT and market-making backtester with queue-position and latency modelling, Binance/Bybit examples | Market-making tutorials (GLFT / grid market-making examples with skew) for queue-aware backtesting. Contents were not verified this session. |
| [ctubio/Krypto-trading-bot](https://github.com/ctubio/Krypto-trading-bot) | 3,727 | 2024-12-15 | Self-hosted C++ high-frequency market-making bot | Quoting-mode and "safety" logic (ping-pong, trade-rate limits); not inspected in detail |
| [Drakkar-Software/OctoBot](https://github.com/Drakkar-Software/OctoBot) | 6,837 | 2026-10-21 | Python multi-exchange bot with grid, DCA and AI modes | Grid / staggered-orders trading modes (tentacles); not inspected |
| [Drakkar-Software/OctoBot-Market-Making](https://github.com/Drakkar-Software/OctoBot-Market-Making) | 42 | 2026-03-30 | Market-making edition of OctoBot for MEXC, Bitmart and others | Market-making order-book config; not inspected |
| [Open-Trader/opentrader](https://github.com/Open-Trader/opentrader) | 2,894 | 2025-06-29 | TypeScript bot with grid and DCA plus UI | Grid strategy module |
| [51bitquant/binance_grid_trader](https://github.com/51bitquant/binance_grid_trader) | 984 | 2026-06-07 | Binance spot and futures grid bot (Python) | Simple grid-loop and order-management code |
| [jordantete/grid_trading_bot](https://github.com/jordantete/grid_trading_bot) | 151 | 2026-10-22 | Modular Python grid engine with backtest, paper and live modes and Grafana | Grid config (arithmetic/geometric spacing, range, take-profit/stop-loss); the GitHub page fetch failed, so contents are unverified |
| [DogsTailFarmer/martin-binance](https://github.com/DogsTailFarmer/martin-binance) | 227 | 2026-10-20 | Adaptive martingale reverse-grid spot system | Reverse-grid / take-profit logic. Martingale sizing is high-risk; avoid for drawdown scoring. |
| [fedecaccia/avellaneda-stoikov](https://github.com/fedecaccia/avellaneda-stoikov) | 731 | 2023-07-06 | Python A-S simulation | A compact reservation-price and spread implementation |
| [mdibo/Avellaneda-Stoikov](https://github.com/mdibo/Avellaneda-Stoikov) | 154 | 2020-05-09 | Python A-S replication | Paper formulas |
| [ragoragino/avellaneda-stoikov](https://github.com/ragoragino/avellaneda-stoikov) | 94 | 2017-10-30 | A-S (2008) replication | Paper formulas |
| [javifalces/HFTFramework](https://github.com/javifalces/HFTFramework) | 308 | 2026-10-18 | Research framework using RL to tune A-S | A-S parameter tuning (too heavy for this hackathon) |
| [joaquinbejar/market-maker-rs](https://github.com/joaquinbejar/market-maker-rs) | 108 | 2026-10-19 | Rust market-making library, A-S first | Clean A-S math module |
| [keanekwa/Optiver-Ready-Trader-Go](https://github.com/keanekwa/Optiver-Ready-Trader-Go) | 101 | 2023-05-19 | A-S bot for the Optiver trading competition | Competition-style A-S with inventory limits, closest analogue to a hackathon |
| [SebastianBoehler/bybit_market_maker_cpp](https://github.com/SebastianBoehler/bybit_market_maker_cpp) | 4 | 2026-08-01 | C++ Bybit-perp market maker with laddered quotes and inventory skew | Ladder plus skew example |
| [HenrisonTao/ftx_grid_trading_bot](https://github.com/HenrisonTao/ftx_grid_trading_bot) | 170 | 2022-08-30 | FTX grid bot, now defunct | Historical only |

Recommendation for hours-scale implementation (my inference): do not adopt a framework, since none has a Wallex connector. Port about 60 lines of logic from Hummingbot:
- `inventory_skew_calculator.pyx` for size skew;
- the A-S reservation-price and min_spread clamp;
- the PMM refresh/tolerance/hanging-order loop;
- `price_source=external_market`-style Binance fair value.
