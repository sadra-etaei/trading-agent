# Lead-lag and cross-exchange strategies: a small exchange (Wallex) following Binance

Context: Wallex hackathon. Spot only, REST only (about 1-2 s per call), USDT pairs (BTC, ETH, XRP, SOL, PEPE, BNB, XAUT). Fees: USDT maker 0.10%, taker 0.13%. Limit orders only. About $2,700 account, about 1 day left. We measured that Wallex follows Binance by about 1-2 min. An earlier fair-value-gap test on Toman pairs found gaps smaller than the spread.

## 1. Academic and practitioner evidence on crypto price discovery and lead-lag (size and persistence of lags)

### Takeaway
Big, liquid venues lead and thin venues lag. Between Western exchanges the lag is seconds: 1-3 s among the majors and up to about 15 s for thinner ones (2018 data). Cross-border premia caused by capital controls (Korea, and by analogy Iran) are much larger and last for days or weeks. But they reflect blocked arbitrage, not a tradeable intraday lag. Cross-exchange arbitrage profits have shrunk sharply since 2018. Our measured 1-2 min lag on Wallex is far longer than lags on Western venues. Iranian venues have not been studied.

### Cited Findings
- Makarov & Schoar (JFE 2020, vol. 135, pp. 293-319) document large, recurring arbitrage gaps across countries. The Korean "kimchi premium" reached up to about 20% over US prices. Unexploited arbitrage profits were at least about $2B (late 2017 to early 2018). Gaps between exchanges in the same country usually stay below about 1%. Gaps open during fast BTC rallies. — [Harvard Law Forum summary](https://corpgov.law.harvard.edu/2019/09/21/trading-and-arbitrage-in-cryptocurrency-markets/); [Decrypt summary](https://decrypt.co/245963/professor-coin-free-lunch-cryptocurrency-markets)
- A Monash paper on the kimchi premium argues that frictions partly sustain it. The trading fee is positively correlated with the no-arbitrage threshold. — [Monash PDF](https://researchmgt.monash.edu/ws/files/590976616/586624035_oa.pdf)
- Crépellière, Pelster & Zeisberger (J. Financial Markets 2023, vol. 64, 100817) confirm the gaps exist. However, they shrank sharply from April 2018 on, and the authors judge them "barely possible to exploit" after that. They link the gain in efficiency to more informed trading. — [Paderborn record](https://ris.uni-paderborn.de/record/34449); [working paper PDF](https://wiwi.uni-paderborn.de/fileadmin-wiwi/cetar/TAF_Working_Paper_Series/TAF_WP_067_CrepellierePelsterZeisberger2022_rev.pdf)
- Schei (CBS MSc thesis, 2019) applies the Hayashi-Yoshida (HY) estimator to 2018 BTC tick data across 7 exchanges. Some exchanges follow others by up to 15 s. Binance leads Kraken by 15 s (lead-lag ratio, LLR, 2.27). Bitstamp leads Kraken by 11 s. Coinbase leads Kraken by 8 s. Poloniex lags Binance, Bitfinex and Bitstamp by 7 s. Binance, Bitfinex and Coinbase lag each other by only 1-2 s. Peak HY correlations are small (0.01-0.09). — [CBS thesis PDF, Table 6.8 p.50, pp.52-53](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- The same thesis finds lead-lag stronger when unexpected volume (information arrival) is high, and when sophisticated traders are active. It names exchange liquidity as the most likely cause of lags. — [CBS thesis abstract](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- HY method: it handles asynchronous tick data without resampling. The lead time is the lag (in ms) that maximizes the absolute lagged HY correlation. LLR > 1 means X leads Y. — [Plazuelo Pascual et al., arXiv:2506.08718](https://arxiv.org/html/2506.08718v1)
- Plazuelo Pascual et al. (2025) compare Binance with Uniswap v2 for ETH over 2024. Centralized venues generally lead price discovery. Futures lead overall, but results are mixed in high-volatility periods. — [arXiv:2506.08718](https://arxiv.org/abs/2506.08718v1)
- A UFSC study of ETH/USDC across Binance, Gemini, Bitstamp and Uniswap V2 finds Binance central to price discovery. Binance has the highest gross spillover to other venues and corrects deviations fastest. — [UFSC repository](https://repositorio.ufsc.br/handle/123456789/247716)
- Alexander & Heck (J. Financial Stability 2020) use minute-level data. Offshore perpetuals and futures (Huobi, OKEx, BitMEX) dominate BTC price discovery. CME futures and US spot venues react to them rather than lead. — [SSRN author page](https://papers.ssrn.com/author=3583843); [Sussex figshare](https://sussex.figshare.com/articles/journal_contribution/Price_discovery_in_Bitcoin_the_impact_of_unregulated_markets/23308262)
- Other studies conflict. Corbet et al. (2018) and Baur & Dimpfl (2019) find spot leads futures at high frequency. Kapar & Olmo (2019) find CME futures lead on daily data. Leadership also shifted during the pandemic. — [Frijns et al. PDF](https://acfr.aut.ac.nz/__data/assets/pdf_file/0009/296424/B-Frijns-Bitcoin_Paper.pdf); [City Univ. PDF](https://openaccess.city.ac.uk/id/eprint/30528/1/Price_Discovery___BTC_Spot_vs_ETP___v4.pdf)
- A CoinAPI tutorial on lead-lag between Binance and OKX finds a best lag of 0. In other words, the two majors are synchronized at its sampling resolution. — [CoinAPI tutorial](https://www.coinapi.io/tutorials/analyzing-lead-lag-effects-between-exchanges-flat-files)
- Brauneis et al. find BTC/USD prices are not fully integrated across Bitfinex, Bitstamp and Coinbase Pro, with persistent price differences. Exchange liquidity depends mostly on local factors. — [Mannheim MADOC](https://madoc.bib.uni-mannheim.de/53986)
- Iran: Nobitex is the dominant Iranian venue (about $5B observed volume 2025 to Mar 2026, per TRM). Much of its cross-exchange flow went via Binance (Chainalysis, as reported). After a reopening Nobitex reported "prices outside the market's normal trend". I found no academic study of Iranian exchange premia or lags. — [TRM Labs Nobitex profile](https://www.trmlabs.com/resources/intel-library/nobitex); [CriptoNoticias / Reuters](https://www.criptonoticias.com/mercados/reuters-exchange-bitcoin-iran-gobierno-nobitex/)

### Inferences
- Lag on Western venues is set by liquidity and the presence of fast arbitrageurs, and is about 1-15 s. A 1-2 min lag on Wallex fits a venue with few or no fast arbitrageurs, which is plausible under sanctions and with no Binance access for locals. That is about 10x the lags in the literature, long enough to exploit with 1-2 s REST calls.
- Two separate effects should be kept apart. (a) A level premium or discount versus Binance, kimchi-style, driven by USDT/TMN capital controls. It is persistent and not tradeable on our own. Remove it by tracking a rolling basis, for example the median of (Wallex mid / Binance mid - 1) over 30-60 min. (b) A transient lag, which is the tradeable part: Wallex price minus (Binance price × (1 + basis)).
- Lead-lag is stronger in high-volume, fast-move periods (Schei; Makarov-Schoar). The signal will be bursty: most minutes have no edge, and edge clusters around Binance moves.

### Gaps
- No published lag estimates for Iranian venues (Wallex, Nobitex) against Binance. Our 1-2 min measurement is the only evidence.
- No recent (2023-2026) HY estimates for Binance against small regional spot venues. Numbers from 2018 data may overstate today's lags between Western venues.
- No Kaiko or Upbit lag statistics found.

## 2. How open-source bots implement it (Hummingbot XEMM, external price source, Avellaneda, custom bots)

### Takeaway
Hummingbot is the reference design. XEMM prices maker quotes on the slow venue from the fast venue's depth-weighted price (VWAP for the order size), with a profitability margin. Orders are cancelled when margin falls below a minimum or when they are older than about 120 s. PMM with `price_source: external_market` is the one-sided version: it quotes on one venue around another venue's mid price. Custom lead-lag bots use Binance 1-minute data, thresholds of about 0.5-1% over 5 min, and time-based exits at 15-30 min.

### Cited Findings
- Hummingbot XEMM (V1): it places limit orders on a less liquid maker exchange, with spreads wider than the taker side by `min_profitability`. It hedges each fill immediately on the taker exchange, whatever the profitability at that moment. Each cycle runs a cancel flow (is the order still profitable, still funded, still at the optimal price?), a create flow and a hedge flow. — [Hummingbot XEMM docs](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)
- XEMM parameter defaults, all from the same docs page:
  - `adjust_order_enabled` True
  - `order_refresh_mode` active_order_refresh
  - `cancel_order_threshold` 5 (a DEX example uses -0.05)
  - `limit_order_min_expiration` 130 s
  - `top_depth_tolerance` 0
  - `anti_hysteresis_duration` 60 s (minimum time between order adjustments)
  - `order_size_taker_volume_factor` 25%
  - `order_size_taker_balance_factor` 99.5%
  - `order_size_portfolio_ratio_limit` 16.67%
  - `slippage_buffer` 5%

  — [Hummingbot XEMM docs](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)
- Hummingbot `scripts/simple_xemm.py` logic:
  - Reference prices: on each tick it gets the taker's depth price for `order_amount` on each side via `get_price_for_volume`.
  - Maker BUY price = taker_sell_price / (1 + target_profitability).
  - Maker SELL price = taker_buy_price / (1 - target_profitability).
  - Cancel rule: cancel if current profitability < `min_profitability` or order age > `max_order_age`.
  - Defaults: target_profitability 0.001 (0.1%), min_profitability 0.0005 (0.05%), max_order_age 120 s, order_amount 0.1 ETH. Defaults pair KuCoin (maker) with Binance (taker).
  - On fill: hedge with a limit order on the taker side at the depth price, then cancel the rest of the maker order.

  — [simple_xemm.py](https://raw.githubusercontent.com/hummingbot/hummingbot/master/scripts/simple_xemm.py)
- Hummingbot Pure Market Making (PMM) parameters for external-price quoting:
  - Price source: `price_source` = current_market / external_market / custom_api, set with `price_source_exchange` and `price_source_market`. `price_type` = mid_price, last_price, best_bid, best_ask, etc.
  - Refresh: `order_refresh_time`, `order_refresh_tolerance_pct` (default 0; the price change needed before orders are refreshed).
  - Fills and inventory: `filled_order_delay` 60 s, `inventory_skew_enabled` (target base 50%), `ping_pong_enabled`, `hanging_orders_enabled` (cancel at 10%).
  - Bounds: `minimum_spread` -100 (cancel threshold), `price_ceiling`/`price_floor`, moving price band.
  - Custom API: `custom_api_update_interval` 5 s (minimum 0.5 s).

  — [Hummingbot PMM docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)
- External price source: an externally priced order that would cross the book is ignored unless `take_if_crossed` is enabled. This is exactly the "take stale quotes" switch. — [Hummingbot external price source docs](https://hummingbot.org/strategy-configs/external-price-source)
- Hummingbot also has an Avellaneda-Stoikov strategy, with reservation price and optimal-spread maths. I could not verify from these sources whether it supports an external price source. — [Hummingbot A-S deep dive](https://hummingbot.org/blog/technical-deep-dive-into-the-avellaneda--stoikov-strategy/)
- mamipour/lead-lag-trader uses Binance 1-minute data to trade altcoins on BTC moves. Its rules:
  - Bear regime (BTC 7-day return < -5%): short altcoins when BTC drops > 1% in 5 min; hold 15 min. Skip if the drop exceeds 2%.
  - Bull regime (7-day return ≥ 0%): long when BTC dips > 0.5% in 5 min; hold 30 min. Skip if BTC's 3-day return < 0.
  - Skip UTC hours 7-11.
  - Exits are time-based only; no take-profit or stop-loss.
  - Backtest (Apr 2022 to Mar 2026, assuming 0.02% maker fee + 1 bp slippage): +378%, 55% win rate, 25,230 trades, 47% max drawdown. The bot is idle about 98% of the time.
  - The author warns the edge is compressing in 2025-2026 and that faster bots can front-run.

  — [lead-lag-trader README](https://raw.githubusercontent.com/mamipour/lead-lag-trader/main/README.md)

### Inferences
- The XEMM pricing formula carries over to our one-sided case: fair_bid = Binance_bid_vwap × (1 + basis), quote_bid = fair_bid / (1 + m). With no hedge, m must cover the round-trip fee plus expected adverse drift, so it has to be at least about 0.2-0.3% rather than XEMM's 0.1%.
- The lead-lag-trader thresholds (0.5-1% over 5 min) were tuned for 0.02% fees. With our 0.10-0.13% fees, only the larger moves are likely to be worth trading.

### Gaps
- I did not inspect the Avellaneda config map, or the Hummingbot V2 XEMM controller source (`controllers/generic/xemm_multiple_levels.py`, if present).
- No public fill-rate statistics for XEMM bots.

## 3. One-sided strategies (trade only on the slow venue, no hedge)

### Takeaway
Without a hedge, lead-lag trading is directional, short-horizon stat-arb. The best documented one-sided result (Schei) works as follows. Enter on the lagging venue when its gap to the leader exceeds a threshold. Exit when a profit target equal to the threshold is reached, or as soon as the leader reverses. With threshold and target at 0.6% and fees of 0.1%, it beat buy-and-hold with about 75% winning trades. It roughly broke even at 0.14% fees and lost heavily at 0.25%. That fee range brackets our 0.10-0.13%.

### Cited Findings
- Schei's "next tick" test traded the lagging venue in the direction of the leader's last move, with mid-quote execution and no fees. It predicted Kraken's next mid-quote move from Binance with 69.15% accuracy, and Hitbtc's from Coinbase with 60.95%. Most pairs scored 60-70%. Its paper profit ($2,380 over 20 days for Binance to Kraken) turned into "extreme losses" once fees were included. — [CBS thesis pp.66-67](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- Schei's algorithm-based strategy (Jan-Feb 2019, 1 BTC per trade, long and short):
  - Rules: look at the leader's move over a window matched to the measured lag (for example 10 s for an 8 s lag). Open on the lagging venue when the gap is at least 0.6%. Exit at a 0.6% target, raise the target if the gap keeps widening, and close immediately if the leader reverses.
  - Binance to Kraken (strong lead-lag): 141 trades, 77.3% profitable, average win $20.4. It beat buy-and-hold at 0.1% fees, was about flat at 0.14%, and lost about $1,000 at 0.26%.
  - Binance to Coinbase (weak lead-lag): 142 trades, 73.2% profitable, average win $16.4. It did not profit even at 0.1% fees.

  — [CBS thesis pp.67-70](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- The author warns about frequency: too many trades let fees exceed profits, so a minimum threshold is needed, but too high a threshold produces no trades. Results assume mid-quote execution, which is unrealistic, and cover a quiet, cherry-picked period. — [CBS thesis pp.68-70](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- The Hummingbot PMM external price source, with `take_if_crossed`, is the standard one-venue way to quote around another venue's price and optionally take crossing quotes. — [Hummingbot external price source](https://hummingbot.org/strategy-configs/external-price-source); [PMM docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)

### Inferences (concrete designs implementable in hours, with REST at 1-2 s)
These are our own inferences, not sourced results.

Cost arithmetic: maker in and maker out costs 0.20% round trip. Taker in and maker out costs 0.23%. Taker in and taker out costs 0.26%. Paying the spread adds to this. The take threshold therefore needs to be at least about 0.3%, and the quote margin at least about 0.12-0.15% per side.

- **A. Stale-quote taker (closest to Schei).**
  - Fair price: Binance mid × (1 + rolling basis).
  - Entry: if Wallex best ask < fair × (1 - h), place a limit buy at the ask (marketable). Use h ≈ 0.30-0.40%, starting about 0.35%. Mirror this for sells, but only if we hold the coin; spot only means no shorts, and the rule is to hold USDT when idle.
  - Exit: a maker sell at about fair, or entry × (1 + 0.25-0.35%). Time stop at 2-3× the measured lag (about 3-5 min). Exit early if Binance reverses by more than about 0.15%.
  - Because the lag is 1-2 min, Binance can be polled every 2-5 s, so the 1-2 s REST latency is not binding.
- **B. Binance-anchored maker quoting (one-sided PMM / XEMM without the hedge).**
  - Quotes: bid = fair × (1 - s), ask = fair × (1 + s), with s ≈ 0.15-0.25%. Skew toward USDT when flat.
  - Refresh: cancel and replace when Binance moves more than about 0.05-0.1% (`order_refresh_tolerance_pct`), or the order is older than about 60-120 s.
  - Price limits: never let the bid sit above the Wallex best ask, and never let the ask sit below the Wallex best bid, unless deliberately taking.
  - Why it can work: the quotes anticipate where Wallex will be in 1-2 min. Wallex traders who hit them are trading on staler information than ours, so adverse selection runs in our favour. The exception is any other Binance-aware bot on Wallex.
  - Activity: this generates many small fills, which helps the trade-count score.
- **C. Binance momentum over 1-5 min.** Buy on Wallex if Binance has risen more than about 0.4-0.6% over the last 1-3 min and Wallex has not yet moved. Hold to the lag horizon plus a buffer (about 2-5 min), then exit with a maker order. This is A triggered by the leader's move instead of the price gap. Scale the threshold per asset by volatility; PEPE and SOL need larger thresholds than BTC or XAUT.
- Combine A and C: require both a Binance move and a Wallex gap. This filters noise, at the cost of fewer trades.
- Size: start small per pair, then scale.

### Gaps
- No published fill rates for passive quotes on thin regional books. These must be measured live.
- No source quantifies how fast a lag closes once a bot starts trading it. Our own trades on a thin book may help close the gap.

## 4. Hurdles and failure modes (fees, thin books, adverse selection, stale data, latency, rate limits)

### Takeaway
Fees are the main killer in every study. Schei's strategy was profitable at about 0.1% fees, flat at 0.14% and heavily loss-making at 0.25%. Crépellière et al. find cross-exchange arbitrage barely exploitable since 2018. Thin books add slippage. Other failure modes: shifts in the level basis mistaken for lag, faster competitors, and stale Binance or Wallex snapshots.

### Cited Findings
- Fees, slippage and lack of liquidity are the main limits to arbitrage. Thresholds offset fees only partly. Trading with market orders, the price can move in the fraction of a second it takes an order to arrive. Thin venues often lacked 1 BTC of volume per minute, so trade size has to be small. — [CBS thesis pp.70-71](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- In 2019, average spreads on the top-10 spot venues were 0.01-0.10%, held up by fees and tick sizes. That makes arbitrage between major venues nearly impossible; only deviations from these levels are tradeable. — [CBS thesis p.70, citing Bitwise 2019](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf)
- Arbitrage profits fell sharply after April 2018, a fall linked to informed trading. — [Crépellière et al.](https://ris.uni-paderborn.de/record/34449)
- Fee level is positively tied to the size of the kimchi-premium band: a higher fee means a wider no-arbitrage band. — [Monash PDF](https://researchmgt.monash.edu/ws/files/590976616/586624035_oa.pdf)
- Competition and decay: other traders with faster infrastructure running the same logic would front-run you, and the lag appears to be shrinking. Live results will be worse than backtests (slippage, fill uncertainty, downtime). — [lead-lag-trader README](https://raw.githubusercontent.com/mamipour/lead-lag-trader/main/README.md)
- In XEMM, an immediate hedge can fail or slip, which leaves the position unhedged. Maker quotes go stale as prices move, so cancel logic, `anti_hysteresis_duration` (60 s) and order expiry are central. — [Hummingbot XEMM docs](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/)
- Even between Western venues, prices are not fully integrated: price differences persist. A persistent basis is normal and is not by itself a signal. — [Brauneis et al.](https://madoc.bib.uni-mannheim.de/53986)

### Inferences (failure modes specific to Wallex)
These are our own inferences, not sourced results.
- **The earlier "gap smaller than spread" result.** On Toman pairs the TMN fee is 0.25-0.30% and the USDT/TMN conversion adds noise, which likely explains the failure. On USDT pairs (0.10-0.13% fees, direct quote) the hurdle is roughly halved. The test is still worth redoing on USDT pairs, using the gap measured against the Wallex best ask or bid, not the mid.
- **Spread.** If the Wallex bid-ask spread is often larger than about 0.3%, taking (design A) is uneconomic. Quote inside the spread instead (design B), and use the lag only to choose the side and price.
- **Stale data.** Treat Binance data older than about 5 s, or a Wallex book older than about 3 s, as stale, and skip that cycle. Wallex's 1-2 min lag may partly be quote staleness, with no trades happening. A displayed stale ask may not be fillable at size.
- **Basis drift.** A move in USDT/TMN, or a Wallex-local premium, can look like a lag that never closes. Use a rolling-median basis and a time stop.
- **Rate limits.** Our 1-2 s REST calls cover polling, order placement and cancels. Polling 7 pairs plus managing orders can approach exchange limits. Prioritize the 2-3 most liquid or most-lagging pairs, and batch book reads if Wallex supports it. I found no Wallex rate-limit documentation.
- **Inventory.** Under the hold-USDT-when-idle rule, sell-side signals work only on inventory we already hold. That halves the opportunity set, unless we keep a small base inventory per pair.

### Gaps
- Wallex rate limits, order-book depth and typical spreads per USDT pair were not found in public sources and must be measured.
- No source quantifies adverse selection for passive quotes on a venue where the quoter is the faster party. The argument that it runs in our favour is theoretical.

## GitHub repos

| Repo | Stars / last push (as visible) | What it does | Key files / modules worth copying |
|---|---|---|---|
| [hummingbot/hummingbot](https://github.com/hummingbot/hummingbot) | 20,630 stars; pushed 2026-10-20 ([GitHub API](https://api.github.com/repos/hummingbot/hummingbot)) | Market-making and arbitrage framework (XEMM, PMM with external price, Avellaneda-Stoikov, AMM arb) | `scripts/simple_xemm.py`, the most copyable file (fair price from taker VWAP via `get_price_for_volume`; maker prices `taker_sell/(1+target)` and `taker_buy/(1-target)`; cancel when profitability < min or age > 120 s) ([source](https://raw.githubusercontent.com/hummingbot/hummingbot/master/scripts/simple_xemm.py)). The V1 XEMM strategy and PMM `price_source: external_market` config (`order_refresh_tolerance_pct`, `take_if_crossed`, inventory skew, `filled_order_delay`) ([XEMM docs](https://hummingbot.org/strategies/v1-strategies/cross-exchange-market-making/), [PMM docs](https://hummingbot.org/strategies/v1-strategies/pure-market-making/)) |
| [mamipour/lead-lag-trader](https://github.com/mamipour/lead-lag-trader) | 0 stars; pushed 2026-03-18 ([GitHub API](https://api.github.com/repos/mamipour/lead-lag-trader)) | Lead-lag from BTC moves to altcoins, using Binance 1-minute data, regime filter, time exits | `src/paper/config.py` (parameters as dataclasses: 1% / 5 min, 15 min hold, 0.5% dip, 30 min hold); `src/paper/price_tracker.py` (BTC monitor and regime stats); `src/paper/position_manager.py`; `scripts/collect_data.py` (Binance 1m fetcher) ([README](https://raw.githubusercontent.com/mamipour/lead-lag-trader/main/README.md)) |
| [Drakkar-Software/OctoBot-Market-Making](https://github.com/Drakkar-Software/OctoBot-Market-Making) | 42 stars; pushed 2026-03-30 ([GitHub API](https://api.github.com/repos/Drakkar-Software/OctoBot-Market-Making)) | Python market-making bot built on OctoBot, covering Binance, MEXC, Bitmart and 15+ more exchanges; keeps a ladder of buy and sell orders updated | I did not inspect the order-refresh module; I could not confirm whether it supports an external reference price ([ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fdrakkar-software%2Foctobot-market-making)) |
| [andrei-zgirvaci/Arbitrage-Bot](https://github.com/andrei-zgirvaci/Arbitrage-Bot) | not checked | ccxt-based search for the best cross-exchange spread (includes Binance) | ccxt exchange plumbing and spread scan; no lead-lag model |
| [bmino/binance-triangle-arbitrage](https://github.com/bmino/binance-triangle-arbitrage) | not checked | Triangular arbitrage on a single exchange (Binance); the author says network delay, not software speed, is the bottleneck ([ecosyste.ms](https://awesome.ecosyste.ms/projects/github.com%2Fbmino%2Fbinance-triangle-arbitrage)) | Not directly relevant; a reference for latency handling only |
| Schei thesis code (appendix 10.3, not a repo) | n/a | Python for the HY cross-correlation and the threshold strategy backtest | HY estimator and lead-lag ratio code, in the appendix of the [CBS thesis PDF](https://research-api.cbs.dk/ws/portalfiles/portal/59795067/612646_High_Frequency_Lead_Lag_Relationships_in_The_Bitcoin_Market.pdf) (p.121 onward; not read) |

Gaps: I found no public repo that specifically trades a slow regional venue against a Binance reference price without a hedge. The closest are Hummingbot PMM with an external price source and `simple_xemm.py` with the hedge removed. Stars and dates were not checked for the last two repos.
