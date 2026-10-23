# ML and rule-based crypto trading systems: open-source, competitions, short-horizon research

Context: Wallex hackathon, spot only, about 1 day live left, USDT pairs (BTC, ETH, XRP, SOL, PEPE, BNB, XAUT), maker 0.10% / taker 0.13%, limit orders only (about 0.20-0.26% round trip). The current system is an hourly LightGBM+logreg ensemble with triple-barrier labels and Binance outside features. Its edge is weak: +3.8% on validation and -0.9% on test.

Research budget note: Kaggle discussion pages render empty through the fetch tool. FreqAI docs at freqtrade.io returned 403, so I read the GitHub raw markdown instead. Several numbers below come from abstracts or search snippets only. Those are flagged.

## 1. What did top Kaggle G-Research Crypto Forecasting solutions use, and what transfers?

### Takeaway
All top-3 G-Research solutions used LightGBM, and the winners said feature engineering mattered far more than the model. The task was 15-minute residualised returns, scored by weighted Pearson correlation and judged on 3 months of live data. That is a correlation (ranking) task with no costs, so it does not prove you can beat a 0.26% round trip on 1-hour bars.

### Cited Findings
- The competition ran Nov 2021 to May 2022. Competitors forecast short-term returns of 14 coins, scored on a weighted Pearson correlation. Each team's two best models were run for 3 months on live data for the final leaderboard. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- "All of our top three used a LightGBM model." Some competitors tried neural nets. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- The winners believed feature engineering contributed most. "Developing and testing different features had a far greater impact on their final scores than model development." — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- The leaderboard was very noisy, with many big jumps and precipitous drops, especially early. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- Winners: 1st Meme Lord Capital (Jose and Eduardo, $50k), 2nd Nathaniel Maddux, 3rd GABA. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)
- The 2nd-place write-up lists its toolkit (CPython, Numba, Pandas, LightGBM, scikit-learn) and explicitly withholds its features ("I won't be sharing my code or any specific insight on features"). — [Kaggle discussion 323098, via search snippet](https://www.kaggle.com/c/g-research-crypto-forecasting/discussion/323098)
- The target was a 15-minute residualised return (coin return minus a beta-adjusted market return). A community reimplementation rebuilds the residualised target, minute-grid alignment across 14 coins, and a cross-asset GRU with attention. — [Wy66wy/g-research-crypto-forecasting](https://github.com/Wy66wy/g-research-crypto-forecasting)
- Related cross-asset evidence: on Binance minute data, lagged returns of other coins predict a focal coin for up to 10 minutes. A long-short portfolio reportedly earned 2.16% per day out-of-sample after costs (working-paper abstract, not verified). — [SMU working paper](https://ink.library.smu.edu.sg/lkcsb_research/6901)

### Inferences
- Transferable: (a) residualise the target, predicting coin return minus beta times the BTC or market return, so the model learns relative moves; (b) cross-asset lagged returns as features; (c) put effort into features, not model tuning; (d) judge models by correlation or information coefficient (IC) as well as PnL, because PnL over about 1 month is too noisy to rank models.
- Not transferable: the competition had no fees, no direction constraint and no execution. Our test of +3.8% validation against -0.9% test sits within the noise G-Research described.
- A residual (relative-value) signal is hard to use spot-only and long-only. The usable version is to pick which coin to hold when the model says hold something, and hold USDT otherwise.

### Gaps
- The 1st- and 3rd-place feature lists could not be retrieved: Kaggle discussion pages rendered empty, and the 2nd-place author withheld details. Do not claim specific winner features.

## 2. FreqAI and popular Freqtrade strategies: features, labels, live vs backtest, what survives fees

### Takeaway
FreqAI's default recipe is a forward rolling mean of close relative to current close as the target. Features are indicators expanded across periods, timeframes, correlated pairs and shifted candles, and the model is retrained on a sliding window with recency weighting and outlier gating. The most popular community strategy (NostalgiaForInfinity) is a 5-minute, many-pair, many-open-trades rule system. I found no rigorous live-vs-backtest evidence for either.

### Cited Findings
- FreqAI features use the `%` prefix and targets the `&` prefix. Features come from `feature_engineering_expand_all`, which is multiplied across `indicator_periods_candles`, plus `expand_basic` and `standard` (for example day of week). The example target `&-s_close` is a forward rolling mean of close relative to current close over `label_period_candles`. — [FreqAI feature engineering docs (GitHub)](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/freqai-feature-engineering.md)
- Feature expansion: `include_timeframes`, `include_corr_pairlist` (for example BTC and ETH features on every pair), `include_shifted_candles`, `indicator_periods_candles`. The example config gives 3x3x3x2x2 = 108 features. — [same](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/freqai-feature-engineering.md)
- Robustness tools: `weight_factor` (exponential recency weighting of training rows); the Dissimilarity Index (`DI_threshold`), which discards predictions on inputs unlike the training data; SVM and DBSCAN outlier removal; PCA; noise injection. The default pipeline is MinMaxScaler(-1,1) plus VarianceThreshold. — [same](https://raw.githubusercontent.com/freqtrade/freqtrade/develop/docs/freqai-feature-engineering.md)
- NostalgiaForInfinity requires a 5-minute timeframe. It recommends 6-12 open trades, 40-80 pairs from a volume pairlist, stablecoin quotes over BTC or ETH quotes, and leveraged tokens blacklisted. Backtest results appear only in commit comments, and the repo makes no live-performance claims. — [NostalgiaForInfinity README](https://github.com/iterativv/NostalgiaForInfinity)
- General backtest-overfitting evidence: optimised in-sample strategies often do very poorly out of sample. — [Bailey et al., SSRN](https://papers.ssrn.com/abstract=2507040); [How should you discount your backtest PnL? (arXiv)](https://arxiv.org/pdf/1902.01802)

### Inferences
- Cheap FreqAI ideas to test in our pipeline within hours:
  - BTC and ETH features on every pair, like `include_corr_pairlist` (we already have lead returns).
  - Recency weighting, like `weight_factor`.
  - A DI-style novelty gate: skip trades when the current feature vector is far from the training distribution. This works like a regime filter and cuts trades in unfamiliar conditions.
- Strategy styles that plausibly survive fees, from reasoning about cost structure and not from cited live data: few, larger moves; maker entries; holding times of several hours. Exits at small fixed ROI targets (NFI style) rely on high win rates and many pairs, which conflicts with our 7-pair universe.

### Gaps
- I found no systematic data on FreqAI or NFI live-vs-backtest performance. The search returned only generic overfitting literature. Discord and GitHub-issue anecdotes were not retrieved.

## 3. Short-horizon predictability: momentum/reversal, time of day, order flow, BTC lead, funding. Effect sizes vs about 0.26% round trip

### Takeaway
The evidence for predictability is real but mostly small and short-lived. BTC-to-altcoin lead-lag shows up at minute horizons (up to about 10 minutes) and decays fast, so hourly bars mostly miss it. Order-imbalance signals reported in 2026 work over 4-12 hour horizons on Binance perps. The best-documented time-of-day effect is positive BTC returns around 21:00-23:00 UTC. That evidence comes from a single source and one exchange.

### Cited Findings
- Bitcoin intraday time-series momentum: the first half-hour's return (using a volume-based clock) predicts the last half-hour's return. The authors attribute it to liquidity provision, and gains are larger in downturns. — [Shen, Urquhart, Wang, Financial Review 2022](https://research.birmingham.ac.uk/en/publications/bitcoin-intraday-time-series-momentum/); [working paper PDF](https://reading-9.eprints-hosting.org/100181/3/21Sep2021Bitcoin%20Intraday%20Time-Series%20Momentum.R2.pdf)
- Intraday return, volume and volatility patterns in a crypto index (CRIX) look like algorithmic-trading footprints. — [Rise of the Machines? (arXiv 2009.04200)](https://arxiv.org/pdf/2009.04200)
- Time of day: on hourly Gemini data, returns at 22:00 and 23:00 UTC are the most economically significant and several positive hours are significant at 5%. No hour is significantly negative, and 03:00-04:00 is worst. A follow-up through early 2022 found buying at 21:00 and selling at 23:00 UTC gave a backtested annualised return of 40.64% and a Calmar ratio of 1.79. It was best on Fridays and in uptrends. — [Quantpedia: intraday or overnight anomalies in Bitcoin](https://quantpedia.com/are-there-seasonal-intraday-or-overnight-anomalies-in-bitcoin/); [Quantpedia: Seasonality of Bitcoin](https://quantpedia.com/the-seasonality-of-bitcoin/)
- BTC leads altcoins: small-cap coins respond to BTC moves with a delay, lower liquidity means slower reactions, and Granger causality runs from BTC to altcoins. A BTC-lag ML trading rule beat buy-and-hold (Kurihara and Matsumoto, Asia-Pacific Financial Markets 2026, via search snippet). — [search result, t2r2 Titech](https://t2r2.star.titech.ac.jp/rrws/file/CTT100941776/ATD100000413/)
- Minute-level cross-coin predictability lasts up to 10 minutes (Binance). — [SMU working paper](https://ink.library.smu.edu.sg/lkcsb_research/6901)
- Leading venues at 500-1000 ms horizons: the Binance USDT-margined perpetual and Huobi futures lead other Bitcoin markets. This supports using Binance perps as the "outside" price for a slower local exchange. — [Fragmentation, Price Formation, and Cross-Impact in Bitcoin Markets (arXiv 2108.09750)](https://arxiv.org/pdf/2108.09750)
- A practitioner reports persistent hourly lead-lag between some coins and BTC, more useful for execution than for slow strategies. — [Robot Wealth](https://robotwealth.com/topic-tag/data-analysis)
- Quarter-hour effect: on Binance perpetual trade data, opening returns at quarter-hour marks are predictable out of sample, and opening order imbalance predicts returns over about 4-12 hours, with weaker effects at finer frequencies. The abstract gives no effect sizes or cost analysis. — [arXiv 2607.09426](https://arxiv.org/abs/2607.09426)
- Order-flow imbalance against BTC 1-second price changes has weak explanatory power (R² about 0.094). Cont et al.-style OFI is mainly contemporaneous, not predictive. — [arXiv 2308.14235](https://arxiv.org/pdf/2308.14235); [TDS commentary](https://towardsdatascience.com/price-impact-of-order-book-imbalance-in-cryptocurrency-markets-bf39695246f6)

### Inferences
- Effect size vs costs: the 21-23 UTC effect works out to roughly 0.1-0.2% per day by my arithmetic (40% a year over about 365 two-hour windows, about 0.1% per trade). That is below our 0.20-0.26% round trip, so it is not tradable as a standalone strategy. It may work as a tilt: prefer entries before or into 21-23 UTC and avoid opening at 03-04 UTC. Easy test: add hour-of-day as a feature or gate in the existing model.
- On a slower local exchange, Wallex's lag relative to Binance is likely a bigger effect than BTC-to-alt lead. Our premium-vs-Binance feature is the right idea. Its value peaks at minute scale, so it may work better as an execution trigger: place the limit buy when Binance has moved up and Wallex has not repriced yet.
- The 4-12 hour order-imbalance horizon matches our 12-bar hourly label. Binance taker-buy-share imbalance summed over several hours (not 1 bar) is a candidate feature that fits the horizon.

### Gaps
- No peer-reviewed effect sizes for funding or basis signals at intraday horizons came up in this budget. I found no direct evidence on predictability specific to PEPE or XAUT.
- The time-of-day evidence comes from one research outlet (Quantpedia) on Gemini data through 2022, with no independent replication found.

## 4. Regime detection, meta-labeling, ensembling, vol targeting; does RL beat supervised models?

### Takeaway
Meta-labeling (a primary rule picks direction, then a calibrated classifier gates and sizes) is the most directly applicable idea. Its own proponents say it "amplifies an existing edge; it cannot create one." The best-known honest crypto RL study (FinRL lineage, with overfitting rejection) still lost 35% in its test window, beating only falling benchmarks. I found no evidence that RL beats supervised models.

### Cited Findings
- Jesse meta-labeling workflow:
  - The primary signal has high recall, for example a Supertrend flip with ADX above 25.
  - Triple-barrier labels are placed only in the signal's direction, with a 24-bar vertical barrier.
  - Features cover trend, momentum, ATR%, volume, cyclical hour and day of week, and signal side.
  - Model selection uses GridSearchCV with TimeSeriesSplit and ROC AUC, and probabilities are calibrated with CalibratedClassifierCV.
  - Live, trade only if P is at least 0.60, and scale risk from 1% to 2% with P.
  - Warnings: miscalibration makes confidence sizing harmful (use flat size and treat P as a gate only); the meta-model is valid only for the exact primary signal it was trained on; claimed precision gains of +5-20% are typical, not guaranteed.
  - [Jesse docs: meta-labeling](https://docs.jesse.trade/docs/research/ml/meta-labeling)
- "Meta-labeling amplifies an existing edge; it cannot create one where none exists." — [Jesse docs](https://docs.jesse.trade/docs/research/ml/meta-labeling)
- Counter-evidence: a QuantConnect user found meta-labeling on top of a decent strategy did significantly worse than without it. — [QuantConnect forum](https://www.quantconnect.com/forum/discussion/14706/why-meta-labeling-is-not-a-silver-bullet/)
- Meta-labeling aims to improve precision without sacrificing recall, and the meta-model's probability can size positions. — [Wikipedia: Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling); [Hudson & Thames toy example](https://hudsonthames.org/meta-labeling-a-toy-example/)
- Gort et al. (FinRL lineage) say earlier DRL crypto studies reported inflated backtests. They treat overfitting detection as a hypothesis test and reject agents with a high probability of overfitting at the 10% level. The test covered 10 coins, 2022-05-01 to 2022-06-27. — [arXiv 2209.05559](https://arxiv.org/abs/2209.05559)
- The best PPO agent returned -34.96%, against -50.78% for the S&P Crypto Broad Digital Market index and -47.78% for equal weight. It won by losing less in a crash. — [FinRL Contests benchmark paper (arXiv 2504.02281)](https://arxiv.org/pdf/2504.02281)
- Validation hygiene: overlapping triple-barrier labels leak, so use purged and embargoed cross-validation. — [Hudson & Thames](https://hudsonthames.org/meta-labeling-a-toy-example/); [Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling)

### Inferences
- Our model is already a meta-labeler in spirit, since the triple barrier is long-only and the output is calibrated. What it lacks is a high-recall primary rule. Training on every bar gives a noisy base rate. Concrete test: condition training and trading on a simple primary event, then let the model gate those events. Candidate events: price above the 24h EMA and a 3-6h pullback; a BTC lead move above k·ATR while the alt has not moved; a 21-23 UTC window entry.
- Regime gate: the cheapest version is "trade only when BTC is above its 24-72h EMA, or when the DI novelty is low," following FreqAI's DI idea. RL is not worth trying with 1 day left.
- Sizing: per Jesse's warning, use calibrated P for sizing only if the reliability curve on the test month is good. Otherwise use a flat size and a P gate.

### Gaps
- I found no rigorous crypto-specific out-of-sample numbers for meta-labeling, HMM regime models or volatility targeting at intraday horizons.

## 5. Short live-window trading competitions and hackathons: what won, and was it luck, leverage or concentration?

### Takeaway
I found no published post-mortems that analyse what won short crypto competitions. Public competitions rank on volume, profit or a snapshot, and those rules reward variance and activity, not skill.

### Cited Findings
- Bybit's spot campaign ranks by cumulative spot trading volume. TradingView's Leap ranked by virtual profit. — [TradingView Leap crypto results](https://www.tradingview.com/blog/en/the-leap-crypto-edition-results-57323/); [Finance Magnates on Leveraged Cup](https://www.financemagnates.com/thought-leadership/leveraged-cup-awards-20000-grand-prize-in-global-trading-competition/)
- BitMEX picks winners from a leaderboard snapshot near the end of the competition. — [BitMEX CNY 2024 competition](https://www.bitmex.com/cny-2024-competition)
- The Leveraged Cup removed account size from scoring and judged returns, risk management and performance under pressure. The winner also took four Sprint payouts. — [Finance Magnates](https://www.financemagnates.com/thought-leadership/leveraged-cup-awards-20000-grand-prize-in-global-trading-competition/)
- In the ACM ICAIF 2025 crypto forecasting competition (a forecasting task), Team 8k3 (Edinburgh) won with a Stuart-Landau recurrent neural network. — [search result referencing CryptoPulse/ICAIF, arXiv 2502.19349](https://arxiv.org/pdf/2502.19349)
- The G-Research leaderboard over a 3-month live window still showed big jumps and drops. — [G-Research wrap-up](https://www.gresearch.com/news/wrapping-up-the-g-research-crypto-forecasting-competition/)

### Inferences
- Over a 1-day window, the expected value of any edge we have (a few basis points per trade) is tiny next to the variance of BTC or PEPE moves. The ranking will mostly reflect exposure size and direction luck.
- Under our scoring (dollar return, trade count, stability, reproducibility), the defensible moves are:
  - keep a modest, reproducible rule set;
  - get trade count from small maker-only trades in liquid pairs, not from loosening filters, which lost money;
  - hold idle capital in USDT, which keeps the stability score;
  - avoid concentrating on one high-volatility pair, which is luck-driven.

### Gaps
- No write-ups found for 1-7 day crypto trading hackathon winners, or for Wallex or other Iranian exchange competitions.

## GitHub repos

Stars and last-push dates come from the GitHub API on 2026-10-22, except Wy66wy, where the API timed out.

| Repo | Stars | Last push | What it is | Files or modules worth copying |
|---|---|---|---|---|
| [freqtrade/freqtrade](https://github.com/freqtrade/freqtrade) | 55,752 | 2026-10-22 | Bot framework with backtesting, hyperopt and FreqAI | `freqtrade/templates/FreqaiExampleStrategy.py` (feature_engineering_expand_all/basic/standard, set_freqai_targets `&-s_close`); `freqtrade/freqai/` (data_kitchen.py: DI/SVM/DBSCAN outlier and novelty gate, `weight_factor` recency weights, sliding-window retrain); `docs/freqai-feature-engineering.md` |
| [iterativv/NostalgiaForInfinity](https://github.com/iterativv/NostalgiaForInfinity) | 3,491 | 2026-10-22 | Most popular community Freqtrade strategy: 5m, 40-80 pairs, 6-12 open trades | `NostalgiaForInfinityX*.py`: dip-buy entry condition blocks and profit-dependent exit ladders. Useful as a catalogue of rule-based entry filters (BTC-state checks, pump/dump protections) |
| [freqtrade/freqtrade-strategies](https://github.com/freqtrade/freqtrade-strategies) | 5,582 | 2026-09-08 | Official example strategy collection | `user_data/strategies/` classic rule sets (for example BB/RSI reversion), small enough to port as primary signals for meta-labeling |
| [jesse-ai/jesse](https://github.com/jesse-ai/jesse) | 8,685 | 2026-10-22 | Python crypto bot framework with research mode | Meta-labeling workflow (gather/deploy modes, triple-barrier directional labels, TimeSeriesSplit plus CalibratedClassifierCV, P to size mapping) per the [docs](https://docs.jesse.trade/docs/research/ml/meta-labeling) |
| [AI4Finance-Foundation/FinRL](https://github.com/AI4Finance-Foundation/FinRL) | 16,784 | 2026-10-20 | Deep RL trading library | Environment and agent wrappers. Low priority: the honest crypto results are negative (see section 4) |
| [AI4Finance-Foundation/FinRL_Crypto](https://github.com/AI4Finance-Foundation/FinRL_Crypto) | 205 | 2025-12-15 | Code for Gort et al., DRL crypto with overfitting rejection | Probability-of-backtest-overfitting rejection of candidate models with combinatorial purged CV. The concept is reusable to choose among our LGBM configs |
| [hudson-and-thames/mlfinlab](https://github.com/hudson-and-thames/mlfinlab) | 4,949 | 2023-10-02 (stale) | Lopez de Prado AFML tooling | Triple-barrier labelling, meta-labels, purged and embargoed k-fold, sample uniqueness weights, bet sizing from probabilities. Later versions are commercial, so check the licence and prefer reimplementation |
| [stefan-jansen/machine-learning-for-trading](https://github.com/stefan-jansen/machine-learning-for-trading) | 21,514 | 2026-10-22 | Book code (3rd ed.) covering end-to-end ML for trading | Alpha-factor notebooks, LightGBM IC evaluation, MultipleTimeSeriesCV (purged walk-forward) |
| [nkaz001/hftbacktest](https://github.com/nkaz001/hftbacktest) | 4,952 | 2025-12-23 | Tick and L2 backtester with queue position and latency | Realistic fill modelling for limit-only execution; Binance and Bybit examples. Relevant for testing maker-fill assumptions |
| [Wy66wy/g-research-crypto-forecasting](https://github.com/Wy66wy/g-research-crypto-forecasting) | n/a (API timeout) | n/a | Reimplementation of the G-Research task, not a winning solution | Residualised-target construction (coin return minus beta times market), minute-grid alignment across 14 coins, cross-asset GRU |

Repo caveats:
- No official 1st-, 2nd- or 3rd-place G-Research solution code was found. The 2nd-place author declined to share code.
- Kaggle "3rd place solution" training notebooks exist on the competition's [code tab](https://www.kaggle.com/competitions/g-research-crypto-forecasting/code), but I could not verify their contents or provenance.

## Concrete changes testable in a few hours (ranked by evidence and effort)

These are synthesised from the above.

1. **Residualised or relative target plus pair selection.** Predict the alt's 12h return minus beta times the BTC return (G-Research style). Then hold only the top-ranked coin, and only when the absolute triple-barrier model also agrees. Otherwise stay in USDT.
2. **Primary-event conditioning (meta-labeling).** Train and trade only on bars where a simple high-recall rule fires, for example a BTC 1-3h lead move above 1 ATR while the alt lags, or a pullback in an uptrend. Pick the threshold where precision is clearly above the event's base win rate.
3. **Hour-of-day feature or gate.** Test hour-of-day as a feature, or as a gate that opens no new trades at 02-05 UTC and favours 20-23 UTC entries. Check that it helps both the validation and test months, not just one.
4. **Multi-hour taker-imbalance features.** Sum Binance taker-buy share over 4h, 8h and 12h, matching the 4-12h order-imbalance horizon from arXiv 2607.09426.
5. **Novelty gate (FreqAI DI).** Compute the mean distance of the current feature vector to its k nearest training rows. Skip trades above the 90th-95th percentile.
6. **Model selection by IC plus a probability-of-backtest-overfitting estimate (CSCV) across configs,** instead of picking the best validation PnL. Then freeze the configuration for reproducibility.
7. **Execution.** Post maker limits at or just inside the best bid only when Binance leads upward (premium feature negative, meaning Wallex is cheap). This turns the lead-lag effect into better fills rather than a forecast.
