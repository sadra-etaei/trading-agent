# LLM Trading Agent Architectures: Designs, Measured Results, Critiques, and Cheap Patterns

Research date: 2026-10-22. Context: Wallex spot hackathon, about 1 day of live trading left, hourly LightGBM+logreg signal, GPT veto/exit overlay, about $1.4 of LLM budget left.

## Q1. Architectures and reported results of published LLM trading agents, and which components ablations showed matter

### Takeaway
The architectures that publish ablations show that **risk control (CVaR / explicit risk rules), reflection on past outcomes, and structured market/transaction context** move results most. Multi-agent debate has no clean ablation evidence. The best live benchmark found that agent design matters more than the choice of LLM backbone. Headline returns come from short windows of 2 to 6 months and do not hold up well under independent re-tests (see Q3).

### Cited Findings
**TradingAgents (Tauric Research, arXiv 2412.20138, Dec 2024 with revisions into 2025)**
- Design: four analysts (fundamentals, sentiment, news, technical) write structured reports. Bull and bear researchers debate for n rounds and a facilitator records the prevailing view. A trader agent decides buy/sell/hold. Aggressive, neutral and conservative risk debaters argue for n rounds, then a fund manager approves the trade. Quick models (gpt-4o-mini, gpt-4o) handle retrieval and summaries; deep models (o1-preview) handle decisions — [arXiv HTML v4](https://arxiv.org/html/2412.20138v4)
- Results, Jan 1 to Mar 29 2024: cumulative return 26.62% on AAPL (Sharpe 8.21, max drawdown 0.91%), 24.36% on GOOGL, 23.21% on AMZN, against the best rule baselines at 2.05%, 6.23% and 11.01%. The authors flag the Sharpe ratios as implausibly high for a short window with few pullbacks — [arXiv HTML v4](https://arxiv.org/html/2412.20138v4)
- **No component ablation** (debate, risk team or reflection removed) is reported. Each prediction costs about 11 LLM calls and 20+ tool calls — [arXiv HTML v4](https://arxiv.org/html/2412.20138v4)
- Independent re-run over 2025-2026 with realistic costs: Sharpe 0.43 gross, 0.22 net, and it lost to buy-and-hold on 4 of 5 stocks — [The Alpha Illusion, arXiv 2605.16895 (May 2026)](https://www.alphaxiv.org/overview/2605.16895)

**FinCon (NeurIPS 2024, arXiv 2407.06567)**
- Design: a manager-analyst hierarchy. Within an episode, CVaR-based risk control. Across episodes, "conceptual verbal reinforcement" (CVRF) updates the manager's beliefs from P&L trends and sends them back to the relevant analysts — [NeurIPS slides](https://nips.cc/media/neurips-2024/Slides/94352_FOt8cLF.pdf); [arXiv](https://arxiv.org/html/2407.06567v3)
- Ablation, risk control: with CVaR versus without, GOOG 25.08% vs -1.46%, NIO 17.46% vs -52.89%, portfolio 113.8% vs 14.7% — [NeurIPS slides](https://nips.cc/media/neurips-2024/Slides/94352_FOt8cLF.pdf)
- Ablation, belief/reflection updates: with versus without, GOOG 25.08% vs -11.94%, NIO 17.46% vs 8.20%, portfolio 113.8% vs 28.4%. The "without" baselines differ between the two tables, an internal inconsistency — [NeurIPS slides](https://nips.cc/media/neurips-2024/Slides/94352_FOt8cLF.pdf)

**CryptoTrade (EMNLP 2024, arXiv 2407.09546; the only peer-reviewed crypto-spot LLM agent found)**
- Design: on-chain data, off-chain news and technical indicators feed GPT-3.5 analyst agents. A trader agent (GPT-4) makes daily decisions, and a reflection agent reviews the outcomes of prior decisions — [arXiv HTML v2](https://arxiv.org/html/2407.09546v2)
- Results on 2023 BTC/ETH/SOL across bull, sideways and bear windows: it **often lost to buy-and-hold** (BTC bull 26.4% vs 39.7%, SOL bull 99.8% vs 176.7%). It beat buy-and-hold in bear markets (SOL bear -19.6% vs -36.1%, BTC bear -11.7% vs -15.6%), so most of its edge was losing less in downtrends — [arXiv HTML v2](https://arxiv.org/html/2407.09546v2)
- Ablation, ETH bull: full prompt 28.47%, without reflection 17.14%, without news 19.69%, without transaction statistics 12.70%, without technical indicators 17.27%, base prompt only 8.40%. The full-prompt number does not match the main table — [arXiv HTML v2](https://arxiv.org/html/2407.09546v2)

**FinMem (layered memory and character design)**
- The repo implements memory scoring from recency decay, importance and access-counter terms, plus a reflection module (`puppy/memory_functions/*`, `puppy/reflection.py`, `puppy/prompts.py`) — [GitHub](https://github.com/pipiku915/FinMem-LLM-StockTrading)
- FinMem and FinAgent were each evaluated over about 6 months on about 5-6 symbols — [FINSABER, arXiv 2505.07078](https://arxiv.org/pdf/2505.07078)
- Performance of FinMem and QuantAgent reportedly dropped by more than 50% in some cases after the model's training cutoff — [The Alpha Illusion](https://www.alphaxiv.org/overview/2605.16895)

**Agent Market Arena (AMA, arXiv 2510.11695, Oct 2025): live benchmark, Aug 1 to Sep 30 2025, on TSLA, BMRN, BTC and ETH; one daily decision from BUY/SELL/HOLD**
- Four agents: InvestorAgent (single agent with memory), TradeAgent (the TradingAgents-style debate plus risk manager), HedgeFundAgent (persona agents, from the ai-hedge-fund design) and DeepFundAgent (memory-based). Each ran on GPT-4o, GPT-4.1, Gemini-2.0-flash, Claude-3.5-haiku and Claude-sonnet-4 — [arXiv PDF](https://arxiv.org/pdf/2510.11695)
- "Changing the LLM backbone… shifted outcomes by less than agent switching." Agent architecture was the dominant factor — [arXiv PDF](https://arxiv.org/pdf/2510.11695)
- Crypto was hard. Buy-and-hold BTC returned 0.66% and **most agent/backbone combinations lost money on BTC**. The best was DeepFundAgent with Claude-3.5-haiku at 1.67% (Sharpe 0.50); TradeAgent with GPT-4.1 returned -5.65%. On ETH (buy-and-hold 18.56%), HedgeFundAgent made 39.66% while TradeAgent lost on every backbone — [arXiv PDF, Table 1](https://arxiv.org/pdf/2510.11695)
- A **majority vote across backbones ("Vote")** was usually middle of the pack, not best — [arXiv PDF, Table 1](https://arxiv.org/pdf/2510.11695)
- The authors found the most successful agents were those "capable of exploiting periods of volatility rather than merely following long-term trends" — [arXiv PDF](https://arxiv.org/pdf/2510.11695)

**LiveTradeBench (arXiv 2511.03628, Nov 2025)**
- 50-day live run of 21 LLMs on US stocks and Polymarket. A high LMArena score did not predict trading performance, and models showed distinct risk appetites — [HF papers](https://huggingface.co/papers/2511.03628)

### Inferences
- The strongest ablation effects come from **explicit risk control** (FinCon CVaR) and **reflection / feedback from realized P&L** (FinCon CVRF, CryptoTrade). Debate is the costliest part and has no isolated evidence.
- CryptoTrade's ablation says feeding the LLM **transaction and position statistics** matters more than news. Our overlay prompt should include the position, unrealized P&L, recent fills and fee drag.
- AMA suggests a stronger GPT model will not fix a weak design. The structure of the overlay matters more.

### Gaps
- No isolated ablation of multi-agent debate in TradingAgents was found.
- FinAgent (multimodal, with tool-augmented reflection) and FinRobot numbers were not fetched. FinRobot is a platform more than a benchmarked trader.
- StockAgent was not checked.

## Q2. Public live LLM trading competitions (nof1 Alpha Arena and others)

### Takeaway
Alpha Arena Season 1 (Oct 18 to about Nov 3 2025, real $10k each, Hyperliquid crypto perpetuals at 10-20x leverage) ended with **4 of 6 models losing money**. The winners were the **low-frequency, concentrated, high-conviction** traders (Qwen3-Max, DeepSeek). The **over-traders** (GPT-5, Gemini 2.5 Pro) lost about 55-63%. The organizers attributed much of the result to randomness and fixed model "personalities".

### Cited Findings
- Final standings: Qwen3-Max $12,231, DeepSeek $10,489, Claude Sonnet 4.5 $5,799, Gemini 2.5 Pro $5,445, Grok $4,208, GPT-5 $4,126 — [ForkLog](https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/); [helloacm](https://helloacm.com/alpha-arena-how-ai-performs-in-the-real-crypto-market/)
- Qwen3-Max reportedly returned 22.3% with a 30.2% win rate across only 43 trades in about 17 days. A low win rate with a positive result points to letting winners run — [iWeaver](https://iweaver.ai/blog/alpha-arena-ai-trading-season-1-results); [GNcrypto](https://www.gncrypto.news/news/qwen-wins-alpha-arena-season-1-with-22-percent-returns/)
- GPT-5 was described as a "restless scalper" paying heavy fees. Gemini traded the most and often hit the maximum position count. DeepSeek was reported (mid-season snapshot) at 17 trades and the lowest fees among top models, with a median leverage of 10x. These trade counts are unofficial snapshots — [innobu](https://www.innobu.com/en/articles/ai-trading-competition-chinese-llms-investors.html); [helloacm](https://helloacm.com/alpha-arena-how-ai-performs-in-the-real-crypto-market/)
- Coverage summarized the lesson as "disciplined, low-frequency strategies outperformed over-trading". Biases were persistent: Claude was mostly long, Grok and Gemini leaned short, and Qwen used the largest sizes and the highest self-reported confidence — [iWeaver](https://iweaver.ai/blog/alpha-arena-ai-trading-season-1-results); [helloacm](https://helloacm.com/alpha-arena-how-ai-performs-in-the-real-crypto-market/)
- The founder said the models showed consistent biases across prompt iterations and that "LLMs don't really handle numerical time series data very well", while that was all the context they were given — [ForkLog](https://forklog.com/en/four-out-of-six-ai-models-suffer-losses-in-trading-tournament/)
- Season 1.5 (US stocks, including a "Max Leverage" mode) reportedly ended with most models losing again. This is secondary-source only — [ForkLog](https://forklog.com/en/new-season-of-ai-trading-tournament-kicks-off/); [abit.ee](https://abit.ee/et/tehisintellekt/ai-mudelid-ebaonnestusid-taas-borsil-alpha-arena-season-1-5-tulemused)
- The prompt format is **not officially published**. Community reconstructions use a JSON output per coin with `signal`, `quantity`, `profit_target`, `stop_loss`, `invalidation_condition`, `confidence`, `risk_usd` and `justification` — [gist (wquguru)](https://gist.github.com/wquguru/7d268099b8c04b7e5b6ad6fae922ae83); [CSDN note on unofficial status](https://gitcode.csdn.net/6a2c18e1662f9a54cb7dd917.html)
- A community replication (kojott/LLM-trader-test, DeepSeek) moved from a 3-minute loop to **15m/1h/4h multi-timeframe context** to cut noise. It uses mechanical exits (SL/TP hit, 1h structure break, 4h trend reversal) and will not close a position by discretion while it is within 20% of the stop — [GitHub](https://github.com/kojott/LLM-trader-test)

### Inferences
- The relevant lesson for us is **fee drag from overtrading**, not leverage, which we cannot use on spot. Our scoring rewards trade count, so the trades we add should be cheap maker-limit trades sized for positive expectancy, not LLM-driven scalps.
- A pre-committed **exit plan with an invalidation condition** at entry was the common pattern in the reconstructed prompts, and it stopped models from closing trades on noise. This maps directly onto our exit-review overlay.

### Gaps
- No official nof1 per-model trade counts, fees or holding times were found (nof1 promised them for Season 2).
- No verified results for a 2026 Season 2 were found; search returned nothing relevant.

## Q3. Critical evaluations: look-ahead leakage, costs, overtrading, calibration; does LLM-as-filter help?

### Takeaway
Independent evaluations consistently find that reported LLM alpha **shrinks or disappears** over longer windows, with realistic costs, and after the model's knowledge cutoff. LLMs are poorly calibrated and weak at numeric risk. The recommended architecture is the one we already have: **a quant model makes the decision, the LLM extracts information or acts as a filter, and hard-coded modules handle risk and execution**. Direct evidence that an LLM veto over a technical signal helps is thin. The best support is an LLM news-scoring overlay on momentum that improved Sharpe out of sample.

### Cited Findings
- FINSABER (KDD 2026): over two decades and 100+ symbols, earlier-reported LLM advantages "deteriorate significantly". LLM strategies are too conservative in bull markets and too aggressive in bear markets. The authors recommend prioritizing trend detection and regime-aware risk controls over adding framework complexity — [arXiv 2505.07078](https://arxiv.org/pdf/2505.07078)
- The Alpha Illusion (May 2026):
  - TradingAgents' Sharpe falls from 0.43 to 0.22 net of costs, and QuantAgent's from -0.96 to -1.15.
  - Many reported Sharpe confidence intervals include zero, and most frameworks leave frictions unmodeled.
  - LLM confidence is "linguistic" and overconfident. Models keep their prior beliefs: one still recommended buying a tech stock when 60% of the evidence was negative.
  - The authors recommend that the LLM extract information upstream while calibration, risk and execution sit in separate modules — [alphaxiv overview](https://www.alphaxiv.org/overview/2605.16895)
- Parametric look-ahead bias: a model's memory of past outcomes inflates in-sample backtests on memorized dates — [arXiv 2605.24564](https://arxiv.org/abs/2605.24564)
- A position paper reviewing 164 papers (2023-2025) names look-ahead as one of five recurring biases; no single bias was explicitly addressed in more than 28% of papers — [alphaxiv 2602.14233](https://www.alphaxiv.org/abs/2602.14233)
- LLM-as-overlay evidence: "ChatGPT in Systematic Investing" used ChatGPT to score firm news as an overlay on cross-sectional momentum. It reported higher Sharpe and Sortino ratios, including in an out-of-sample period after the pre-training cutoff, and the result held after costs. This search snippet came via [arXiv 2510.26228](https://arxiv.org/pdf/2510.26228v1); the primary paper was not fetched.
- A comparison of LLMs on technical buy/sell/hold signals found persistent numerical hallucination, and its headline numbers are labelled simulated or illustrative — [arXiv 2607.15414](https://arxiv.org/pdf/2607.15414)
- In AMA, the CryptoPulse critique and LiveTradeBench, the LLMs' risk styles were fixed per model — [AMA](https://arxiv.org/pdf/2510.11695); [LiveTradeBench](https://huggingface.co/papers/2511.03628)

### Inferences
- In our live setting look-ahead does not apply, because live decisions cannot leak the future. But any backtest of our GPT overlay on past dates is contaminated. Present the GPT overlay to judges as **validated live**, with a decision log, rather than by backtest.
- Our design (LightGBM signal, then GPT veto, then hard risk rules) matches the critics' recommended split. Keep sizing, stops and limits in code; do not let GPT size positions.
- Because LLMs are overconfident and keep their priors, a veto should be **asymmetric**. Let it block an entry only when it names a concrete red flag (a news event, a broken level), not because of general caution. Otherwise FINSABER's "too conservative in bull markets" failure will cut our trade count and return.

### Gaps
- No controlled study was found of an LLM veto over an ML/technical signal on crypto at hourly frequency. The meta-labeling-with-LLM literature was not located.
- Latency and cost per decision were rarely reported, beyond TradingAgents' 11 LLM calls per decision.

## Q4. Cheap prompt/agent patterns (~$1 budget) that work at hourly frequency on crypto spot

### Takeaway
Full multi-agent debate (about 11+ calls per decision) does not fit the budget and lacks evidence. The cheap patterns with the best support are:
1. One structured-JSON call per pair per hour with a mini model, given a compact numeric and position context (CryptoTrade's ablation).
2. A pre-committed exit plan with an invalidation condition (Alpha Arena reconstructions, kojott).
3. A once-a-day or event-triggered reflection call that writes a short "lessons" string reused in later prompts (FinCon CVRF, CryptoTrade reflection).
4. Hard-coded risk rules outside the LLM (FinCon CVaR, Alpha Illusion).

### Cited Findings
- CryptoTrade's reflection ablation (+11 points on ETH bull) and its transaction-statistics ablation (+16 points) support adding a short reflection summary and position/P&L stats to the prompt — [arXiv HTML v2](https://arxiv.org/html/2407.09546v2)
- FinCon's belief update is a verbal summary generated from P&L trends and passed to the agents, a cheap single text update per episode — [NeurIPS slides](https://nips.cc/media/neurips-2024/Slides/94352_FOt8cLF.pdf)
- TradingAgents itself splits work between cheap "quick-think" models for summaries and an expensive model only for the final decision — [arXiv HTML v4](https://arxiv.org/html/2412.20138v4)
- AMA used GPT-5-nano to summarize and filter news once per day for all agents, a shared, cached news digest — [arXiv PDF](https://arxiv.org/pdf/2510.11695)
- kojott's bot sends multi-timeframe indicators (EMA 20/50/200, RSI14, MACD, ATR, volume) plus a portfolio snapshot, and asks for per-coin JSON with SL/TP/invalidation — [GitHub](https://github.com/kojott/LLM-trader-test)
- nof0 includes an LLM budget module (`go/pkg/llm/budget.go`) and a prompt builder (`go/pkg/executor/prompt_build.go`) — [GitHub](https://github.com/wquguru/nof0)

### Inferences (concrete, implementable in hours)
1. **Batch all 7 pairs into one hourly call.** Send a compact table (model probability, ensemble agreement, 1h/4h returns, RSI, ATR%, distance to EMA50, current position and unrealized P&L, fee break-even) and get back one JSON object per pair, e.g. `{pair: {veto: bool, reason_code, invalidation_price, confidence}}`. That is about 24 calls a day instead of 168. At about 2k input and 300 output tokens with a mini/nano model, a day should cost well under $1.4. This is an estimate; check the gateway's price list.
2. **Asymmetric veto rule.** Default to allowing the ML entry. Veto only with a reason code from a fixed list (major negative headline, extreme stretch beyond the ATR threshold, illiquid book). Log every veto and its counterfactual outcome for the judges as "real AI use" evidence.
3. **Exit plan at entry.** When an entry fills, have GPT (or code) record a take-profit, a stop and a textual invalidation condition. The hourly exit review may close early only if the invalidation condition is met. This avoids GPT-5-style churn.
4. **One reflection call every 6-12 hours.** Feed the last N closed trades (pair, entry reason, P&L net of fees, whether a veto was right) and ask for 3 bullet "lessons" of 60 words or fewer. Prepend them to the hourly prompt, FinCon/CryptoTrade-style. This costs only cents.
5. **Shared news digest.** Summarize headlines once an hour into per-asset sentiment (we already do this) and reuse the digest in the decision call rather than sending raw headlines, as AMA did.
6. **Keep risk in code.** Fixed fractional sizing, maximum exposure per pair, a daily loss stop and a per-trade stop. Do not let GPT choose size; Qwen's large self-chosen sizes were luck-dependent.
7. **Skip** bull/bear debate, persona agents and backbone voting. They multiply cost and have no supporting ablation, and AMA's "Vote" ensemble was mediocre.

### Gaps
- No published study measured the token cost against the benefit at an hourly frequency.
- No evidence was found on maker-only limit-order execution for LLM agents.

## GitHub repos

Stars and last push come from the GitHub API on 2026-10-22. "Crypto" means native crypto data or exchange support.

| Repo | Stars | Last push | What it does | Files worth copying | Crypto? |
|---|---|---|---|---|---|
| [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) | 112,867 | 2026-10-16 | LangGraph multi-agent trader: analysts, bull/bear debate, trader, three risk debaters, portfolio manager | `tradingagents/graph/{trading_graph,setup,conditional_logic,propagation}.py` (agent graph); `agents/researchers/{bull,bear}_researcher.py`; `agents/risk_mgmt/*_debator.py`; `agents/managers/portfolio_manager.py`; `memory/{reflection,log,settlement}.py` (reflection on settled trades); `agents/schemas.py`, `structured.py` (structured outputs) | Stock-focused vendors (yahoo, alpha_vantage, polymarket, reddit, stocktwits); no native crypto exchange found in the file tree |
| [Tomortec/CryptoTradingAgents](https://github.com/Tomortec/CryptoTradingAgents) | 284 | 2025-12-22 | Crypto fork of TradingAgents | `tradingagents/dataflows/{binance_utils,coindesk_utils,blockbeats_utils,alternativeme_utils,coinstats_utils}.py` (crypto data incl. Fear & Greed); `agents/managers/risk_manager.py`; `agents/utils/memory.py` | Yes (Binance, CoinDesk, alternative.me) |
| [Xtra-Computing/CryptoTrade](https://github.com/Xtra-Computing/CryptoTrade) | 166 | 2024-12-01 | EMNLP 2024 reflective agent for BTC/ETH/SOL daily trading | `run_agent.py` (analyst and trader prompts), `generate_reflections.py` (reflection loop) | Yes (on-chain plus news) |
| [pipiku915/FinMem-LLM-StockTrading](https://github.com/pipiku915/FinMem-LLM-StockTrading) | 963 | 2024-08-18 (stale) | Layered memory trader | `puppy/memorydb.py`, `puppy/memory_functions/{recency,importance_score,decay,compound_score}.py` (memory scoring), `puppy/reflection.py`, `puppy/prompts.py` | No (stocks) |
| [wquguru/nof0](https://github.com/wquguru/nof0) | 2,752 | 2025-12-07 | Open-source reproduction of the nof1 Alpha Arena (Go) | `go/pkg/executor/prompt_build.go`, `go/pkg/llm/{budget,prompt,retry}.go`, `go/internal/model/decisioncyclesmodel.go` | Yes (Alpha Arena-style perpetuals) |
| [kojott/LLM-trader-test](https://github.com/kojott/LLM-trader-test) | 144 | 2025-12-29 | Alpha Arena-inspired DeepSeek bot, 15-minute loop, multi-timeframe | `bot.py` (loop and JSON schema), `prompts/`, `backtest.py`, `dashboard.py` | Yes (Binance data; Hyperliquid execution) |
| [51bitquant/ai-hedge-fund-crypto](https://github.com/51bitquant/ai-hedge-fund-crypto) | 626 | 2025-09-05 | Crypto port of ai-hedge-fund; LangGraph nodes combining classic strategies with an LLM portfolio manager | `src/graph/{risk_management_node,portfolio_management_node,data_node,state}.py`, `src/strategies/{macd,rsi}_strategy.py`, `src/agent/workflow.py`. Closest to our "quant signal, then LLM decision" pattern | Yes (Binance) |
| [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) | 64,148 | 2026-10-19 | Persona agents (Buffett, Damodaran, etc.), risk manager and portfolio manager. It is the HedgeFundAgent in AMA | Tree not fetched (API rate limit) | Stocks |
| [AI4Finance-Foundation/FinRobot](https://github.com/AI4Finance-Foundation/FinRobot) | 8,265 | 2026-09-28 | Financial agent platform (reports, analysis), not a benchmarked trader | Not inspected | Partial or unclear |
| [AI4Finance-Foundation/FinGPT](https://github.com/AI4Finance-Foundation/FinGPT) | 21,447 | 2026-09-23 | Financial LLMs and sentiment models | Sentiment fine-tunes; not needed for a GPT gateway | General |
| [HKUDS/AI-Trader](https://github.com/HKUDS/AI-Trader) | 22,922 | 2026-06-11 | "Agent-native" fully automated trading arena | Tree not fetched (rate limit) | Unverified |
| [Open-Finance-Lab/AgenticTrading](https://github.com/Open-Finance-Lab/AgenticTrading) | 783 | 2026-10-22 | Agentic trading framework (no description) | Not inspected | Unverified |
| [0xethanq/astra-quant-agent](https://github.com/0xethanq/astra-quant-agent) | 393 | 2026-10-03 | Crypto system: multi-LLM "council" debate with deterministic code risk gates. The repo name is unrelated to any "astra" model | `astra_backend/council/debate.py`, `astra_backend/execution/risk_gates.py`, `astra_backend/llm/query_cache.py` (LLM response cache, useful for budget) | Yes |
| [TauricResearch/Trading-R1](https://github.com/TauricResearch/Trading-R1) | 507 | 2025-09-15 | Placeholder for an RL-tuned trading LLM | None | n/a |

### Most useful to copy for our system within hours
- **Reflection on settled trades:** TradingAgents `tradingagents/memory/reflection.py` and `settlement.py`, or CryptoTrade `generate_reflections.py`.
- **Quant signal feeding risk and LLM manager nodes:** `51bitquant/ai-hedge-fund-crypto` `src/graph/risk_management_node.py` and `portfolio_management_node.py`.
- **Exit-plan JSON schema:** `kojott/LLM-trader-test` `bot.py` and the nof0 prompt builder.
- **Deterministic risk gates around an LLM decision:** `astra-quant-agent/astra_backend/execution/risk_gates.py`.
- **Crypto data utilities (Fear & Greed, crypto news):** `Tomortec/CryptoTradingAgents/tradingagents/dataflows/`.
