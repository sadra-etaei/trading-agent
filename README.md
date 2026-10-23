# Wallex AI Trading Engine

Automated, AI-driven spot trading engine for the Wallex challenge (Starcoach Agentic AI Hackathon).
It trades liquid Toman (TMN) spot pairs, manages risk with hard limits, logs the reason for every
decision, and finishes every challenge fully in TMN.

- **How it decides:** a calibrated ML entry model ranks trades by expected value after fees; a
  GPT overlay agent reviews each trade with news / Iranian macro context and may veto or reduce;
  a deterministic risk manager has the final word. Details: [STRATEGY.md](STRATEGY.md).
- **One code path** for backtest, shadow and live: only the broker changes.
- **Plan and status:** [PLAN.md](PLAN.md).

## Setup

Python 3.11+.

```bash
pip install -r requirements.txt
```

Secrets go in a local `.env` file, which is git-ignored and loaded automatically by `cli.py`
(real environment variables take precedence). Copy the template and fill it in:

```bash
cp .env.example .env
```

| variable | needed for |
|---|---|
| `WALLEX_API_KEY` | `live`, `verify-api` |
| `OPENAI_API_KEY` | GPT overlay + headline scoring (`agent.enabled: true`), `backtest --agent` |

All parameters live in [config.yaml](config.yaml). Every run stores the full config and its hash.

## Run it end to end

```bash
python cli.py download                 # 1 year of 1-minute candles (public API) + Fear & Greed history
python cli.py train                    # walk-forward evaluation, final model -> models/latest.joblib
python cli.py backtest                 # backtest of record on the held-out test window + 3-day challenge windows
python cli.py experiment-selection     # two-stage selection model vs. EV ranking
python cli.py shadow                   # live market data, simulated fills (no API key needed)
python cli.py verify-api               # read-only check of private endpoints (needs WALLEX_API_KEY)
python cli.py lag-log                  # Strategy 3: log Wallex vs Binance books (no orders)
python cli.py lag-backtest --optimistic # Strategy 3: historical lag-buy test on 1-minute candles
python cli.py lag-replay               # Strategy 3: lag-buy replay on the logged real books
python cli.py veto-report runs/<run>      # what vetoed / event-gate-skipped entries would have done
python cli.py smoke-order --yes        # REAL minimum-size buy + sell of USDT/TMN to verify order handling
python cli.py --set agent.enabled=true live --yes                      # real orders, GPT agent on
python cli.py --set agent.enabled=true live --yes --resume runs/<id>   # restart after a crash
```

Other commands:

```bash
python cli.py record                   # real order books (all pairs, ~2s) + trades -> data/live/, run continuously
python cli.py fv-research              # Strategy 2: fair-value gap stats on recordings
python cli.py fv-backtest              # Strategy 2: replay with conservative maker fills
python cli.py report runs/<run_id>     # rebuild a run's report
python cli.py shadow --resume runs/<run_id>   # resume after a restart (open positions restored)
python -m pytest -q                    # unit tests + end-to-end test on synthetic data
```

For the challenge, set `challenge.start` and `challenge.end` in `config.yaml`: the engine stops
opening trades 3 hours before the end, sells everything to TMN in the last 45 minutes, and stops
once it is flat.

## What a run produces (`runs/<run_id>/`)

| file | content |
|---|---|
| `meta.json` | mode, window, model version, config + hash, data manifest (row counts and file hashes) |
| `decisions.jsonl` | **decision log**: every scan, entry signal, agent veto, risk block, fill, exit, risk event, scored headline |
| `trades.csv` | closed trades: entry/exit, size, fees, P&L, exit reason, holding time |
| `equity.csv` | equity, cash, exposure, drawdown per bar |
| `challenge_windows.csv` | backtest only: every 3-day window simulated with challenge rules |
| `report.md`, `metrics.json`, `equity.png` | metrics the judges asked for (return, drawdown, number of trades, volume, average trade size, ...) |

Models are in `models/` (`entry-<version>.joblib` + `train-report-<version>.md`); every decision
records the model version that made it. GPT prompts and responses are stored in `agent_cache/`
keyed by a hash of the input, so agent decisions can be replayed exactly.

## Repository layout

```
cli.py                  command line
config.yaml             all parameters
engine/
  wallex/               REST client, market rules, API verification
  data/                 history downloader, Parquet store, resampling, order book recorder
  sentiment/            Fear & Greed, funding rates, RSS news + GPT scoring
  features/             indicators and the feature matrix (same code for training and live)
  model/                labels, dataset, training, calibration, selection-model experiment
  strategy/core.py      model -> EV ranking -> agent -> risk -> intents
  agent/                GPT client with replay cache, overlay agent
  risk/                 portfolio/ledger, risk manager, exit logic
  broker/               shadow broker (live order book fills), Wallex broker (real orders)
  backtest/             event-driven simulator (1-minute exits), backtest runner
  live/                 shadow/live loop with crash-safe state
  reports/              metrics and charts
tests/                  unit tests + synthetic end-to-end test
```
