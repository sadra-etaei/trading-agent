# Four-hour return forecasting: quick screening

Offline Wallex-only USDT markets; no live changes. All returns below are in USDT. Benchmark: hold USDT (0%).

Compounded monthly return: **-7.92%**. Completed trades: **14**. Pooled PF: **0.45**. Eligible under prewritten rule: **False**.

| Test month | USDT return | Trades | PF | Max monthly DD | Forecast MAE | No-change MAE |
|---|---:|---:|---:|---:|---:|---:|
| 2026-07 | +5.53% | 5 | 9.13 | 4.32% | 0.740% | 0.733% |
| 2026-08 | -2.23% | 1 | 0.00 | 2.24% | 1.115% | 1.065% |
| 2026-09 | -10.36% | 7 | 0.00 | 11.08% | 0.947% | 0.932% |
| 2026-10 | -0.44% | 1 | 0.00 | 2.79% | 0.818% | 0.823% |

Monthly accounts restart at 18.3 USDT; compounded returns chain those percentages. DD is worst within-month, not full compounded-period DD. 4h endpoint prediction target differs from actual stopped trade payoff. Point forecasts only: uncertainty model not included. Costs: 0.13% fee each side plus per-pair half-spread; fixed net entry buffer 0.20%; one-minute delay and active minutes required. 90% exposure, 2% stops, one position. Cash idle in USDT. Initial/final Toman conversion costs excluded. Candle fills approximate historical quotes; historical periods previously examined; no GPT. This is a screening test, not an untouched prospective validation.