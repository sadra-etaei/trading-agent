# Wallex-only deployment experiment

Protocol: protocol.json. Historical evaluation only; no live configuration changed.

Four chronological monthly folds. Fit, calibration, threshold selection and test are separate, with 13-hour purges. Fees 0.30% and slippage 0.15% per side; entry delay 1 minute. Monthly accounts restart at 5M TMN; compounded return chains their percentage returns. Maximum drawdown below is the worst within-month drawdown, not continuous-period drawdown. GPT/news omitted.

| Candidate | Compound return | Trades | Pooled PF | Positive months | Worst month DD | 3-day mean | 3-day worst | Eligible |
|---|---|---|---|---|---|---|---|---|
| local | +5.27% | 26 | 1.36 | 3/4 | 5.47% | +0.12% | -2.81% | False |
| local_premium | +7.26% | 24 | 1.53 | 3/4 | 7.45% | +0.17% | -2.94% | False |
| disconnected | -5.69% | 25 | 0.64 | 0/4 | 7.64% | -0.09% | -3.85% | False |
| full_reference | +23.77% | 77 | 1.80 | 3/4 | 7.40% | +0.55% | -4.50% | True |

Selection under prewritten rule: **none**.

Monthly details:

| Candidate | Month | Return | DD | Trades | PF |
|---|---|---|---|---|---|
| local | 2026-07-01 | +0.44% | 4.20% | 4 | 1.13 |
| local_premium | 2026-07-01 | +1.02% | 4.11% | 4 | 1.32 |
| disconnected | 2026-07-01 | +0.00% | 0.00% | 0 | 0.00 |
| full_reference | 2026-07-01 | -0.30% | 0.91% | 3 | 0.67 |
| local | 2026-08-01 | +4.43% | 1.66% | 4 | 3.53 |
| local_premium | 2026-08-01 | +6.04% | 0.00% | 3 | inf |
| disconnected | 2026-08-01 | -0.22% | 2.02% | 2 | 0.88 |
| full_reference | 2026-08-01 | +5.34% | 4.13% | 15 | 1.67 |
| local | 2026-09-01 | +0.00% | 0.00% | 0 | 0.00 |
| local_premium | 2026-09-01 | +0.00% | 0.00% | 0 | 0.00 |
| disconnected | 2026-09-01 | -5.13% | 7.64% | 19 | 0.56 |
| full_reference | 2026-09-01 | +8.47% | 7.40% | 46 | 1.54 |
| local | 2026-10-01 | +0.36% | 5.47% | 18 | 1.04 |
| local_premium | 2026-10-01 | +0.12% | 7.45% | 17 | 1.01 |
| disconnected | 2026-10-01 | -0.37% | 2.20% | 4 | 0.84 |
| full_reference | 2026-10-01 | +8.65% | 1.86% | 13 | 3.77 |

Limitations: these historical periods have already been explored elsewhere in this project. This is chronological testing, not an untouched prospective holdout. Rolling windows overlap. Premium-reference coverage is in premium_coverage.json; missing references are not synthesized. Full-reference benchmark needs Binance and is not deployable under the stated constraints. Source data hashed in data_manifest.json. Production config and latest.joblib hashes unchanged.

## Independent audit

Evaluation ends 2026-10-23 17:00:00+00:00. Production config and model hashes unchanged; 12 fitted artifacts and 16 evaluation results checked; local models have no Binance/funding features; all split purges verified.

Both local candidates made zero trades in 73.58% of rolling three-day windows (median 0 trades per window). All simulated challenge windows finished flat in Toman. Detailed verification: audit.json.

## Currency-adjusted returns

All earlier returns are nominal Toman. Over July 1 through October 23 2026, the stored USDTTMN price rose from 173,000 to 268,986 Toman (+55.48%). USDT-equivalent return = (1 + Toman return) / (end USDTTMN / start USDTTMN) - 1. This uses USDT as a dollar proxy, before any additional conversion costs. Monthly account percentage returns are chained as in the original summary.

| Candidate | Toman return | USDT-equivalent return |
|---|---|---|
| local | +5.27% | -32.29% |
| local_premium | +7.26% | -31.02% |
| disconnected | -5.69% | -39.34% |
| full_reference | +23.77% | -20.40% |

Neither Wallex-only candidate preserved dollar-equivalent capital. Holding Toman returned 0% nominal but -35.68% in USDT terms. Holding USDT returned 0% in USDT terms and +55.48% in Toman before conversion costs.
