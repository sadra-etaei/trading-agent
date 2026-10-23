# Hackathon demo evidence

No live strategy trades. Shadow fills are simulated. Earlier shadow configurations differ from current S1.

Backtest: +20.47% Toman, +3.57% USDT-equivalent, 42 trades, 3.68% maximum Toman drawdown. GPT disabled. Source: bt-20261022-154705-test-sz-S1.

## All completed shadow trades

Times: Tehran, October 2026. P&L includes recorded modeled fees.

| Pair | Entry | Exit | Notional TMN | Net P&L TMN | Return | Exit reason |
|---|---|---|---:|---:|---:|---|
| ADATMN | Oct 22 02:30:48 | Oct 22 02:30:51 | 253,996 | -2,465 | -0.97% | stop_loss |
| AVAXTMN | Oct 22 03:30:53 | Oct 22 05:04:23 | 227,905 | -4,204 | -1.84% | stop_loss |
| XRPTMN | Oct 22 03:30:53 | Oct 22 05:09:21 | 182,572 | -2,818 | -1.54% | stop_loss |
| SUITMN | Oct 22 05:30:51 | Oct 22 06:07:53 | 219,733 | -3,852 | -1.75% | stop_loss |
| ADATMN | Oct 22 07:31:02 | Oct 22 09:24:41 | 206,397 | -3,771 | -1.83% | stop_loss |
| UNITMN | Oct 22 09:30:57 | Oct 22 10:43:21 | 266,975 | -5,506 | -2.06% | stop_loss |
| UNITMN | Oct 22 13:31:01 | Oct 22 14:31:14 | 903,357 | -5,329 | -0.59% | agent_exit |
| SUITMN | Oct 22 14:31:23 | Oct 22 14:31:25 | 861,976 | -14,803 | -1.72% | stop_loss |

Recorded closed P&L sum: -42,747 Toman across independent experiments; not a continuous-account return.

## Unfinished historical entries

| Pair | Entry | Source run |
|---|---|---|
| SUITMN | Oct 22 11:31:10 | shadow-20261021-232050 |
| ADATMN | Oct 22 12:31:37 | shadow-20261021-232050 |
| SUITMN | Oct 22 16:31:00 | shadow-20261022-092720 |

Open index.html for the demo view. s1-backtest.png is 1920×1080; s1-backtest.svg is scalable. shadow-trades.png shows all eight outcomes. CSV files retain decision IDs and original UTC timestamps. evidence.json records source hashes.

USDT-equivalent return was recomputed from the saved equity curve and explicit boundary prices (229,298 to 266,721 Toman/USDT), giving +3.57%; older strategy notes reported +3.38% using a different valuation boundary. Source quotes are carried forward across gaps (maximum age 610 minutes); no missing prices are interpolated. Additional USDT conversion costs are excluded.
