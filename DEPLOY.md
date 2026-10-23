# Deploying the bot on Liara (PaaS)

This is the brief for the deployment agent. The bot runs strategy **S1** (hourly ML model on 16 Toman pairs,
GPT overlay, Trump / Fed / news event gate). It trades the Wallex challenge account until
**Monday 2026-10-26 20:00 Tehran** (deadline extended; submission Monday 23:00), then stays idle in Toman.

## Files

| file | purpose |
|---|---|
| `Dockerfile` | Python 3.14 image, exact library versions (`requirements-deploy.txt`), code, model, seed data |
| `.liaraignore`, `.gitignore` | keep the datasets (`data/`, 411 MB), `runs/`, model experiments, research notes, tests and `.env` out of the `liara deploy` upload (about 21 MB is uploaded: code, `models/latest.joblib`, `deploy/seed/`, `agent_cache/`) |
| `.dockerignore` | keeps `.env`, local data, runs and tests out of the image |
| `liara.json` | Docker platform, port 8080, persistent disk `bot-data` mounted at `/app/persist` |
| `deploy/start.py` | entry point: status page, first-start seeding, auto-resume, restart after crashes |
| `deploy/seed/` | last 50 days of 1-minute candles (16 pairs), market rules, Fear & Greed (made by `deploy/make_seed.py`) |

## Steps

1. **Create the app** on Liara (Docker platform) and put its ID in `liara.json` (`"app"`, now `CHANGE-ME-liara-app-id`).
2. **Create a disk** named `bot-data` (1 GB is plenty) for the app. It is mounted at `/app/persist`. Everything
   that must survive a restart lives there: `runs/` (decision logs, `state.json`), `data/`, `agent_cache/`
   (GPT responses and the spend counter).
3. **Set environment variables** in the Liara panel (never in the repo or the image):

   | variable | value |
   |---|---|
   | `WALLEX_API_KEY` | Wallex API key of the challenge account |
   | `OPENAI_API_KEY` | GPT gateway key |
   | `OPENAI_BASE_URL` | `https://ai.boxapi.ir/ai/v1` (the gateway; check the value in your local `.env`) |
   | `BOT_MODE` | `shadow` first; `live` to trade |
   | `BOT_CONFIRM_LIVE` | `yes` only when `BOT_MODE=live` (second switch, on purpose) |
   | `BOT_AGENT` | `true` (GPT overlay + event gate scoring); `false` = ML only |
   | `HTTPS_PROXY`, `NO_PROXY` | only if Binance is blocked, see step 5 |

4. **Deploy** with `BOT_MODE=shadow`: `liara deploy` from the project root. Liara builds the Dockerfile.
5. **Check the network from Liara**, in the app's console (`liara shell` or the web console):

   ```bash
   python cli.py check-net
   ```

   Critical services: Wallex (markets, order book, candles) and **Binance spot candles**. S1 uses Binance data for
   its strongest features and for the crash guard (without it the held-out result drops from +20.47% to +2.33%,
   and the bot skips new entries). On Liara it failed with "Connection refused". First try the mirror: if the
   line `alt: Binance mirror data-api.binance.vision` is OK, set `BINANCE_SPOT_URL=https://data-api.binance.vision`
   and redeploy. Otherwise set
   `HTTPS_PROXY=http://<proxy outside Iran>:<port>` and `NO_PROXY=api.wallex.ir,ai.boxapi.ir` (Wallex and the
   gateway stay direct), redeploy and re-run `check-net`. `BINANCE_SPOT_URL` / `BINANCE_FUTURES_URL` can point to
   another mirror of the same API. News feeds are optional; a failed feed is skipped.
6. **Watch the shadow** for at least one hourly decision (decisions happen at :30 Tehran time, 3 s after each UTC
   hour closes): open the app URL. `/health` returns `ok`; `/` returns JSON with the run, equity, positions,
   the last decisions and GPT spend. Logs: Liara's log view.
7. **Smoke test, once, before live** (a real minimum-size buy and sell of USDT/TMN, about 110,000 TMN):

   ```bash
   python cli.py --set data.dir=/app/persist/data smoke-order --yes
   ```

   It must report the order as filled with sensible price, quantity and fee. Send the output to the developer.
8. **Go live**: stop any other copy of the bot (the shadow on the developer's PC too: never two bots on one
   account), set `BOT_MODE=live` and `BOT_CONFIRM_LIVE=yes`, redeploy. The live run starts from the account's
   Toman balance.

## Binance through a WireGuard tunnel (Liara cannot reach Binance directly)

On Liara every Binance request fails ("Connection refused" / "Network is unreachable"), and without Binance data S1
falls from +20.47% to +2.33% on the held-out month, so the bot skips new entries until Binance is reachable. The
container ships **wireproxy** (userspace WireGuard, pinned v1.1.3, checksum-verified at build). It needs no root and
no special container permissions. `deploy/tunnel.py` starts it before the bot when a WireGuard config is present,
sends Binance and the news feeds through it, and keeps **Wallex and the GPT gateway direct** (`NO_PROXY`).

**Server:** a WireGuard server outside Iran and outside the US (Binance blocks US IPs). Use its normal client
`.conf` file (`[Interface]` + `[Peer]`). Lines like `PostUp`, `PostDown`, `Table` are ignored automatically.

**Give the config to Liara: option A, one environment variable (recommended)**

Encode the file as a single base64 line and set it as `WG_CONFIG_B64` in the Liara panel. On Windows PowerShell
(copies the line to the clipboard; replace the path with your file's):

```powershell
[Convert]::ToBase64String([IO.File]::ReadAllBytes("C:\Users\you\wg.conf")) | Set-Clipboard
```

On Linux: `base64 -w0 wg.conf`. On macOS: `base64 -i wg.conf`. Paste the result as the value of `WG_CONFIG_B64`
(a secret, like the API keys; never commit the file or the line).

**Option B, a file on the disk:** put the file at `/app/persist/wg.conf` on the `bot-data` disk (Liara console
or file manager), or point `WG_CONFIG_FILE` at another path.

**Check it after redeploying:**
- the status page (`/`) shows `"tunnel": "up"` (or the reason it is not);
- the logs show `tunnel: up (proxy 127.0.0.1:25345, direct: api.wallex.ir,ai.boxapi.ir,...)` and no more
  `binance 1h ... failed`;
- `check-net` from the Liara console must be told to use the tunnel, because only the bot's process sets it:

  ```bash
  HTTPS_PROXY=http://127.0.0.1:25345 NO_PROXY=api.wallex.ir,ai.boxapi.ir python cli.py check-net
  ```

**If the tunnel stays down:** Iran's filtering sometimes blocks the WireGuard protocol itself (the log shows no
handshake). Then use an HTTP(S) proxy outside Iran instead: unset `WG_CONFIG_B64`, set
`HTTPS_PROXY=http://user:pass@host:port` and `NO_PROXY=api.wallex.ir,ai.boxapi.ir`. If the Liara build cannot
download wireproxy from GitHub, the build fails at that step; tell the developer to vendor the binary.

## Behaviour

- **Restarts:** after a redeploy or crash the bot resumes the latest run of its mode (`--resume`) with its open
  positions and decision log. A crash is retried after 30 s. Switching `shadow` -> `live` starts a new live run
  (shadow and live runs are separate).
- **Challenge end:** no new entries from Monday 19:00, every position sold to Toman from 19:45, trading stops at
  20:00 Tehran. The run is then marked `FINISHED` and the container only serves the status page. To trade again later,
  change `challenge.end` in `config.yaml` and delete the `FINISHED` file on the disk.
- **Risk:** S1 limits (8% daily loss stops new entries, 30% drawdown sells everything and halts), Binance crash
  guard, event gate (blocks entries after severe negative news, sells on a confirmed shock).
- **GPT budget:** hard cap $1.50 (`agent.budget_usd`), spend so far is carried over from the image on first start;
  at the cap the agent switches itself off and trading continues on the ML model and the risk rules.
- **Time:** the bot works in UTC internally; the container's timezone does not matter.

## Checked locally

`deploy/start.py` was run on the developer PC with a temporary persistent folder: first start seeds the disk and
starts a shadow run; a second start resumes the same run; `BOT_MODE=live` without `BOT_CONFIRM_LIVE=yes` refuses
to start; `/health` and `/` answer. The Docker image itself was not built locally (Docker Desktop was not
running); Liara builds it from the Dockerfile.
