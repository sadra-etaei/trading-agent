"""Container entry point for Liara (DEPLOY.md).

- Serves a status page on $PORT (default 8080): `/health` -> "ok", `/` -> JSON with mode, run, equity, positions,
  last decisions and GPT spend. Liara expects the app to listen on a port.
- Keeps everything that must survive a restart on the persistent disk $BOT_PERSIST (default /app/persist):
  runs/ (decision logs, state.json), data/ (market data), agent_cache/ (GPT responses and spend).
  The first start seeds them from the image (deploy/seed, agent_cache).
- Runs the bot: BOT_MODE=shadow (default) or live. Live also needs BOT_CONFIRM_LIVE=yes (two switches, on purpose).
  After a restart it resumes the latest unfinished run of that mode (`--resume`), so open positions and the
  decision log continue. If the bot stops with an error it is restarted after 30 s, again with --resume.
- If a WireGuard config is provided (WG_CONFIG_B64 / WG_CONFIG / $BOT_PERSIST/wg.conf), starts the userspace tunnel
  first (deploy/tunnel.py): Binance and news go through it, Wallex and the GPT gateway stay direct.
- Once the challenge window is over and the run has finished, it never starts trading again: it only serves the
  status page.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

PERSIST = Path(os.environ.get("BOT_PERSIST", "/app/persist"))
MODE = os.environ.get("BOT_MODE", "shadow").strip().lower()
AGENT = os.environ.get("BOT_AGENT", "true").strip().lower() in ("1", "true", "yes")
PORT = int(os.environ.get("PORT", "8080"))
RUNS, DATA, CACHE = PERSIST / "runs", PERSIST / "data", PERSIST / "agent_cache"
log = logging.getLogger("start")
STATUS = {"mode": MODE, "agent": AGENT, "started_at": datetime.now(timezone.utc).isoformat(), "bot": "starting",
          "restarts": 0, "last_error": None}


def seed() -> None:
    for d in (RUNS, DATA, CACHE):
        d.mkdir(parents=True, exist_ok=True)
    src = ROOT / "deploy" / "seed"
    if src.exists():
        for f in src.rglob("*"):
            dst = DATA / f.relative_to(src)
            if f.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
    if not any(CACHE.iterdir()) and (ROOT / "agent_cache").exists():  # GPT spend so far + cached answers
        shutil.copytree(ROOT / "agent_cache", CACHE, dirs_exist_ok=True)


def latest_run() -> Path | None:
    runs = sorted((p for p in RUNS.glob(f"{MODE}-*") if (p / "meta.json").exists()), key=lambda p: p.name)
    return runs[-1] if runs else None


def finished(run: Path | None) -> bool:
    return bool(run) and (run / "FINISHED").exists()


def challenge_over() -> bool:
    from engine.config import load_config

    end = load_config("config.yaml").challenge.end
    return bool(end) and datetime.now(timezone.utc) >= datetime.fromisoformat(str(end))


def status() -> dict:
    run = latest_run()
    out = dict(STATUS, run=run.name if run else None, now=datetime.now(timezone.utc).isoformat())
    if run:
        try:
            st = json.loads((run / "state.json").read_text())
            out.update(cash=st.get("cash"), positions=st.get("positions"), halted=st.get("halted"),
                       halt_reason=st.get("halt_reason"))
        except (OSError, ValueError):
            pass
        eq = run / "equity.csv"
        if eq.exists():
            lines = eq.read_text().strip().splitlines()
            out["last_equity_row"] = lines[-1] if len(lines) > 1 else None
        dec = run / "decisions.jsonl"
        if dec.exists():
            tail = dec.read_text(encoding="utf-8").strip().splitlines()[-8:]
            out["last_decisions"] = [{k: json.loads(x).get(k) for k in ("kind", "ts", "symbol", "reason")} for x in tail]
    try:
        out["gpt_spend_usd"] = json.loads((CACHE / "_spend.json").read_text()).get("usd")
    except (OSError, ValueError):
        pass
    return out


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body = b"ok" if self.path.startswith("/health") else json.dumps(status(), indent=1, default=str).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain" if self.path.startswith("/health") else "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # keep the bot's log readable
        pass


def bot_args(resume: Path | None) -> list[str]:
    sets = [f"run.output_dir={RUNS.as_posix()}", f"data.dir={DATA.as_posix()}", f"agent.cache_dir={CACHE.as_posix()}",
            f"agent.enabled={'true' if AGENT else 'false'}"]
    args = [x for s in sets for x in ("--set", s)] + [MODE]
    if MODE == "live":
        args.append("--yes")
    if resume:
        args += ["--resume", str(resume)]
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    if MODE not in ("shadow", "live"):
        raise SystemExit(f"BOT_MODE must be shadow or live, not {MODE!r}")
    if MODE == "live" and os.environ.get("BOT_CONFIRM_LIVE", "").lower() != "yes":
        raise SystemExit("BOT_MODE=live also needs BOT_CONFIRM_LIVE=yes (real orders on the Wallex account)")
    seed()
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    from deploy.tunnel import Tunnel

    Tunnel(PERSIST, STATUS).start()  # before the bot: engine.net and httpx read the proxy settings at use
    log.info("status page on :%d, mode %s, agent %s, persistent dir %s", PORT, MODE, AGENT, PERSIST)

    import cli

    while True:
        run = latest_run()
        if finished(run) or (challenge_over() and (run is None or not json.loads(
                (run / "state.json").read_text() if (run / "state.json").exists() else "{}").get("positions"))):
            STATUS["bot"] = "idle: challenge window over" if challenge_over() else "idle: run finished"
            log.info(STATUS["bot"])
            while True:
                time.sleep(3600)
        resume = run  # the latest run of this mode (the runner restores state.json when it exists)
        STATUS["bot"] = f"running ({'resumed ' + resume.name if resume else 'new run'})"
        log.info("starting bot: %s", " ".join(bot_args(resume)))
        try:
            cli.main(bot_args(resume))
            if latest_run():
                (latest_run() / "FINISHED").write_text(datetime.now(timezone.utc).isoformat())
            STATUS["bot"] = "finished"
            log.info("bot finished normally")
        except SystemExit as e:
            if e.code not in (0, None):
                STATUS.update(last_error=f"exit {e.code}", restarts=STATUS["restarts"] + 1)
                log.error("bot exited with %s: restarting in 30 s", e.code)
                time.sleep(30)
        except Exception as e:  # noqa: BLE001 - a crash must never leave the account unattended
            STATUS.update(last_error=repr(e)[:300], restarts=STATUS["restarts"] + 1)
            log.exception("bot crashed: restarting in 30 s with --resume")
            time.sleep(30)


if __name__ == "__main__":
    main()
