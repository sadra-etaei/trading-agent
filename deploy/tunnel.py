"""Userspace WireGuard tunnel for Binance (Liara cannot reach Binance from Iran: "Network is unreachable").

wireproxy (https://github.com/windtf/wireproxy, pinned in the Dockerfile) connects to a WireGuard server and offers
an HTTP proxy on 127.0.0.1:25345. It needs no root, no NET_ADMIN and no TUN device, so it runs in a PaaS container.
The bot's HTTP client (httpx) then uses HTTPS_PROXY for everything except NO_PROXY hosts: Wallex and the GPT gateway
stay on the direct Iranian connection, Binance and the news feeds go through the tunnel.

The WireGuard config (a normal wg-quick .conf file) comes from, in this order:
  WG_CONFIG_B64   the file encoded as one base64 line (easiest to paste into Liara's env vars)
  WG_CONFIG       the file's text itself
  WG_CONFIG_FILE  a path, default $BOT_PERSIST/wg.conf (a file put on the persistent disk)
No config -> no tunnel (the bot runs as before).
"""
from __future__ import annotations

import base64
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

log = logging.getLogger("tunnel")
PROXY = "127.0.0.1:25345"
DIRECT = ["api.wallex.ir", "ai.boxapi.ir", "localhost", "127.0.0.1"]
# wg-quick keys that only make sense for a kernel interface; wireproxy does not need them
SKIP_KEYS = ("postup", "postdown", "preup", "predown", "table", "saveconfig", "fwmark")


def config_text(persist: Path) -> str | None:
    if os.environ.get("WG_CONFIG_B64"):
        return base64.b64decode(os.environ["WG_CONFIG_B64"].strip()).decode("utf-8-sig")
    if os.environ.get("WG_CONFIG"):
        return os.environ["WG_CONFIG"].replace("\\n", "\n")
    p = Path(os.environ.get("WG_CONFIG_FILE", persist / "wg.conf"))
    return p.read_text(encoding="utf-8-sig") if p.exists() else None


def clean(conf: str) -> str:
    out = []
    for line in conf.splitlines():
        key = line.split("=", 1)[0].strip().lower()
        if "=" in line and key in SKIP_KEYS:
            continue
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


def _port_open(addr: str) -> bool:
    host, port = addr.split(":")
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


class Tunnel:
    def __init__(self, persist: Path, status: dict):
        self.persist, self.status = persist, status
        self.dir = Path(os.environ.get("WG_RUNTIME_DIR", "/tmp/wg"))
        self.proc: subprocess.Popen | None = None

    def start(self) -> bool:
        conf = config_text(self.persist)
        if not conf:
            self.status["tunnel"] = "off (no WireGuard config)"
            return False
        exe = shutil.which("wireproxy")
        if not exe:
            self.status["tunnel"] = "error: wireproxy binary not found"
            log.error(self.status["tunnel"])
            return False
        self.dir.mkdir(parents=True, exist_ok=True)
        wg = self.dir / "wg.conf"
        wg.write_text(clean(conf), encoding="utf-8")
        os.chmod(wg, 0o600)
        (self.dir / "wireproxy.conf").write_text(f"WGConfig = {wg}\n\n[http]\nBindAddress = {PROXY}\n", encoding="utf-8")
        self.exe = exe
        self._launch()
        threading.Thread(target=self._watch, daemon=True).start()
        for _ in range(30):
            if _port_open(PROXY):
                break
            time.sleep(1)
        os.environ.setdefault("HTTPS_PROXY", f"http://{PROXY}")
        os.environ.setdefault("HTTP_PROXY", f"http://{PROXY}")
        no = [h for h in os.environ.get("NO_PROXY", "").split(",") if h.strip()]
        from urllib.parse import urlparse

        gw = urlparse(os.environ.get("OPENAI_BASE_URL", "")).hostname  # the GPT gateway is Iranian: keep it direct
        os.environ["NO_PROXY"] = ",".join(dict.fromkeys(no + DIRECT + ([gw] if gw else [])))
        ok = self.check()
        self.status["tunnel"] = "up" if ok else "started, but Binance not reachable yet"
        log.info("tunnel: %s (proxy %s, direct: %s)", self.status["tunnel"], PROXY, os.environ["NO_PROXY"])
        return ok

    def _launch(self) -> None:
        self.proc = subprocess.Popen([self.exe, "-c", str(self.dir / "wireproxy.conf")], stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
        threading.Thread(target=self._pipe, args=(self.proc,), daemon=True).start()

    @staticmethod
    def _pipe(proc: subprocess.Popen) -> None:
        for line in proc.stdout:  # wireproxy's own log, prefixed so it is easy to find in Liara's log view
            log.info("wireproxy: %s", line.rstrip())

    def _watch(self) -> None:
        """Restart wireproxy if it exits; re-check Binance every 10 minutes for the status page."""
        last = time.time()
        while True:
            time.sleep(15)
            if self.proc and self.proc.poll() is not None:
                log.warning("wireproxy exited with %s -> restarting", self.proc.returncode)
                self.status["tunnel"] = "restarting"
                self._launch()
            if time.time() - last > 600:
                last = time.time()
                self.status["tunnel"] = "up" if self.check() else "down: Binance not reachable through the tunnel"

    def check(self) -> bool:
        import httpx

        from engine.net import binance_spot

        for _ in range(3):
            try:
                r = httpx.get(binance_spot() + "/api/v3/ping", timeout=10)
                if r.status_code == 200:
                    self.status["tunnel_checked"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    return True
                log.warning("tunnel check: Binance answered HTTP %s", r.status_code)
            except httpx.HTTPError as e:
                log.warning("tunnel check failed: %s", e)
            time.sleep(3)
        return False
