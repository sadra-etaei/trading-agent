"""Minimal .env loader (no extra dependency). Real environment variables always win; empty values are skipped."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Loads NAME=value lines into os.environ. Returns the names that were set (never the values)."""
    p = Path(path)
    if not p.exists():
        return []
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip().removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if not value or name in os.environ:
            continue
        os.environ[name] = value
        loaded.append(name)
    return loaded


def apply_env_overrides(cfg) -> None:
    """Optional settings from .env that override config.yaml."""
    for env, key in (("OPENAI_BASE_URL", "base_url"), ("OPENAI_SCREEN_MODEL", "screen_model"),
                     ("OPENAI_DECISION_MODEL", "decision_model")):
        if os.environ.get(env):
            cfg["agent"][key] = os.environ[env]
