"""Append-only decision log (JSONL). One record per decision: scans, entries, vetoes, exits,
risk events. This is a challenge deliverable: it proves every trade came from the algorithm."""
from __future__ import annotations

import json
import math
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def _clean(v):
    if isinstance(v, dict):
        return {str(k): _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else round(f, 8)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.isoformat()
    return v


class DecisionLog:
    def __init__(self, run_dir: str | Path, run_id: str, mode: str):
        self.path = Path(run_dir) / "decisions.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.mode = mode
        self._n = 0
        self._lock = threading.Lock()

    def new_id(self) -> str:
        with self._lock:
            self._n += 1
            return f"{self.run_id}-{self._n:06d}"

    def write(self, kind: str, ts, **fields) -> str:
        did = fields.pop("decision_id", None) or self.new_id()
        rec = {
            "decision_id": did,
            "kind": kind,
            "ts": ts,
            "mode": self.mode,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_clean(rec), ensure_ascii=False) + "\n")
        return did
