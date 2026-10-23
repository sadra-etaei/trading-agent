"""Writes deploy/seed/: the last 50 days of 1-minute candles for the live universe + context pairs, market rules
and the Fear & Greed history, so a fresh container can decide at its first hourly close without a long download.
Run before building the image: python deploy/make_seed.py"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.config import load_config, universe  # noqa: E402
from engine.data import store  # noqa: E402

cfg = load_config(ROOT / "config.yaml")
out = ROOT / "deploy" / "seed"
shutil.rmtree(out, ignore_errors=True)
syms = sorted(set(universe(cfg)) | set(cfg.universe.context_pairs))
cut = int(time.time()) - 50 * 86400
for s in syms:
    df = store.load_1m(ROOT / cfg.data.dir, s)
    store.save_1m(out, s, df[df["ts"] >= cut])
for rel in ("markets.json", "sentiment/fear_greed.parquet"):
    src = ROOT / cfg.data.dir / rel
    if src.exists():
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / rel)
print(f"seed: {len(syms)} pairs -> {out} ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.1f} MB)")
