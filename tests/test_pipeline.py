"""End-to-end on synthetic data: store -> features -> labels -> train -> backtest -> report."""
import json

import numpy as np
import pandas as pd

from engine.config import load_config
from engine.data import store

DAYS = 70


def synth_1m(seed: int, start_price: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = DAYS * 1440
    t0 = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    # regime-switching drift so there is something to learn
    drift = np.repeat(rng.choice([-1, 0, 1], size=n // 2880 + 1) * vol * 0.05, 2880)[:n]
    r = drift + rng.normal(0, vol, n)
    c = start_price * np.exp(np.cumsum(r))
    o = np.r_[c[0], c[:-1]]
    spread = np.abs(rng.normal(0, vol / 2, n)) * c
    return pd.DataFrame({
        "ts": t0 + 60 * np.arange(n), "open": o, "high": np.maximum(o, c) + spread,
        "low": np.minimum(o, c) - spread, "close": c, "volume": rng.exponential(1.0, n),
    })


def test_end_to_end(tmp_path):
    syms = ["BTCTMN", "ETHTMN", "USDTTMN"]
    for i, (s, p, v) in enumerate(zip(syms, [2e10, 6e8, 2.6e5], [0.0015, 0.0018, 0.0003])):
        store.save_1m(tmp_path / "data", s, synth_1m(i, p, v))
    (tmp_path / "data" / "markets.json").write_text(json.dumps(
        {s: {"min_notional": 50000, "qty_dec": 8, "price_dec": 0} for s in syms}))
    cfg = load_config("config.yaml", {
        "data": {"dir": str(tmp_path / "data")},
        "model": {"dir": str(tmp_path / "models"), "val_days": 10, "test_days": 10, "walk_forward_folds": 2,
                  "lgbm": {"n_estimators": 50, "min_child_samples": 50}},
        "run": {"output_dir": str(tmp_path / "runs")},
        "universe": {"core": syms},
        "sentiment": {"fear_greed": False, "funding": False},
    })
    from engine.backtest.runner import run_backtest
    from engine.model.train import train_and_evaluate

    rep = train_and_evaluate(cfg)
    assert rep["train_rows"] > 1000
    assert (tmp_path / "models" / "latest.joblib").exists()

    run_dir = run_backtest(cfg)
    for f in ("meta.json", "decisions.jsonl", "equity.csv", "report.md", "metrics.json", "challenge_windows.csv"):
        assert (run_dir / f).exists(), f
    kinds = {json.loads(line)["kind"] for line in open(run_dir / "decisions.jsonl", encoding="utf-8")}
    assert "scan" in kinds
    eq = pd.read_csv(run_dir / "equity.csv")
    assert eq["equity"].gt(0).all()
    wins = pd.read_csv(run_dir / "challenge_windows.csv")
    assert (wins["open_positions_at_end"] == 0).all()  # always flat in TMN at challenge end

    from engine.model.selection import run_experiment

    exp = run_experiment(cfg)
    res = json.loads((exp / "result.json").read_text())
    assert set(res["results"]) == {"ev_ranking", "two_stage"}
