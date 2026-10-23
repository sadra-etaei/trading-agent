"""What would have happened without the AI filters: replays every vetoed or gate-skipped entry of a run on
1-minute candles with the model's own take-profit / stop / time stop (taker fees both sides).

Evidence for the judges that the GPT veto and the event gate help (or not): a positive "avoided" sum means
the skipped trades would have lost money.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from engine.config import slippage
from engine.data.store import load_1m


def _entries(run_dir: Path) -> list[dict]:
    rows = [json.loads(line) for line in (run_dir / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
    signals = {r["decision_id"]: r for r in rows if r["kind"] == "entry_signal"}
    out = []
    for r in rows:
        if r["kind"] == "veto":
            out.append({"filter": "agent_veto", "reason": (r.get("agent") or {}).get("reason_code"), **_pick(r)})
        elif r["kind"] == "event_gate_skip":
            for did in r.get("entry_decision_ids", []):
                if did in signals:
                    out.append({"filter": "event_gate", "reason": r.get("reason", "")[:80], **_pick(signals[did])})
    return out


def _pick(r: dict) -> dict:
    return {"ts": r["ts"], "symbol": r["symbol"], "tp": r["ml"]["tp_pct"], "sl": r["ml"]["sl_pct"]}


def replay(data_dir: str, e: dict, horizon_h: float, cost: float) -> dict:
    m = load_1m(data_dir, e["symbol"])
    t0 = int(pd.Timestamp(e["ts"]).timestamp()) + 60  # one minute after the bar close
    m = m[(m["ts"] >= t0) & (m["ts"] < t0 + horizon_h * 3600)]
    if m.empty:
        return {**e, "outcome": "no data", "ret_pct": None}
    entry = float(m["close"].iloc[0])
    tp, sl = entry * (1 + e["tp"]), entry * (1 - e["sl"])
    for _, x in m.iloc[1:].iterrows():
        if x["low"] <= sl:  # stop first (conservative)
            return {**e, "outcome": "stop_loss", "ret_pct": round(100 * (-e["sl"] - cost), 3)}
        if x["high"] >= tp:
            return {**e, "outcome": "take_profit", "ret_pct": round(100 * (e["tp"] - cost), 3)}
    if m["ts"].iloc[-1] < t0 + horizon_h * 3600 - 120:
        return {**e, "outcome": "still open", "ret_pct": round(100 * (float(m["close"].iloc[-1]) / entry - 1 - cost), 3)}
    return {**e, "outcome": "time_stop", "ret_pct": round(100 * (float(m["close"].iloc[-1]) / entry - 1 - cost), 3)}


def report(cfg, run_dir: str) -> pd.DataFrame:
    run = Path(run_dir)
    horizon = int(cfg.labels.horizon_bars) * {"30m": 0.5, "1h": 1.0, "4h": 4.0}.get(cfg.data.bar, 1.0)
    rows = [replay(cfg.data.dir, e, horizon, 2 * cfg.costs.fee_rate + 2 * slippage(cfg, e["symbol"])) for e in _entries(run)]
    df = pd.DataFrame(rows)
    if df.empty:
        print("no vetoed or gate-skipped entries in", run_dir)
        return df
    print(df.to_string(index=False))
    done = df.dropna(subset=["ret_pct"])
    for f, g in done.groupby("filter"):
        print(f"{f}: {len(g)} skipped trades, would-have-been sum {g['ret_pct'].sum():+.2f}% "
              f"(positive = the filter cost us profit, negative = it avoided losses)")
    df.to_csv(run / "vetoes_replay.csv", index=False)
    return df
