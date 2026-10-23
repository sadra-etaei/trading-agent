"""Step 1: fair-value gap research on recorded bid/ask (PLAN.md §14.5).

python cli.py fv-research  ->  runs/fv-research-<time>/report.md + stats.json

Answers, per pair:
  - how big are gaps on real bid/ask, and how often does an entry condition (gap_bid <= -theta) start?
  - when a gap closes, is it the TMN price moving to fair value (good for us) or fair value moving
    to the TMN price (adverse: the TMN book was informed)?
  - lead-lag: does fair value move first (correlation of TMN moves with earlier F moves)?
  - are /v1/markets bid/ask snapshots consistent with the recorded order books?
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from engine.fv.data import bases_available, epoch_s, fair_value_frame, live_days, load_tob

THETAS = [0.002, 0.003, 0.0045, 0.006, 0.008]
HORIZONS_MIN = [5, 15, 30, 60]
GRID = "5s"


def entry_events(d: pd.DataFrame, theta: float, cooldown: str = "2min") -> pd.DatetimeIndex:
    """Times when gap_bid first crosses below -theta (with fresh inputs), at most one per cooldown."""
    ok = (d["gap_bid"] <= -theta) & ~d["stale_usd"] & ~d["stale_usdt"]
    starts = d.index[ok & ~ok.shift(1, fill_value=False)]
    out, last = [], None
    for t in starts:
        if last is None or t - last >= pd.Timedelta(cooldown):
            out.append(t)
            last = t
    return pd.DatetimeIndex(out)


def event_outcomes(d: pd.DataFrame, events: pd.DatetimeIndex, horizon_min: int) -> dict:
    """Closure = gap_mid back to >= 0 within the horizon. Decomposes the move between event and
    closure (or horizon end) into TMN-mid change vs fair-value change."""
    if len(events) == 0:
        return {"n": 0}
    closed, tmn_share, ret_tmn, ret_f = 0, [], [], []
    lm, lf = np.log(d["mid"]), np.log(d["F"])
    for t in events:
        w = d.loc[t: t + pd.Timedelta(minutes=horizon_min)]
        if len(w) < 2:
            continue
        hit = w.index[w["gap_mid"] >= 0]
        t_end = hit[0] if len(hit) else w.index[-1]
        closed += bool(len(hit))
        dm, df_ = lm[t_end] - lm[t], lf[t_end] - lf[t]
        ret_tmn.append(dm)
        ret_f.append(df_)
        tot = abs(dm) + abs(df_)
        if len(hit) and tot > 0:
            tmn_share.append(dm / (dm - df_) if dm - df_ != 0 else np.nan)
    n = len(ret_tmn)
    return {
        "n": n,
        "closed_pct": 100 * closed / n if n else 0.0,
        "tmn_moved_share_median": float(np.nanmedian(tmn_share)) if tmn_share else None,
        "tmn_ret_median_pct": 100 * float(np.median(ret_tmn)) if n else None,
        "fair_ret_median_pct": 100 * float(np.median(ret_f)) if n else None,
    }


def lead_lag(d: pd.DataFrame, max_lag: int = 12) -> dict:
    """corr(dlog TMN mid at t+k, dlog F at t) on a 5s grid. Peak at k>0 => fair value leads."""
    g = d[["mid", "F"]].resample(GRID).last().ffill()
    rm, rf = np.log(g["mid"]).diff(), np.log(g["F"]).diff()
    out = {}
    for k in range(-max_lag, max_lag + 1):
        c = rm.shift(-k).corr(rf)
        out[k] = None if pd.isna(c) else round(float(c), 4)
    return out


def tob_vs_depth(data_dir: str, symbol: str, tob: pd.DataFrame) -> dict | None:
    """Share of order-book snapshots whose best bid/ask equals the nearest /v1/markets snapshot."""
    rows = []
    for day in live_days(data_dir):
        p = day / "depth" / f"{symbol}.jsonl"
        if p.exists():
            for line in open(p, encoding="utf-8"):
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                f = r.get("features") or {}
                if f:
                    rows.append((r["ts"], f["best_bid"], f["best_ask"]))  # legacy depth/ files only
    if not rows or symbol not in tob.columns.get_level_values(0):
        return None
    dep = pd.DataFrame(rows, columns=["ts", "bid", "ask"])
    dep.index = pd.to_datetime(dep["ts"], unit="s", utc=True)
    t = tob[symbol][["bid", "ask"]].dropna()
    m = pd.merge_asof(dep.sort_index(), t.sort_index(), left_index=True, right_index=True,
                      suffixes=("_book", "_tob"), direction="nearest", tolerance=pd.Timedelta("10s")).dropna()
    if m.empty:
        return None
    same = ((m["bid_book"] == m["bid_tob"]) & (m["ask_book"] == m["ask_tob"])).mean()
    dev = ((m["bid_tob"] / m["bid_book"] - 1).abs().median())
    return {"n": int(len(m)), "exact_match_pct": round(100 * float(same), 1), "median_bid_dev_pct": round(100 * float(dev), 4)}


def run_research(cfg) -> Path:
    tob = load_tob(cfg.data.dir)
    if tob.empty:
        raise SystemExit("no recordings in data/live — run `python cli.py record` first")
    hours = (tob.index.max() - tob.index.min()).total_seconds() / 3600
    res = {"recorded_from": tob.index.min().isoformat(), "recorded_to": tob.index.max().isoformat(),
           "hours": round(hours, 2), "snapshots": int(len(tob)),
           "median_snapshot_gap_s": float(np.median(np.diff(epoch_s(tob.index)))), "pairs": {}}
    for base in bases_available(tob):
        d = fair_value_frame(tob, base)
        if d.empty:
            continue
        fresh = d[~d["stale_usd"] & ~d["stale_usdt"]]
        r = {
            "rows": int(len(d)),
            "fresh_pct": round(100 * len(fresh) / len(d), 1),
            "offset_pct": round(100 * float(d["offset"].median()), 4),
            "spread_median_pct": round(100 * float(d["spread"].median()), 3),
            "usd_spread_median_pct": round(100 * float(d["usd_spread"].median()), 3),
            "gap_bid_pct": {q: round(100 * float(fresh["gap_bid"].quantile(q)), 3) for q in (0.01, 0.05, 0.1, 0.5)},
            "gap_mid_abs_p50_pct": round(100 * float(fresh["gap_mid"].abs().median()), 3),
            "events": {},
            "lead_lag": lead_lag(d),
            "tob_vs_depth": tob_vs_depth(cfg.data.dir, f"{base}TMN", tob),
        }
        for th in THETAS:
            ev = entry_events(d, th)
            r["events"][f"{th:.4f}"] = {"per_hour": round(len(ev) / max(hours, 1e-9), 2),
                                        **{f"{h}m": event_outcomes(d, ev, h) for h in HORIZONS_MIN}}
        res["pairs"][base] = r

    run_dir = Path(cfg.run.output_dir) / ("fv-research-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "stats.json").write_text(json.dumps(res, indent=2, default=str))
    md = report_md(res)
    (run_dir / "report.md").write_text(md, encoding="utf-8")
    print(md)
    return run_dir


def report_md(res: dict) -> str:
    L = ["# Fair-value gap research (recorded bid/ask)", "",
         f"Recorded {res['recorded_from'][:19]} .. {res['recorded_to'][:19]} UTC — {res['hours']} h, "
         f"{res['snapshots']} snapshots, median {res['median_snapshot_gap_s']:.1f}s apart.", "",
         "gap_bid = log(TMN best bid / F) − offset, F = mid(XUSDT) × mid(USDTTMN). Negative = TMN cheap.", "",
         "## Gap size", "",
         "| pair | fresh % | offset | TMN spread | USDT-pair spread | gap_bid p1 | p5 | p10 | median | |gap_mid| median |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for b, r in res["pairs"].items():
        g = r["gap_bid_pct"]
        L.append(f"| {b} | {r['fresh_pct']} | {r['offset_pct']:+.3f}% | {r['spread_median_pct']:.3f}% | {r['usd_spread_median_pct']:.3f}% | "
                 f"{g[0.01]:+.3f}% | {g[0.05]:+.3f}% | {g[0.1]:+.3f}% | {g[0.5]:+.3f}% | {r['gap_mid_abs_p50_pct']:.3f}% |")
    L += ["", "## Entry events and how they close (horizon 30 min)", "",
          "closed = gap_mid back to ≥ 0. TMN share = fraction of the closing move done by the TMN price "
          "(1.0 = TMN moved to fair value, good for us; 0 = fair value moved to TMN, adverse).", "",
          "| pair | θ | events / h | closed % | TMN share (median) | TMN move | F move |", "|---|---|---|---|---|---|---|"]
    for b, r in res["pairs"].items():
        for th, e in r["events"].items():
            o = e["30m"]
            if not o.get("n"):
                L.append(f"| {b} | {float(th):.2%} | {e['per_hour']} | — | — | — | — |")
                continue
            sh = o["tmn_moved_share_median"]
            L.append(f"| {b} | {float(th):.2%} | {e['per_hour']} | {o['closed_pct']:.0f}% | "
                     f"{'—' if sh is None else f'{sh:.2f}'} | {o['tmn_ret_median_pct']:+.3f}% | {o['fair_ret_median_pct']:+.3f}% |")
    L += ["", "## Lead-lag", "", "corr(TMN mid return at t+k, fair-value return at t), 5s steps. "
          "Peak at k > 0 means fair value moves first (the TMN book follows).", "",
          "| pair | k=−2 | k=−1 | k=0 | k=+1 | k=+2 | k=+3 | k=+6 | peak k |", "|---|---|---|---|---|---|---|---|---|"]
    for b, r in res["pairs"].items():
        ll = {int(k): v for k, v in r["lead_lag"].items()}
        valid = {k: v for k, v in ll.items() if v is not None}
        peak = max(valid, key=lambda k: valid[k]) if valid else None
        f = lambda k: "—" if ll.get(k) is None else f"{ll[k]:+.3f}"
        L.append(f"| {b} | {f(-2)} | {f(-1)} | {f(0)} | {f(1)} | {f(2)} | {f(3)} | {f(6)} | {peak} |")
    L += ["", "## Snapshot quality", "", "| pair | book snapshots | /v1/markets matches book exactly | median bid deviation |", "|---|---|---|---|"]
    for b, r in res["pairs"].items():
        q = r["tob_vs_depth"]
        L.append(f"| {b} | — | — | — |" if not q else f"| {b} | {q['n']} | {q['exact_match_pct']}% | {q['median_bid_dev_pct']}% |")
    return "\n".join(L) + "\n"
