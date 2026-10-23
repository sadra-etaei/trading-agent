"""Compare model/label variants on walk-forward + validation only (never the held-out test).

python cli.py experiments   -> models/experiments-<time>.md
"""
from __future__ import annotations

import copy
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from engine.config import _merge, _to_cfg
from engine.model.train import train_and_evaluate

log = logging.getLogger(__name__)

WIDE = {"labels": {"tp_atr": 2.5, "sl_atr": 1.25, "horizon_bars": 24, "min_tp_pct": 0.02}}
CORE = ["BTCTMN", "ETHTMN", "SOLTMN", "XRPTMN", "BNBTMN", "DOGETMN", "TRXTMN", "USDTTMN"]
EXTRA = ["PEPETMN", "SUITMN", "LTCTMN", "ADATMN", "LINKTMN", "AVAXTMN", "SHIBTMN", "UNITMN"]
BAR15 = {"data": {"bar": "15m"}, "labels": {"horizon_bars": 48}}  # same 12h horizon on 15-minute bars
NO_CAL = {"model": {"exclude_features": ["fng", "fng_chg_7", "dow", "hour_sin", "hour_cos"]}}
VARIANTS = {
    "base": {},
    "base_no_calendar_fng": NO_CAL,
    "base_premium": {"universe": {"usdt_premium": True}},
    "base_logreg": {"model": {"kind": "logreg"}},
    "logreg_premium": {"model": {"kind": "logreg"}, "universe": {"usdt_premium": True}},
    "logreg_no_calendar_fng": _merge(NO_CAL, {"model": {"kind": "logreg"}}),
    # round 2 (relative to the chosen config: logreg without calendar / F&G features)
    "chosen": {},
    "more_pairs": {"universe": {"core": CORE + EXTRA}},
    "ensemble": {"model": {"kind": "ensemble"}},
    "more_pairs_ensemble": {"universe": {"core": CORE + EXTRA}, "model": {"kind": "ensemble"}},
    "bar15m": BAR15,
    # round 3: outside data (Binance spot 1h + futures funding)
    "outside": {"outside": {"binance": True}, "sentiment": {"funding": True}},
    "outside_no_funding": {"outside": {"binance": True}},
    "funding_only": {"sentiment": {"funding": True}},
    "bar15m_more_pairs": _merge(BAR15, {"universe": {"core": CORE + EXTRA}}),
    "wide": WIDE,
    "wide_no_calendar_fng": _merge(WIDE, {"model": {"exclude_features": ["fng", "fng_chg_7", "dow", "hour_sin", "hour_cos"]}}),
    "wide_premium": _merge(WIDE, {"universe": {"usdt_premium": True}}),
    "wide_logreg": _merge(WIDE, {"model": {"kind": "logreg"}}),
}


def summarize(rep: dict) -> dict:
    folds = rep["walk_forward"]
    m = [f["model"]["ev>0.002"] for f in folds]
    n = sum(x["n"] for x in m)
    tot = sum(x["sum_net"] for x in m)
    v = rep["validation"]
    chosen = next((g for g in v["ev_threshold_grid"] if g["ev_thr"] == v["chosen_ev_threshold"]), {})
    vg = next((g for g in v.get("val_only_grid", []) if g["ev_thr"] == v["chosen_ev_threshold"]), {})
    return {
        "oof_trades": chosen.get("n", 0),
        "oof_mean_net": chosen.get("mean_net", 0.0),
        "ev_thr": v["chosen_ev_threshold"],
        "auc_mean": float(np.nanmean([f["auc"] for f in folds])) if folds else float("nan"),
        "wf_trades": n,
        "wf_mean_net": tot / n if n else 0.0,
        "wf_sum_net": tot,
        "wf_positive_folds": sum(1 for x in m if x["n"] and x["mean_net"] > 0),
        "folds": len(folds),
        "rules_mean_net": float(np.mean([f["baseline"]["rules"]["mean_net"] for f in folds])) if folds else 0.0,
        "val_auc": v["auc_raw"],
        "val_trades": vg.get("n", 0),
        "val_mean_net": vg.get("mean_net", 0.0),
        "version": rep["version"],
    }


def run_experiments(cfg, names: list[str] | None = None) -> Path:
    rows = {}
    for name, ov in VARIANTS.items():
        if names and name not in names:
            continue
        ov = _merge(ov, {"model": {"dir": str(Path(cfg.model.dir) / "experiments" / name)}})
        c = _to_cfg(_merge(copy.deepcopy(dict(cfg)), ov))
        log.info("=== variant %s", name)
        try:
            rows[name] = summarize(train_and_evaluate(c))
        except Exception as e:  # keep going; report the failure
            log.exception("variant %s failed", name)
            rows[name] = {"error": str(e)}
    L = ["# Model experiments (walk-forward + validation; held-out test untouched)", "",
         "| variant | WF AUC | WF trades (EV>0.2%) | WF mean net/trade | positive folds | rules mean net | tuned EV thr | OOF+val trades | OOF+val mean net | val AUC | val trades | val mean net |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in rows.items():
        if "error" in r:
            L.append(f"| {k} | error: {r['error'][:60]} |")
            continue
        L.append(f"| {k} | {r['auc_mean']:.3f} | {r['wf_trades']} | {r['wf_mean_net']:+.3%} | "
                 f"{r['wf_positive_folds']}/{r['folds']} | {r['rules_mean_net']:+.3%} | {r['ev_thr']:.2%} | {r['oof_trades']} | "
                 f"{r['oof_mean_net']:+.3%} | {r['val_auc']:.3f} | {r['val_trades']} | {r['val_mean_net']:+.3%} |")
    L += ["", "Variants:", ""] + [f"- **{k}**: `{VARIANTS[k]}`" for k in rows]
    out = Path(cfg.model.dir) / f"experiments-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return out
