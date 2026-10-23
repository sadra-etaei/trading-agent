"""Walk-forward evaluation and final model training.

Outputs (models/):
  entry-<version>.joblib     model bundle (classifier, calibrator, features, thresholds, splits)
  latest.joblib              copy of the newest bundle
  train-report-<version>.json/.md  walk-forward metrics vs. rule baseline
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from engine.config import config_hash
from engine.features.build import feature_list
from engine.model.dataset import build_table, make_splits, net_return
from engine.model.predict import EntryModel, expected_value

log = logging.getLogger(__name__)

EV_GRID = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.01]
MIN_TRADES_FOR_THRESHOLD = 40


def _logreg():
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=2000)),
        ]
    )


def make_model(cfg):
    if cfg.model.kind == "logreg":
        return _logreg()
    import lightgbm as lgb

    gbm = lgb.LGBMClassifier(**dict(cfg.model.lgbm), random_state=cfg.run.seed, verbose=-1)
    if cfg.model.kind == "ensemble":  # average of the linear and the tree model probabilities
        from sklearn.ensemble import VotingClassifier

        return VotingClassifier([("lr", _logreg()), ("gbm", gbm)], voting="soft")
    return gbm


def _cost(cfg) -> float:
    return 2 * cfg.costs.fee_rate + 2 * cfg.costs.slippage_bps / 1e4


def evaluate_selection(df: pd.DataFrame, p: np.ndarray, cfg, ev_thr: float) -> dict:
    ev = expected_value(p, df["tp_pct"].to_numpy(), df["sl_pct"].to_numpy(), _cost(cfg))
    sel = ev > ev_thr
    net = net_return(df["outcome"].to_numpy(), cfg)
    return {
        "n": int(sel.sum()),
        "mean_net": float(net[sel].mean()) if sel.any() else 0.0,
        "sum_net": float(net[sel].sum()) if sel.any() else 0.0,
        "win_rate": float(df["label"].to_numpy()[sel].mean()) if sel.any() else 0.0,
    }


def baselines(df: pd.DataFrame, cfg) -> dict:
    net = net_return(df["outcome"].to_numpy(), cfg)
    rule = ((df["sig_pullback"] == 1) | (df["sig_meanrev"] == 1)).to_numpy()
    return {
        "all_bars": {"n": int(len(df)), "mean_net": float(net.mean()), "win_rate": float(df["label"].mean())},
        "rules": {
            "n": int(rule.sum()),
            "mean_net": float(net[rule].mean()) if rule.any() else 0.0,
            "sum_net": float(net[rule].sum()) if rule.any() else 0.0,
            "win_rate": float(df["label"].to_numpy()[rule].mean()) if rule.any() else 0.0,
        },
    }


def evaluate_trend_filters(cal: pd.DataFrame, p: np.ndarray, cfg, ev_thr: float) -> dict:
    """Out-of-sample effect of each market-trend rule on the model's selected trades.
    Pre-registered choice: highest total net among rules whose mean net per trade >= no filter
    and whose positive-fold count >= no filter; otherwise 'none'."""
    from engine.strategy import trend

    ev = expected_value(p, cal["tp_pct"].to_numpy(), cal["sl_pct"].to_numpy(), _cost(cfg))
    sel = ev > ev_thr
    net = net_return(cal["outcome"].to_numpy(), cfg)
    rows = {}
    for rule in trend.RULES:
        m = sel & trend.allowed(cal, rule).to_numpy()
        folds = pd.Series(net[m]).groupby(cal["fold"].to_numpy()[m]).sum() if m.any() else pd.Series(dtype=float)
        rows[rule] = {"n": int(m.sum()), "mean_net": float(net[m].mean()) if m.any() else 0.0,
                      "sum_net": float(net[m].sum()) if m.any() else 0.0,
                      "win_rate": float(cal["label"].to_numpy()[m].mean()) if m.any() else 0.0,
                      "positive_folds": int((folds > 0).sum()), "folds": int(cal["fold"].nunique()),
                      "worst_fold_sum": float(folds.min()) if len(folds) else 0.0,
                      "fold_sums": {k: round(float(v), 3) for k, v in folds.items()}}
    base = rows["none"]
    ok = {k: r for k, r in rows.items() if k != "none" and r["mean_net"] >= base["mean_net"]
          and r["positive_folds"] >= base["positive_folds"] and r["n"] >= 30}
    choice = max(ok, key=lambda k: ok[k]["sum_net"]) if ok else "none"
    if choice != "none" and ok[choice]["sum_net"] < base["sum_net"] and ok[choice]["worst_fold_sum"] <= base["worst_fold_sum"]:
        choice = "none"  # fewer trades for less total profit and no better worst case -> not worth it
    return {"rules": rows, "choice": choice}


def tune_threshold(df: pd.DataFrame, p: np.ndarray, cfg) -> tuple[float, list[dict]]:
    """Pick the EV threshold that maximises total net return on the validation window."""
    rows = []
    for thr in EV_GRID:
        r = evaluate_selection(df, p, cfg, thr)
        r["ev_thr"] = thr
        rows.append(r)
    ok = [r for r in rows if r["n"] >= MIN_TRADES_FOR_THRESHOLD and r["mean_net"] > 0]
    best = max(ok, key=lambda r: r["sum_net"]) if ok else max(rows, key=lambda r: r["mean_net"])
    return best["ev_thr"], rows


def walk_forward(table: pd.DataFrame, cfg, feats: list[str], end: pd.Timestamp) -> tuple[list[dict], pd.DataFrame]:
    """Expanding-window folds. Returns fold metrics and the out-of-fold predictions (raw scores)."""
    data = table[table.index < end]
    times = data.index.unique().sort_values()
    k = int(cfg.model.walk_forward_folds)
    purge = pd.Timedelta(hours=int(cfg.labels.horizon_bars) + 1)
    first_test = times[int(len(times) * 0.4)]  # at least 40% of history for the first fold
    edges = pd.date_range(first_test, end, periods=k + 1)
    folds, oof = [], []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        tr = data[data.index < lo - purge]
        te = data[(data.index >= lo) & (data.index < hi)]
        if len(tr) < 1000 or len(te) < 100:
            continue
        m = make_model(cfg)
        m.fit(tr[feats], tr["label"].astype(int))
        p = m.predict_proba(te[feats])[:, 1]
        res = {
            "fold": i + 1,
            "train_end": str(lo.date()),
            "test": f"{lo.date()} .. {hi.date()}",
            "n_test": int(len(te)),
            "base_rate": float(te["label"].mean()),
            "auc": float(roc_auc_score(te["label"], p)) if te["label"].nunique() > 1 else float("nan"),
        }
        # Uncalibrated probabilities here; thresholds expressed on raw scores are still comparable
        # across folds via the EV formula as long as the model is reasonably calibrated.
        res["model"] = {f"ev>{t}": evaluate_selection(te, p, cfg, t) for t in (0.0, 0.002, 0.004)}
        res["baseline"] = baselines(te, cfg)
        folds.append(res)
        oof.append(te.assign(raw_p=p, fold=str(i + 1)))
        log.info(
            "fold %d %s auc=%.3f model(ev>0.002) n=%d mean=%.4f | rules n=%d mean=%.4f",
            i + 1, res["test"], res["auc"], res["model"]["ev>0.002"]["n"], res["model"]["ev>0.002"]["mean_net"],
            res["baseline"]["rules"]["n"], res["baseline"]["rules"]["mean_net"],
        )
    return folds, (pd.concat(oof) if oof else table.iloc[:0].assign(raw_p=[]))


def train_and_evaluate(cfg) -> dict:
    t0 = datetime.now(timezone.utc)
    table = build_table(cfg)
    table = table[table["label"].notna()]
    feats = feature_list(bool(cfg.sentiment.funding), bool(cfg.universe.get("usdt_premium")),
                         cfg.model.get("exclude_features"), bool((cfg.get("outside") or {}).get("binance")),
                         bool(cfg.model.get("extra_features")))
    splits = make_splits(table.index, cfg)
    log.info("table: %d rows, %d pairs, %s .. %s", len(table), table["symbol"].nunique(), table.index.min(), table.index.max())
    log.info("splits: %s", splits.as_dict())

    folds, oof = walk_forward(table, cfg, feats, splits.val_end)

    train = table[table.index < splits.train_end]
    val = table[(table.index >= splits.val_start) & (table.index < splits.val_end)]
    model = make_model(cfg)
    model.fit(train[feats], train["label"].astype(int))
    raw_val = model.predict_proba(val[feats])[:, 1]
    # Calibrate and tune the EV threshold on all out-of-sample predictions: walk-forward folds
    # (out-of-fold) + the validation window. Far more rows than validation alone -> stabler.
    if cfg.model.get("calibration", "oof") == "oof" and len(oof):
        cal = pd.concat([oof, val.assign(raw_p=raw_val, fold="val")])
    else:
        cal = val.assign(raw_p=raw_val, fold="val")
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(cal["raw_p"], cal["label"].astype(int))
    ev_thr, grid = tune_threshold(cal, calibrator.predict(cal["raw_p"]), cfg)
    p_val = calibrator.predict(raw_val)
    _, val_grid = tune_threshold(val, p_val, cfg)
    trend_eval = evaluate_trend_filters(cal, calibrator.predict(cal["raw_p"]), cfg, ev_thr)

    version = t0.strftime("%Y%m%d-%H%M%S") + f"-{cfg.model.kind}"
    bundle = {
        "model": model,
        "calibrator": calibrator,
        "features": feats,
        "version": version,
        "thresholds": {"min_ev_pct": ev_thr},
        "splits": splits.as_dict(),
        "labels": dict(cfg.labels),
        "costs": dict(cfg.costs),
        "symbols": sorted(table["symbol"].unique().tolist()),
        "config_hash": config_hash(cfg),
        "trained_at": t0.isoformat(),
    }
    out_dir = Path(cfg.model.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"entry-{version}.joblib"
    joblib.dump(bundle, path)
    shutil.copyfile(path, out_dir / "latest.joblib")

    em = EntryModel(bundle)
    report = {
        "version": version,
        "rows": int(len(table)),
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "splits": splits.as_dict(),
        "walk_forward": folds,
        "validation": {
            "auc_raw": float(roc_auc_score(val["label"], raw_val)) if val["label"].nunique() > 1 else None,
            "base_rate": float(val["label"].mean()),
            "calibration_rows": int(len(cal)),
            "ev_threshold_grid": grid,
            "val_only_grid": val_grid,
            "chosen_ev_threshold": ev_thr,
            "baseline": baselines(val, cfg),
            "calibration": _calibration_table(p_val, val["label"].to_numpy()),
        },
        "top_features": em.top_features(15),
        "trend_filters": trend_eval,
    }
    (out_dir / f"train-report-{version}.json").write_text(json.dumps(report, indent=2, default=str))
    (out_dir / f"train-report-{version}.md").write_text(_report_md(report))
    log.info("saved %s (EV threshold %.4f)", path, ev_thr)
    print(_report_md(report))
    return report


def _calibration_table(p: np.ndarray, y: np.ndarray, bins: int = 8) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p <= hi)
        if m.sum():
            rows.append({"p_lo": float(lo), "p_hi": float(hi), "n": int(m.sum()), "pred": float(p[m].mean()), "actual": float(y[m].mean())})
    return rows


def _report_md(r: dict) -> str:
    L = [f"# Training report {r['version']}", ""]
    L.append(f"Rows: {r['rows']} (train {r['train_rows']}, validation {r['val_rows']})  ")
    s = r["splits"]
    L.append(f"Train < {s['train_end'][:10]} | validation {s['val_start'][:10]} .. {s['val_end'][:10]} | "
             f"**held-out test {s['test_start'][:10]} .. {s['data_end'][:10]}** (not used here)")
    L += ["", "## Walk-forward (out-of-sample folds)", "",
          "| fold | test window | AUC | base win rate | model trades (EV>0.2%) | model mean net | rules trades | rules mean net |",
          "|---|---|---|---|---|---|---|---|"]
    for f in r["walk_forward"]:
        m, b = f["model"]["ev>0.002"], f["baseline"]["rules"]
        L.append(f"| {f['fold']} | {f['test']} | {f['auc']:.3f} | {f['base_rate']:.1%} | {m['n']} | {m['mean_net']:+.3%} | {b['n']} | {b['mean_net']:+.3%} |")
    v = r["validation"]
    L += ["", "## Calibration + threshold (out-of-fold + validation)", "",
          f"Validation AUC (raw): {v['auc_raw']:.3f}, base win rate {v['base_rate']:.1%}, "
          f"calibration rows {v.get('calibration_rows', 0)}, chosen EV threshold **{v['chosen_ev_threshold']:.2%}**", "",
          "| EV > | trades (OOF+val) | win rate | mean net | sum net | val-only trades | val-only mean net |",
          "|---|---|---|---|---|---|---|"]
    for g, gv in zip(v["ev_threshold_grid"], v.get("val_only_grid", v["ev_threshold_grid"])):
        L.append(f"| {g['ev_thr']:.2%} | {g['n']} | {g['win_rate']:.1%} | {g['mean_net']:+.3%} | {g['sum_net']:+.2f} | "
                 f"{gv['n']} | {gv['mean_net']:+.3%} |")
    b = v["baseline"]["rules"]
    L += ["", f"Rule baseline on validation: {b['n']} trades, win {b['win_rate']:.1%}, mean net {b['mean_net']:+.3%}", "",
          "### Calibration", "", "| predicted | actual | n |", "|---|---|---|"]
    for c in v["calibration"]:
        L.append(f"| {c['pred']:.2f} | {c['actual']:.2f} | {c['n']} |")
    tf = r.get("trend_filters")
    if tf:
        L += ["", "## Market-trend filter (out-of-fold + validation)", "",
              "| rule | trades | win rate | mean net | sum net | positive folds | worst fold |", "|---|---|---|---|---|---|---|"]
        for k, x in tf["rules"].items():
            L.append(f"| {k} | {x['n']} | {x['win_rate']:.1%} | {x['mean_net']:+.3%} | {x['sum_net']:+.2f} | "
                     f"{x['positive_folds']}/{x['folds']} | {x['worst_fold_sum']:+.2f} |")
        L += ["", f"Pre-registered choice: **{tf['choice']}**"]
    L += ["", "### Top features", ""] + [f"- {n}: {w:.3g}" for n, w in r["top_features"]]
    return "\n".join(L) + "\n"
