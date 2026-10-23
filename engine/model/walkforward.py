"""12-month walk-forward test (PLAN.md §18): improved GRU (G) and the tree ensemble (T) on the D+ data.

Per test month M: fit on all rows before M minus the last 120 days; the GRU early-stops on days −120..−60;
isotonic calibration and the EV threshold use days −60..0 (out-of-sample predictions); then month M is traded
with the normal backtest. Nothing is tuned on any test month.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from engine.config import load_config
from engine.features.build import feature_list
from engine.model.dataset import build_table
from engine.model.train import make_model, tune_threshold

log = logging.getLogger(__name__)
PURGE = pd.Timedelta(hours=13)
GRU = {"window": 48, "hidden": 32, "dropout": 0.3, "epochs": 3, "seeds": (1, 2, 3)}


def months(first: str = "2025-11-01", last_end: str = "2026-10-22") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(first, last_end, freq="MS", tz="UTC")
    out = []
    for s in starts:
        e = min(s + pd.offsets.MonthBegin(1), pd.Timestamp(last_end, tz="UTC"))
        out.append((s, e))
    return out


def fit_gru(table, feats, fit_end, es_start, es_end, seed, dev):
    import torch

    from engine.model.seq import GRUNet, SeqModel
    from sklearn.metrics import roc_auc_score

    torch.manual_seed(seed)
    np.random.seed(seed)
    w = GRU["window"]
    fit_rows = table[table.index < fit_end]
    mean = np.nan_to_num(np.nanmean(fit_rows[feats].to_numpy(np.float64), axis=0)).astype(np.float32)
    std = np.nanstd(fit_rows[feats].to_numpy(np.float64), axis=0)
    std = np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)
    tmp = SeqModel(feats, mean, std, {}, GRU["hidden"], w, GRU["dropout"])
    arrays, tr, es = [], [], []
    for _, g in table.groupby("symbol"):
        g = g.sort_index()
        x, y, ts = tmp.scaled(g), g["label"].to_numpy(), g.index
        sid = len(arrays)
        arrays.append((x, y))
        for i in range(w - 1, len(g)):
            if not np.isfinite(y[i]):
                continue
            if ts[i] < fit_end:
                tr.append((sid, i))
            elif es_start <= ts[i] < es_end:
                es.append((sid, i))
    tr, es = np.array(tr), np.array(es)

    def batch(ix):
        xb = np.stack([arrays[a][0][i - w + 1:i + 1] for a, i in ix])
        yb = np.array([arrays[a][1][i] for a, i in ix], dtype=np.float32)
        return torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)

    net = GRUNet(len(feats), GRU["hidden"], GRU["dropout"]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    pos = float(np.mean([arrays[a][1][i] for a, i in tr]))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=dev))
    best, best_state = np.inf, None
    for ep in range(GRU["epochs"]):
        net.train()
        perm = np.random.permutation(len(tr))
        for k in range(0, len(perm), 1024):
            xb, yb = batch(tr[perm[k:k + 1024]])
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        lo, ys = [], []
        with torch.no_grad():
            for k in range(0, len(es), 4096):
                xb, yb = batch(es[k:k + 4096])
                lo.append(net(xb).cpu())
                ys.append(yb.cpu())
        lo, ys = torch.cat(lo), torch.cat(ys)
        es_loss = float(loss_fn(lo.to(dev), ys.to(dev)))
        log.info("  seed %d epoch %d: early-stop loss %.4f auc %.3f", seed, ep + 1, es_loss,
                 roc_auc_score(ys.numpy(), torch.sigmoid(lo).numpy()))
        if es_loss < best:
            best, best_state = es_loss, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    return best_state, mean, std


def run(train_overrides: dict, bt_overrides: dict, out_dir: str = "models/wf", kinds=("gru", "trees")) -> dict:
    import torch

    from engine.backtest.runner import run_backtest
    from engine.model.seq import SeqModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = load_config("config.yaml", train_overrides)
    table = build_table(cfg)
    lab = table[table["label"].notna()]
    feats = feature_list(bool(cfg.sentiment.funding), False, cfg.model.get("exclude_features"),
                         bool((cfg.get("outside") or {}).get("binance")), bool(cfg.model.get("extra_features")))
    bt_cfg = load_config("config.yaml", bt_overrides)
    results = {k: [] for k in kinds}
    for m_start, m_end in months():
        cut = m_start - PURGE
        cal_start = cut - pd.Timedelta(days=60)
        es_start = cal_start - pd.Timedelta(days=60)
        cal = lab[(lab.index >= cal_start) & (lab.index < cut)]
        for kind in kinds:
            if kind == "gru":
                states, mean, std = [], None, None
                for seed in GRU["seeds"]:
                    st, mean, std = fit_gru(table[table.index < cut], feats, es_start - PURGE, es_start, cal_start - PURGE, seed, dev)
                    states.append(st)
                model = SeqModel(feats, mean, std, states, GRU["hidden"], GRU["window"], GRU["dropout"])
                per = {s: model.proba_frame(g.sort_index()) for s, g in table[table.index < cut].groupby("symbol")}
                raw = np.concatenate([per[s].reindex(g.index).to_numpy() for s, g in cal.groupby("symbol")])
                cal_rows = pd.concat([g for _, g in cal.groupby("symbol")])
            else:
                model = make_model(cfg)
                fit = lab[lab.index < cal_start - PURGE]
                model.fit(fit[feats], fit["label"].astype(int))
                cal_rows = cal
                raw = model.predict_proba(cal[feats])[:, 1]
            ok = np.isfinite(raw)
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw[ok], cal_rows["label"].to_numpy()[ok].astype(int))
            thr, _ = tune_threshold(cal_rows[ok], iso.predict(raw[ok]), cfg)
            bundle = {"model": model, "calibrator": iso, "features": feats, "version": f"wf-{kind}-{m_start:%Y%m}",
                      "thresholds": {"min_ev_pct": thr}, "symbols": sorted(table["symbol"].unique().tolist()),
                      "splits": {"val_end": str(cut), "test_start": str(m_start), "data_end": str(m_end)}}
            if kind == "gru":
                bundle["sequence"] = GRU["window"]
            path = Path(out_dir) / kind / f"{m_start:%Y%m}.joblib"
            path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(bundle, path)
            run_dir = run_backtest(bt_cfg, str(m_start), str(m_end - pd.Timedelta(hours=1)), str(path),
                                   tag=f"wf-{kind}-{m_start:%Y%m}", windows=False)
            met = json.loads((Path(run_dir) / "metrics.json").read_text())
            row = {"month": f"{m_start:%Y-%m}", "ev_threshold": thr, "trades": met.get("trades", 0),
                   "return_pct": round(met.get("net_return_pct", 0.0), 2), "max_dd_pct": round(met.get("max_drawdown_pct", 0.0), 2),
                   "run": str(run_dir)}
            results[kind].append(row)
            log.info("WF %s %s: %s", kind, row["month"], row)
            Path(out_dir, "walkforward.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return summarize(results)


def summarize(results: dict) -> dict:
    out = {}
    for kind, rows in results.items():
        df = pd.DataFrame(rows)
        trades = pd.concat([pd.read_csv(Path(r) / "trades.csv") for r in df["run"]
                            if (Path(r) / "trades.csv").exists() and (Path(r) / "trades.csv").stat().st_size > 2]) \
            if len(df) else pd.DataFrame()
        pf = None
        if len(trades):
            g, l_ = trades.loc[trades.pnl_tmn > 0, "pnl_tmn"].sum(), -trades.loc[trades.pnl_tmn <= 0, "pnl_tmn"].sum()
            pf = round(float(g / l_), 2) if l_ > 0 else float("inf")
        total = round(100 * (float(np.prod(1 + df["return_pct"] / 100)) - 1), 2)
        months_ok = int((df["return_pct"] >= 0).sum())
        out[kind] = {"months": len(df), "total_pct": total, "profit_factor": pf, "trades": int(df["trades"].sum()),
                     "months_ge_0": months_ok,
                     "passes": bool(total > 0 and (pf or 0) >= 1.2 and df["trades"].sum() >= 30 and months_ok >= 8)}
        print(f"\n{kind}:")
        print(df.drop(columns=["run"]).to_string(index=False))
        print(out[kind])
    return out
