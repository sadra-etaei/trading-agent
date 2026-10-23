"""Candidate F (PLAN.md §15c): a GRU sequence model on the last `window` hourly feature rows of a pair.

Same table, labels, splits, calibration and EV-threshold tuning as the tree ensemble (engine/model/train.py),
so the comparison is fair. Saved as an EntryModel-compatible bundle: `SeqModel.predict_proba(X)` needs the
pair's recent rows, so live and backtest pass full feature frames (`proba_frames`).
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
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from engine.config import config_hash
from engine.features.build import feature_list
from engine.model.dataset import build_table, make_splits
from engine.model.train import tune_threshold

log = logging.getLogger(__name__)
WINDOW = 72


class GRUNet(torch.nn.Module):
    def __init__(self, n_feat: int, hidden: int = 48, dropout: float = 0.2):
        super().__init__()
        self.gru = torch.nn.GRU(n_feat, hidden, batch_first=True)
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 1))

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class SeqModel:
    """Scaler + GRU; `windows(frame)` builds the (rows, WINDOW, features) tensor for one pair."""

    def __init__(self, feats: list[str], mean: np.ndarray, std: np.ndarray, state: dict | list[dict], hidden: int = 48,
                 window: int = WINDOW, dropout: float = 0.2):
        self.feats, self.mean, self.std, self.hidden = feats, mean, std, hidden
        self.states = state if isinstance(state, list) else [state]  # several seeds -> averaged probabilities
        self.window, self.dropout = window, dropout
        self._nets = None

    def nets(self):
        if getattr(self, "_nets", None) is None:
            states = getattr(self, "states", None) or [self.state]  # bundles saved before seed averaging
            self._nets = []
            for st in states:
                n = GRUNet(len(self.feats), self.hidden, getattr(self, "dropout", 0.2))
                n.load_state_dict(st)
                n.eval()
                self._nets.append(n)
        return self._nets

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_nets"] = None
        return d

    def scaled(self, frame: pd.DataFrame) -> np.ndarray:
        x = (frame[self.feats].to_numpy(np.float32) - self.mean) / self.std
        return np.nan_to_num(np.clip(x, -6, 6), nan=0.0).astype(np.float32)

    def proba_frame(self, frame: pd.DataFrame) -> pd.Series:
        """Raw probability for every row of one pair's time-ordered feature frame (first WINDOW-1 rows NaN)."""
        x = self.scaled(frame)
        win = getattr(self, "window", WINDOW)
        out = np.full(len(x), np.nan, dtype=np.float32)
        if len(x) < win:
            return pd.Series(out, index=frame.index)
        idx = np.arange(win - 1, len(x))
        with torch.no_grad():
            for k in range(0, len(idx), 4096):
                sl = idx[k:k + 4096]
                w = torch.from_numpy(np.stack([x[i - win + 1:i + 1] for i in sl]))
                out[sl] = np.mean([torch.sigmoid(n(w)).numpy() for n in self.nets()], axis=0)
        return pd.Series(out, index=frame.index)


def _windows(table: pd.DataFrame, model: SeqModel) -> dict[str, pd.Series]:
    return {s: model.proba_frame(g.sort_index()) for s, g in table.groupby("symbol")}


def add_seq_proba(model_bundle: dict, feats: dict[str, pd.DataFrame]) -> None:
    """Backtest / live: a sequence model scores whole feature frames; the per-bar raw probability is stored
    in the `seq_p` column, which EntryModel reads instead of calling the classifier on single rows."""
    if not model_bundle.get("sequence"):
        return
    for s, f in feats.items():
        f["seq_p"] = model_bundle["model"].proba_frame(f.sort_index()).reindex(f.index)


def train_seq(cfg, epochs: int = 8, seed: int = 42) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = datetime.now(timezone.utc)
    table = build_table(cfg)
    feats = feature_list(bool(cfg.sentiment.funding), bool(cfg.universe.get("usdt_premium")),
                         cfg.model.get("exclude_features"), bool((cfg.get("outside") or {}).get("binance")),
                         bool(cfg.model.get("extra_features")))
    splits = make_splits(table[table["label"].notna()].index, cfg)
    train_rows = table[table.index < splits.train_end]
    mean = np.nanmean(train_rows[feats].to_numpy(np.float64), axis=0).astype(np.float32)
    std = (np.nanstd(train_rows[feats].to_numpy(np.float64), axis=0) + 1e-8).astype(np.float32)
    mean, std = np.nan_to_num(mean), np.where(np.isfinite(std) & (std > 0), std, 1.0).astype(np.float32)

    # windows are cut batch by batch from per-pair arrays (stacking them all would need ~3.5 GB of RAM);
    # early-stopping slice = the last 60 days before train_end
    es_start = splits.train_end - pd.Timedelta(days=60)
    tmp = SeqModel(feats, mean, std, {})
    arrays, tr_idx, es_idx = [], [], []
    for s_, g in table.groupby("symbol"):
        g = g.sort_index()
        x, y, ts = tmp.scaled(g), g["label"].to_numpy(), g.index
        sid = len(arrays)
        arrays.append((x, y))
        for i in range(WINDOW - 1, len(g)):
            if np.isfinite(y[i]) and ts[i] < splits.train_end:
                (es_idx if ts[i] >= es_start else tr_idx).append((sid, i))
    tr_idx, es_idx = np.array(tr_idx), np.array(es_idx)

    def batch(ix):
        xb = np.stack([arrays[a][0][i - WINDOW + 1:i + 1] for a, i in ix])
        yb = np.array([arrays[a][1][i] for a, i in ix], dtype=np.float32)
        return torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)

    log.info("GRU: %d train windows, %d early-stop windows, %d features, device %s", len(tr_idx), len(es_idx), len(feats), dev)
    net = GRUNet(len(feats)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos = float(np.mean([arrays[a][1][i] for a, i in tr_idx]))
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-6), device=dev))
    best, best_state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(len(tr_idx))
        for k in range(0, len(perm), 1024):
            xb, yb = batch(tr_idx[perm[k:k + 1024]])
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        logits, ys = [], []
        with torch.no_grad():
            for k in range(0, len(es_idx), 4096):
                xb, yb = batch(es_idx[k:k + 4096])
                logits.append(net(xb).cpu())
                ys.append(yb.cpu())
        es_logits, yes = torch.cat(logits), torch.cat(ys)
        es_loss = float(loss_fn(es_logits.to(dev), yes.to(dev)))
        es_auc = roc_auc_score(yes.numpy(), torch.sigmoid(es_logits).numpy()) if yes.unique().numel() > 1 else float("nan")
        log.info("epoch %d: early-stop loss %.4f auc %.3f", ep + 1, es_loss, es_auc)
        if es_loss < best - 1e-4:
            best, best_state, bad = es_loss, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 2:
                break
    model = SeqModel(feats, mean, std, best_state)

    # calibrate + tune the EV threshold on the validation window (no out-of-fold rows for a sequence model)
    per = _windows(table, model)
    rows = []
    for s, g in table[table["label"].notna()].groupby("symbol"):
        g = g.copy()
        g["raw_p"] = per[s].reindex(g.index).to_numpy()
        rows.append(g)
    lab = pd.concat(rows).sort_index()
    val = lab[(lab.index >= splits.val_start) & (lab.index < splits.val_end) & lab["raw_p"].notna()]
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(val["raw_p"], val["label"].astype(int))
    ev_thr, grid = tune_threshold(val, calibrator.predict(val["raw_p"]), cfg)
    auc = float(roc_auc_score(val["label"], val["raw_p"])) if val["label"].nunique() > 1 else None

    version = t0.strftime("%Y%m%d-%H%M%S") + "-gru"
    bundle = {"model": model, "calibrator": calibrator, "features": feats, "version": version, "sequence": WINDOW,
              "thresholds": {"min_ev_pct": ev_thr}, "splits": splits.as_dict(), "labels": dict(cfg.labels),
              "costs": dict(cfg.costs), "symbols": sorted(table["symbol"].unique().tolist()),
              "config_hash": config_hash(cfg), "trained_at": t0.isoformat()}
    out_dir = Path(cfg.model.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"entry-{version}.joblib"
    joblib.dump(bundle, path)
    shutil.copyfile(path, out_dir / "latest.joblib")
    report = {"version": version, "validation_auc": auc, "ev_threshold": ev_thr, "train_windows": int(len(tr_idx)),
              "early_stop_loss": best, "grid": grid, "splits": splits.as_dict()}
    (out_dir / f"train-report-{version}.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "grid"}, indent=2, default=str))
    return report
