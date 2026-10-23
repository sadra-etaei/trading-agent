"""Model bundle: classifier + isotonic calibrator + metadata. Used everywhere the model is scored."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class EntryModel:
    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.clf = bundle["model"]
        self.calibrator = bundle.get("calibrator")
        self.features: list[str] = bundle["features"]
        self.version: str = bundle["version"]
        self.thresholds: dict = bundle.get("thresholds", {})

    @classmethod
    def load(cls, path: str | Path) -> "EntryModel":
        return cls(joblib.load(path))

    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.bundle.get("sequence"):  # GRU (engine/model/seq.py): scored on whole frames -> `seq_p` column
            return np.nan_to_num(X["seq_p"].to_numpy(float), nan=0.0)
        return self.clf.predict_proba(X[self.features])[:, 1]

    def proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.raw_proba(X)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    def top_features(self, n: int = 10) -> list[tuple[str, float]]:
        imp = getattr(self.clf, "feature_importances_", None)
        if imp is None and hasattr(self.clf, "named_steps"):
            coef = self.clf.named_steps["lr"].coef_[0]
            imp = np.abs(coef)
        if imp is None:
            return []
        order = np.argsort(imp)[::-1][:n]
        return [(self.features[i], float(imp[i])) for i in order]


def expected_value(p: np.ndarray, tp_pct: np.ndarray, sl_pct: np.ndarray, cost_pct: float) -> np.ndarray:
    """EV of a long trade per unit notional: p*TP - (1-p)*SL - round-trip costs."""
    return p * tp_pct - (1 - p) * sl_pct - cost_pct
