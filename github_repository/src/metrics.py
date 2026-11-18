"""Metric computation and fold-level summaries (NumPy / scikit-learn / SciPy only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_recall_fscore_support

CLASS_NAMES = ("Benign", "Malignant", "Normal")


def compute_metrics(y_true, y_pred, labels=(0, 1, 2)) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean() * 100)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    p, r, f = p * 100, r * 100, f * 100
    f_hm = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {
        "accuracy": acc,
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),                 # unweighted mean of class-wise F1
        "macro_f1_harmonic": float(f_hm),     # harmonic mean of macro P and macro R
    }


def summarise_folds(df: pd.DataFrame, columns=None, level: float = 0.95) -> pd.DataFrame:
    """Mean, sample SD and t-based interval of the mean across folds."""
    columns = columns or [c for c in df.columns if c != "fold"]
    n = len(df)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    rows = []
    for c in columns:
        v = df[c].to_numpy(dtype=float)
        m, s = v.mean(), v.std(ddof=1)
        h = t * s / np.sqrt(n)
        rows.append({"metric": c, "mean": m, "sd": s, "ci_low": m - h, "ci_high": m + h})
    return pd.DataFrame(rows)
