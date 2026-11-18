import numpy as np
import pandas as pd

from src.metrics import compute_metrics, summarise_folds
from src.stats_tests import bootstrap_ci, mcnemar_exact


def test_summary_matches_reported_table():
    df = pd.read_csv("results/fold_metrics.csv")
    s = summarise_folds(df, ["accuracy", "macro_precision", "macro_recall", "macro_f1"]).set_index("metric")
    assert abs(s.loc["accuracy", "mean"] - 98.00) < 0.01
    assert abs(s.loc["accuracy", "sd"] - 0.52) < 0.01
    assert abs(s.loc["macro_recall", "mean"] - 96.39) < 0.01
    assert abs(s.loc["accuracy", "ci_low"] - 97.35) < 0.01
    assert abs(s.loc["accuracy", "ci_high"] - 98.65) < 0.01


def test_compute_metrics_perfect():
    y = np.array([0, 1, 2, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 100.0 and m["macro_f1"] == 100.0


def test_mcnemar_symmetric_and_bounds():
    y = np.array([0, 1, 2] * 20)
    a = y.copy()
    b = y.copy(); b[:6] = (b[:6] + 1) % 3
    r = mcnemar_exact(y, a, b)
    assert r["a_only_correct"] == 6 and r["b_only_correct"] == 0
    assert 0 <= r["p_value"] <= 1 and r["p_value"] < 0.05


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200)
    p = np.where(rng.random(200) < 0.9, y, (y + 1) % 3)
    est, lo, hi = bootstrap_ci(y, p, lambda a, b: (a == b).mean() * 100, n_boot=300)
    assert lo <= est <= hi
