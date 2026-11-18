"""Optional uncertainty and significance tools (not used for the paper's reported numbers)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def mcnemar_exact(y_true, pred_a, pred_b) -> dict:
    """Exact (binomial) McNemar test for two classifiers evaluated on the same slices."""
    y_true, a, b = map(np.asarray, (y_true, pred_a, pred_b))
    a_ok, b_ok = a == y_true, b == y_true
    n01 = int(np.sum(a_ok & ~b_ok))   # A right, B wrong
    n10 = int(np.sum(~a_ok & b_ok))   # A wrong, B right
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(min(n01, n10), n, 0.5).pvalue)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": p}


def bootstrap_ci(y_true, y_pred, metric, n_boot: int = 2000, level: float = 0.95,
                 seed: int = 0, groups=None):
    """Percentile bootstrap interval. With ``groups`` the resampling unit is the group (case)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if groups is None:
        idx_pool = [np.arange(n)]
        draw = lambda: rng.integers(0, n, n)  # noqa: E731
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        members = {g: np.where(groups == g)[0] for g in uniq}
        draw = lambda: np.concatenate(  # noqa: E731
            [members[g] for g in rng.choice(uniq, len(uniq), replace=True)]
        )
    vals = np.empty(n_boot)
    for i in range(n_boot):
        idx = draw()
        vals[i] = metric(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(metric(y_true, y_pred)), float(lo), float(hi)
