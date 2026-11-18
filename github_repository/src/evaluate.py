"""Summarise a finished run: fold table, mean/SD/95% interval and a pooled classification report.

    python -m src.evaluate --run runs/hybrid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .metrics import CLASS_NAMES, summarise_folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = Path(args.run)

    folds = pd.read_csv(run / "fold_metrics.csv")
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_f1_harmonic"]
    per_fold = folds.groupby("fold")[cols].mean().reset_index()
    print(per_fold.round(2).to_string(index=False))
    print()
    print(summarise_folds(per_fold, cols).round(2).to_string(index=False))

    preds = [pd.read_csv(p) for p in sorted(run.glob("seed*/fold*/predictions.csv"))]
    if preds:
        allp = pd.concat(preds)
        print("\nPooled classification report (all folds, all seeds):")
        print(classification_report(allp["true"], allp["pred"], target_names=CLASS_NAMES, digits=3))
        print(confusion_matrix(allp["true"], allp["pred"]))


if __name__ == "__main__":
    main()
