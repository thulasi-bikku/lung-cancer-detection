#!/usr/bin/env python3
"""Confusion matrix from the predictions.csv files written by src.train.

    python scripts/plot_confusion_matrix.py --run runs/hybrid --seed 42 --fold 1 --out confusion_fold1
    python scripts/plot_confusion_matrix.py --run runs/hybrid --pooled --out confusion_pooled
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

NAMES = ["Benign", "Malignant", "Normal"]
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--pooled", action="store_true")
ap.add_argument("--out", default="confusion_matrix")
a = ap.parse_args()

run = Path(a.run)
files = sorted(run.glob("seed*/fold*/predictions.csv")) if a.pooled else \
    [run / f"seed{a.seed}" / f"fold{a.fold}" / "predictions.csv"]
df = pd.concat([pd.read_csv(f) for f in files])
cm = confusion_matrix(df["true"], df["pred"], labels=[0, 1, 2])

plt.rcParams.update({"font.family": "serif"})
fig, ax = plt.subplots(figsize=(4.0, 3.5))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3)); ax.set_xticklabels(NAMES); ax.set_yticks(range(3)); ax.set_yticklabels(NAMES)
ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
for i in range(3):
    for j in range(3):
        ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.tight_layout()
fig.savefig(f"{a.out}.pdf"); fig.savefig(f"{a.out}.png", dpi=600)
print(cm)
