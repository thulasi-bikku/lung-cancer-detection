#!/usr/bin/env python3
"""Plot accuracy-versus-epoch curves from the history.json files written by src.train.

    python scripts/plot_training_curves.py --run runs/hybrid --seed 42 --fold 1 --out acc_vs_epoch
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fold", type=int, default=1)
ap.add_argument("--out", default="accuracy_vs_epoch")
a = ap.parse_args()

h = json.loads((Path(a.run) / f"seed{a.seed}" / f"fold{a.fold}" / "history.json").read_text())
ep = range(1, len(h["train_acc"]) + 1)
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.plot(ep, h["train_acc"], marker="o", ms=4, lw=1.4, color="#2a78d6", label="Training")
ax.plot(ep, h["val_acc"], marker="s", ms=4, lw=1.4, color="#eb6834", label="Validation")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.grid(True, color="#d9d8d3", lw=0.6)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{a.out}.pdf"); fig.savefig(f"{a.out}.png", dpi=600)
print("saved", a.out)
