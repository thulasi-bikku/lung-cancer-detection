#!/usr/bin/env python3
"""Generate the vector/high-resolution figures used in the manuscript.

All charts are drawn from the CSV files in ``results/`` (the numbers reported in
the paper). Nothing is simulated.  CT-slice figures are composed from image
files supplied through ``--assets``.

Usage
-----
    python paper_figures/make_figures.py --results results --out paper_figures/output \
        [--assets path/to/ct_panels]

Each figure is written as a vector PDF and as a 600 dpi PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy import stats

# --------------------------------------------------------------------------- #
# Global style: serif text to match the manuscript, thin marks, quiet axes.
# --------------------------------------------------------------------------- #
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#52514e",
        "axes.linewidth": 0.8,
        "grid.color": "#d9d8d3",
        "grid.linewidth": 0.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "figure.dpi": 150,
    }
)

BLUE = "#2a78d6"      # proposed model / emphasis
ORANGE = "#eb6834"    # secondary series
GREY = "#8d8c86"      # reference models
INK = "#0b0b0b"
INK2 = "#52514e"

# Stage colours for the architecture diagrams (muted, print-safe, labelled).
STAGE = {
    "io": "#eceae4",
    "cnn": "#d6e4f7",
    "proj": "#e6dff5",
    "tok": "#fbe3d6",
    "enc": "#fdf0cc",
    "head": "#d9efe2",
}


def save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False, ec="#3a3935", lw=0.9, sub=None):
    p = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
        fc=fc, ec=ec, lw=lw, zorder=2,
    )
    ax.add_patch(p)
    if sub:
        ax.text(x + w / 2, y + h * 0.64, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, zorder=3)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=fs - 1.3,
                color=INK2, zorder=3)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=INK, zorder=3)
    return p


def arrow(ax, x0, y0, x1, y1, lw=1.0, rad=0.0, color="#3a3935"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=9, lw=lw,
            color=color, connectionstyle=f"arc3,rad={rad}", zorder=1,
        )
    )


# --------------------------------------------------------------------------- #
# Figure 1: workflow
# --------------------------------------------------------------------------- #
def fig_workflow(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    bw, bh = 2.1, 1.05
    top = [
        ("CT slice", "single 2D image", STAGE["io"]),
        ("Preprocessing", "resize, normalise", STAGE["io"]),
        ("ResNet18", "ImageNet weights", STAGE["cnn"]),
        ("1\u00d71 projection", "512 \u2192 256 channels", STAGE["proj"]),
    ]
    bot = [
        ("Prediction", "three classes", STAGE["head"]),
        ("Linear classifier", "256 \u2192 3 logits", STAGE["head"]),
        ("Token averaging", "global average pooling", STAGE["head"]),
        ("Transformer", "2 encoder layers", STAGE["enc"]),
    ]
    xs = [0.15 + i * 2.5 for i in range(4)]
    ytop, ybot = 2.9, 0.75
    for x, (t, s_, c) in zip(xs, top):
        box(ax, x, ytop, bw, bh, t, c, sub=s_, bold=True, fs=8)
    for x, (t, s_, c) in zip(xs, bot):
        box(ax, x, ybot, bw, bh, t, c, sub=s_, bold=True, fs=8)
    for i in range(3):
        arrow(ax, xs[i] + bw, ytop + bh / 2, xs[i + 1], ytop + bh / 2)
    for i in range(3, 0, -1):
        arrow(ax, xs[i], ybot + bh / 2, xs[i - 1] + bw, ybot + bh / 2)
    arrow(ax, xs[3] + bw / 2, ytop, xs[3] + bw / 2, ybot + bh)
    ax.text(xs[3] + bw / 2 + 0.15, (ytop + ybot + bh) / 2, "flatten to 49\ntokens, add\nposition\nembedding",
            fontsize=7, color=INK2, va="center", ha="left")
    ax.text(5.0, 0.2, "Training: weighted sampling, class-weighted loss, Adam, early stopping.  "
            "Evaluation: stratified five-fold cross-validation.", fontsize=7.2, color=INK2, ha="center")
    save(fig, out, "fig01_workflow")


# --------------------------------------------------------------------------- #
# Figure 2: detailed architecture with tensor shapes
# --------------------------------------------------------------------------- #
def fig_architecture(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # Row 1: input -> backbone -> projection -> flatten
    box(ax, 0.1, 6.0, 2.0, 1.4, "Input", STAGE["io"], bold=True, sub="3\u00d7224\u00d7224", fs=8)
    box(ax, 3.0, 6.0, 3.4, 1.4, "ResNet18 backbone", STAGE["cnn"], bold=True,
        sub="conv1 and layer1\u20134 (no fc)", fs=8)
    box(ax, 7.5, 6.0, 1.7, 1.4, "1\u00d71 conv", STAGE["proj"], bold=True, sub="512 \u2192 256", fs=8)
    box(ax, 10.1, 6.0, 1.8, 1.4, "Flatten", STAGE["tok"], bold=True, sub="49\u00d7256", fs=8)
    arrow(ax, 2.1, 6.7, 3.0, 6.7)
    arrow(ax, 6.4, 6.7, 7.5, 6.7)
    arrow(ax, 9.2, 6.7, 10.1, 6.7)
    ax.text(6.9, 6.95, "512\u00d77\u00d77", fontsize=6.3, color=INK2, ha="center")
    ax.text(9.65, 6.95, "256\u00d77\u00d77", fontsize=6.0, color=INK2, ha="center", va="bottom")

    # positional embedding
    box(ax, 10.0, 4.5, 1.9, 0.95, "+ E_pos", STAGE["tok"], bold=True, sub="learnable", fs=8)
    arrow(ax, 10.95, 6.0, 10.95, 5.45)

    # Encoder panel
    ax.add_patch(FancyBboxPatch((3.0, 1.5), 6.3, 4.3, boxstyle="round,pad=0.0,rounding_size=0.08",
                                fc="#fffaf0", ec="#b89a3c", lw=1.0, ls="--", zorder=0))
    ax.text(3.2, 5.45, "Transformer encoder layer (\u00d72)", fontsize=8.3, fontweight="bold", color=INK, va="center")
    bw, bh = 2.55, 0.85
    box(ax, 3.2, 3.95, bw, bh, "Multi-head attention", STAGE["enc"], fs=7.6)
    box(ax, 6.55, 3.95, bw, bh, "Add & LayerNorm", STAGE["enc"], fs=7.6)
    box(ax, 6.55, 2.45, bw, bh, "Feed-forward net", STAGE["enc"], fs=7.6)
    box(ax, 3.2, 2.45, bw, bh, "Add & LayerNorm", STAGE["enc"], fs=7.6)
    arrow(ax, 5.75, 4.375, 6.55, 4.375)
    arrow(ax, 7.825, 3.95, 7.825, 3.3)
    arrow(ax, 6.55, 2.875, 5.75, 2.875)
    ax.text(6.15, 1.8, "input and output: 49\u00d7256 tokens", fontsize=7, color=INK2, ha="center")
    arrow(ax, 10.0, 4.97, 9.3, 4.97)
    ax.text(9.65, 5.12, "Z\u2080", fontsize=7, color=INK2, ha="center")

    # Head
    box(ax, 0.1, 2.45, 2.2, 0.95, "Global average\npooling", STAGE["head"], fs=7.6, bold=True)
    box(ax, 0.1, 0.85, 2.2, 0.95, "Linear 256\u21923", STAGE["head"], fs=7.6, bold=True)
    arrow(ax, 3.2, 2.875, 2.3, 2.875)
    arrow(ax, 1.2, 2.45, 1.2, 1.8)
    arrow(ax, 1.2, 0.85, 1.2, 0.55)
    ax.text(1.2, 0.3, "Benign | Malignant | Normal", fontsize=7.2, ha="center", color=INK)
    ax.text(2.75, 3.0, "256-d", fontsize=6.0, color=INK2, ha="center", va="bottom")
    save(fig, out, "fig02_architecture")


# --------------------------------------------------------------------------- #
# Results charts
# --------------------------------------------------------------------------- #
def fig_folds(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "fold_metrics.csv")
    metrics = [("accuracy", "Accuracy"), ("macro_precision", "Macro precision"),
               ("macro_recall", "Macro recall"), ("macro_f1", "Macro F1-score")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=False)
    for ax, (col, lab) in zip(axes, metrics):
        v = df[col].to_numpy()
        m, s = v.mean(), v.std(ddof=1)
        h = stats.t.ppf(0.975, len(v) - 1) * s / np.sqrt(len(v))
        ax.axhspan(m - h, m + h, color=BLUE, alpha=0.14, lw=0)
        ax.axhline(m, color=BLUE, lw=1.2)
        ax.scatter(df["fold"], v, s=26, color=BLUE, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_xticks(df["fold"])
        ax.set_xlabel("Fold")
        ax.set_title(lab, fontsize=9)
        ax.grid(axis="y")
        ax.set_ylim(min(v.min(), m - h) - 0.35, max(v.max(), m + h) + 0.35)
        ax.text(0.5, 0.04, f"mean {m:.2f}", transform=ax.transAxes, ha="center", fontsize=7.5, color=INK2)
    axes[0].set_ylabel("Percent")
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_perfold")


def fig_baselines(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "baselines.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.05, 1]})
    d = df.sort_values("accuracy")
    cols = [BLUE if "Hybrid" in n else GREY for n in d["model"]]
    a1.barh(d["model"], d["accuracy"], color=cols, height=0.62)
    for y, v in enumerate(d["accuracy"]):
        a1.text(v + 0.25, y, f"{v:.2f}", va="center", fontsize=7.8, color=INK)
    a1.set_xlim(80, 101)
    a1.set_xlabel("Mean accuracy over five folds (%)")
    a1.grid(axis="x")
    a1.set_axisbelow(True)
    a1.set_title("(a) Accuracy", fontsize=9)

    for _, r in df.iterrows():
        hy = "Hybrid" in r["model"]
        a2.scatter(r["params_m"], r["accuracy"], s=46, color=BLUE if hy else GREY,
                   edgecolor="white", linewidth=0.8, zorder=3)
    offs = {"CNN baseline": (6, -9), "ResNet18": (6, -10), "DenseNet-121": (-6, -15),
            "EfficientNet-B3": (7, 1), "ViT-B/16": (-30, -14), "Swin-T": (6, 4),
            "Hybrid (proposed)": (-8, 8)}
    for _, r in df.iterrows():
        dx, dy = offs.get(r["model"], (5, 5))
        a2.annotate(r["model"], (r["params_m"], r["accuracy"]), xytext=(dx, dy),
                    textcoords="offset points", fontsize=7.6, color=INK)
    a2.set_xscale("log")
    a2.set_xlabel("Parameters (millions, log scale)")
    a2.set_ylabel("Accuracy (%)")
    a2.set_ylim(86, 100)
    a2.set_xlim(6, 130)
    a2.grid(True)
    a2.set_title("(b) Accuracy versus model size", fontsize=9)
    fig.tight_layout(w_pad=1.5)
    save(fig, out, "fig_baselines")


def fig_ablation(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "ablation.csv")
    order = ["ResNet18 baseline", "ResNet18 + projection", "ResNet18 + Transformer (no projection)",
             "Hybrid without positional embedding", "Hybrid 1 layer", "Hybrid 2 layers (proposed)",
             "Hybrid 4 layers"]
    df = df.set_index("variant").loc[order].reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    y = np.arange(len(df))[::-1]
    cols = [BLUE if "proposed" in v else GREY for v in df["variant"]]
    ax.barh(y, df["accuracy"], color=cols, height=0.62)
    ax.scatter(df["macro_f1"], y, marker="D", s=22, color="white", edgecolor=INK, linewidth=0.9,
               zorder=4, label="Macro F1-score")
    for yy, a in zip(y, df["accuracy"]):
        ax.text(a + 0.2, yy + 0.02, f"{a:.2f}", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(df["variant"])
    ax.set_xlim(86, 101)
    ax.set_xlabel("Mean over five folds (%); bars: accuracy")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    save(fig, out, "fig_ablation")


def fig_efficiency(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "efficiency.csv")
    cols = [BLUE if "Hybrid" in m else GREY for m in df["model"]]
    panels = [("gpu_ms", "GPU latency (ms)"), ("cpu_ms", "CPU latency (ms)"),
              ("gpu_mem_mb", "GPU memory (MB)"), ("flops_g", "FLOPs (G)")]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6))
    for ax, (c, t) in zip(axes, panels):
        ax.bar(range(len(df)), df[c], color=cols, width=0.66)
        for i, v in enumerate(df[c]):
            ax.text(i, v * 1.02, f"{v:g}", ha="center", va="bottom", fontsize=7.3, color=INK)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(["ResNet18", "EffNet-B3", "Swin-T", "Hybrid"], fontsize=7.4, rotation=35, ha="right")
        ax.set_title(t, fontsize=8.6)
        ax.set_ylim(0, df[c].max() * 1.18)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    fig.tight_layout(w_pad=0.8)
    save(fig, out, "fig_efficiency")


def fig_class_report(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "classification_report_fold.csv")
    mat = df[["precision", "recall", "f1"]].to_numpy()
    fig, ax = plt.subplots(figsize=(3.6, 2.2))
    im = ax.imshow(mat, cmap="Blues", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Precision", "Recall", "F1-score"])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(df["class"], df["support"])])
    for i in range(mat.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if mat[i, j] > 0.85 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    save(fig, out, "fig_class_report")


# --------------------------------------------------------------------------- #
# Raster figures composed from supplied CT panels
# --------------------------------------------------------------------------- #
def load_gray(p: Path):
    from PIL import Image
    return np.array(Image.open(p).convert("L"))


def fig_example_predictions(assets: Path, out: Path) -> None:
    labels = [("Benign", "Benign"), ("Malignant", "Malignant"), ("Benign", "Benign")]
    files = [assets / f"pred_{i}.png" for i in range(3)]
    if not all(f.exists() for f in files):
        print("skip example predictions (panels missing)")
        return
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.5))
    for ax, f, (t, p), letter in zip(axes, files, labels, "abc"):
        ax.imshow(load_gray(f), cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        ax.set_title(f"({letter}) True: {t}\nPredicted: {p}", fontsize=8.6)
        ax.axis("off")
    fig.tight_layout(w_pad=0.6)
    save(fig, out, "fig_example_predictions")


def fig_dataset_grid(assets: Path, out: Path) -> None:
    names = ["Benign", "Malignant", "Normal"]
    files = [[assets / f"ds_{r}_{c}.png" for c in range(3)] for r in range(3)]
    if not all(f.exists() for row in files for f in row):
        print("skip dataset grid (panels missing)")
        return
    fig, axes = plt.subplots(3, 3, figsize=(5.0, 5.2))
    for r, row in enumerate(files):
        for c, f in enumerate(row):
            ax = axes[r, c]
            ax.imshow(load_gray(f), cmap="gray", vmin=0, vmax=255, interpolation="lanczos", aspect="auto")
            ax.axis("off")
            if c == 0:
                ax.text(-0.04, 0.5, names[r], transform=ax.transAxes, rotation=90, ha="right",
                        va="center", fontsize=9.5, fontweight="bold")
    fig.subplots_adjust(wspace=0.03, hspace=0.03)
    save(fig, out, "fig_dataset")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("paper_figures/output"))
    ap.add_argument("--assets", type=Path, default=None, help="folder with CT panels (see README)")
    a = ap.parse_args()

    fig_workflow(a.out)
    fig_architecture(a.out)
    fig_folds(a.results, a.out)
    fig_baselines(a.results, a.out)
    fig_ablation(a.results, a.out)
    fig_efficiency(a.results, a.out)
    fig_class_report(a.results, a.out)
    if a.assets:
        fig_dataset_grid(a.assets, a.out)
        fig_example_predictions(a.assets, a.out)


if __name__ == "__main__":
    main()
