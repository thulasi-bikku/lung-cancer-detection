"""Stratified k-fold training and evaluation.

Example
-------
    python -m src.train --data "/path/to/The IQ-OTHNCCD lung cancer dataset" \
        --config configs/default.yaml --out runs/hybrid

Optional
--------
    --groups case_ids.csv   CSV (filename,case_id): uses StratifiedGroupKFold so that all
                            slices of a case stay in one fold (needs case identifiers).
    --seeds 42 43 44        repeat the whole cross-validation for several seeds.
    --pretrained / --no-pretrained

Outputs (per seed, per fold): history.json, predictions.csv, best.pt; and fold_metrics.csv.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config
from .data import (CTSliceDataset, build_transforms, class_weights, load_groups,
                   sample_weights, scan_dataset)
from .metrics import compute_metrics, summarise_folds
from .model import CLASS_NAMES, build_model, count_parameters


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    losses, ys, ps, probs = [], [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if criterion is not None:
            losses.append(criterion(out, y).item() * len(y))
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
        probs.append(out.softmax(1).cpu().numpy())
    y_true, y_pred, p = np.concatenate(ys), np.concatenate(ps), np.concatenate(probs)
    loss = float(np.sum(losses) / len(y_true)) if losses else float("nan")
    return loss, y_true, y_pred, p


def run_fold(cfg, frame, tr_idx, va_idx, device, out_dir: Path, pretrained: bool):
    tcfg = cfg["train"]
    train_df, val_df = frame.iloc[tr_idx], frame.iloc[va_idx]
    size = cfg["image_size"]

    train_ds = CTSliceDataset(train_df, build_transforms(True, cfg["augmentation"], size), size)
    val_ds = CTSliceDataset(val_df, build_transforms(False, size=size), size)

    sampler = WeightedRandomSampler(sample_weights(train_df["label"].tolist()),
                                    num_samples=len(train_df), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], sampler=sampler,
                              num_workers=tcfg["num_workers"])
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False,
                            num_workers=tcfg["num_workers"])

    model = build_model(cfg["model"], pretrained=pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(train_df["label"].tolist()).to(device))
    optim = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode="min", factor=tcfg["scheduler_factor"], patience=tcfg["scheduler_patience"])

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_score, best_state, bad = -np.inf, None, 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        run_loss, correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optim.step()
            run_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            seen += len(y)
        val_loss, yt, yp, _ = evaluate(model, val_loader, device, criterion)
        val_acc = float((yt == yp).mean())
        sched.step(val_loss)

        history["train_loss"].append(run_loss / seen)
        history["train_acc"].append(correct / seen)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(optim.param_groups[0]["lr"])
        print(f"  epoch {epoch:02d}  train_loss {run_loss/seen:.4f}  train_acc {correct/seen:.4f}  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

        score = val_acc if tcfg["monitor"] == "val_acc" else -val_loss
        if score > best_score:
            best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= tcfg["early_stopping_patience"]:
                print("  early stopping")
                break

    model.load_state_dict(best_state)
    _, yt, yp, probs = evaluate(model, val_loader, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    pred = pd.DataFrame({"filename": val_df["filename"].values, "true": yt, "pred": yp})
    for i, n in enumerate(CLASS_NAMES):
        pred[f"p_{n.lower()}"] = probs[:, i]
    pred.to_csv(out_dir / "predictions.csv", index=False)
    return compute_metrics(yt, yp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root containing the three class folders")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="runs/hybrid")
    ap.add_argument("--groups", default=None, help="optional CSV with columns filename,case_id")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = args.seeds or [cfg["seed"]]
    device = pick_device()
    frame = scan_dataset(args.data).reset_index(drop=True)
    groups = load_groups(frame, args.groups)
    print(f"device: {device} | images: {len(frame)} | per class: "
          f"{frame['label'].value_counts().sort_index().to_dict()}")
    print(f"parameters: {count_parameters(build_model(cfg['model'], pretrained=False))/1e6:.2f} M")

    all_rows = []
    for seed in seeds:
        set_seed(seed)
        if groups is None:
            splitter = StratifiedKFold(cfg["n_folds"], shuffle=True, random_state=seed)
            splits = splitter.split(frame, frame["label"])
        else:
            splitter = StratifiedGroupKFold(cfg["n_folds"], shuffle=True, random_state=seed)
            splits = splitter.split(frame, frame["label"], groups)
        for fold, (tr, va) in enumerate(splits, start=1):
            print(f"seed {seed} | fold {fold}/{cfg['n_folds']}  (train {len(tr)}, val {len(va)})")
            m = run_fold(cfg, frame, tr, va, device,
                         Path(args.out) / f"seed{seed}" / f"fold{fold}", not args.no_pretrained)
            all_rows.append({"seed": seed, "fold": fold, **m})
            print("  ", {k: round(v, 2) for k, v in m.items()})

    res = pd.DataFrame(all_rows)
    res.to_csv(Path(args.out) / "fold_metrics.csv", index=False)
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_f1_harmonic"]
    print(summarise_folds(res.groupby("fold")[cols].mean().reset_index(), cols).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
