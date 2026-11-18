"""Dataset utilities for the IQ-OTH/NCCD CT image set."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .model import CLASS_NAMES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# The public release spells the Benign folder "Bengin cases"; both spellings are accepted.
_FOLDER_TO_LABEL = {
    "bengin cases": 0, "benign cases": 0, "benign": 0, "bengin": 0,
    "malignant cases": 1, "malignant": 1,
    "normal cases": 2, "normal": 2,
}


def scan_dataset(root: str | Path) -> pd.DataFrame:
    """Return a DataFrame with columns ``path``, ``label``, ``filename``."""
    root = Path(root)
    rows = []
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        label = _FOLDER_TO_LABEL.get(d.name.lower())
        if label is None:
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_EXT:
                rows.append({"path": str(f), "label": label, "filename": f.name})
    if not rows:
        raise FileNotFoundError(f"No class folders with images found under {root}")
    return pd.DataFrame(rows)


class AddGaussianNoise:
    def __init__(self, std: float = 0.01) -> None:
        self.std = std

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)


def build_transforms(train: bool, aug: dict | None = None, size: int = 224) -> Callable:
    aug = aug or {}
    normalise = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if not train:
        return T.Compose([T.ToTensor(), normalise])
    b = aug.get("brightness", 0.2)
    return T.Compose([
        T.RandomResizedCrop(size, scale=tuple(aug.get("crop_scale", (0.8, 1.0)))),
        T.RandomRotation(aug.get("rotation_deg", 10)),
        T.ColorJitter(brightness=(1 - b, 1 + b)),
        T.ToTensor(),
        AddGaussianNoise(aug.get("noise_std", 0.01)),
        normalise,
    ])


class CTSliceDataset(Dataset):
    """Reads an image with OpenCV (BGR->RGB), resizes to ``size`` and applies a transform."""

    def __init__(self, frame: pd.DataFrame, transform: Callable, size: int = 224) -> None:
        self.paths = frame["path"].tolist()
        self.labels = frame["label"].astype(int).tolist()
        self.transform = transform
        self.size = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(self.paths[idx], cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"Cannot read image: {self.paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return self.transform(Image.fromarray(img)), self.labels[idx]


def class_weights(labels: Sequence[int], num_classes: int = 3) -> torch.Tensor:
    """w_c = n / (K * n_c)."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    return torch.tensor(len(labels) / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)


def sample_weights(labels: Sequence[int], num_classes: int = 3) -> torch.Tensor:
    """Per-sample weights proportional to inverse class frequency (for WeightedRandomSampler)."""
    counts = np.bincount(np.asarray(labels), minlength=num_classes).astype(np.float64)
    inv = 1.0 / np.maximum(counts, 1)
    return torch.tensor([inv[l] for l in labels], dtype=torch.double)


def load_groups(frame: pd.DataFrame, groups_csv: str | Path | None):
    """Optional case identifiers: CSV with columns ``filename,case_id``."""
    if groups_csv is None:
        return None
    g = pd.read_csv(groups_csv)
    merged = frame.merge(g, on="filename", how="left")
    if merged["case_id"].isna().any():
        missing = int(merged["case_id"].isna().sum())
        raise ValueError(f"{missing} images have no case_id in {groups_csv}")
    return merged["case_id"].to_numpy()


__all__ = ["CLASS_NAMES", "scan_dataset", "CTSliceDataset", "build_transforms",
           "class_weights", "sample_weights", "load_groups"]
