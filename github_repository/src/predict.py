"""Single-image inference with an optional Grad-CAM overlay.

    python -m src.predict --weights runs/hybrid/seed42/fold1/best.pt --image slice.jpg --cam cam.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from .config import load_config
from .data import build_transforms
from .gradcam import GradCAM, overlay
from .model import CLASS_NAMES, build_model
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--cam", default=None, help="optional output path for a Grad-CAM overlay")
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    img = cv2.cvtColor(cv2.imread(args.image, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (cfg["image_size"],) * 2, interpolation=cv2.INTER_LINEAR)
    x = build_transforms(False)(Image.fromarray(img)).unsqueeze(0)

    cam_tool = GradCAM(model)
    cam, cls, probs = cam_tool(x)
    print(f"Image: {args.image}\nPredicted class: {CLASS_NAMES[cls]}\nConfidence: {probs[cls]*100:.2f}%\n")
    print("Class probabilities:")
    for n, p in zip(CLASS_NAMES, probs):
        print(f"  {n}: {p*100:.2f}%")
    if args.cam:
        cv2.imwrite(args.cam, cv2.cvtColor(overlay(img, cam), cv2.COLOR_RGB2BGR))
        print(f"Grad-CAM overlay written to {args.cam}")


if __name__ == "__main__":
    main()
