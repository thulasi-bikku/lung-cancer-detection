"""Grad-CAM on the final ResNet18 feature map (7 x 7 for a 224 x 224 input)."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model.eval()
        self._act = None
        self._grad = None
        self._fh = model.backbone.register_forward_hook(self._save_act)

    def _save_act(self, _m, _i, out):
        self._act = out
        out.register_hook(lambda g: setattr(self, "_grad", g))

    def remove(self) -> None:
        self._fh.remove()

    def __call__(self, x: torch.Tensor, target: int | None = None):
        """x: 1 x 3 x H x W. Returns (cam in [0,1] at input size, predicted class, probabilities)."""
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            logits = self.model(x)
            probs = logits.softmax(dim=1)[0].detach().cpu().numpy()
            cls = int(logits.argmax(1).item()) if target is None else int(target)
            logits[0, cls].backward()
        weights = self._grad.mean(dim=(2, 3), keepdim=True)          # alpha_k
        cam = F.relu((weights * self._act).sum(dim=1, keepdim=True))  # L^c
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, cls, probs


def overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return (alpha * heat + (1 - alpha) * image_rgb).astype(np.uint8)
