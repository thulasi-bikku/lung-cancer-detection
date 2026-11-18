"""Parameter count, model size, latency and (optionally) FLOPs.

    python -m src.complexity --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .model import build_model, count_parameters


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def latency_ms(model, device, runs: int = 100, warmup: int = 10, size: int = 224) -> float:
    x = torch.randn(1, 3, size, size, device=device)
    model.eval().to(device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    _sync(device)
    return (time.perf_counter() - t0) / runs * 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()

    cfg = load_config(args.config)
    model = build_model(cfg["model"], pretrained=False).eval()
    n = count_parameters(model)
    print(f"parameters : {n/1e6:.2f} M")
    print(f"fp32 size  : {n*4/1024**2:.1f} MB")

    try:
        from fvcore.nn import FlopCountAnalysis
        flops = FlopCountAnalysis(model, torch.randn(1, 3, 224, 224)).total()
        print(f"FLOPs      : {flops/1e9:.2f} G (fvcore, multiply-adds)")
    except ImportError:
        print("FLOPs      : install fvcore to enable (pip install fvcore)")

    print(f"CPU latency: {latency_ms(model, torch.device('cpu'), args.runs):.1f} ms")
    for name, ok in (("cuda", torch.cuda.is_available()),
                     ("mps", getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())):
        if ok:
            print(f"{name.upper()} latency: {latency_ms(model, torch.device(name), args.runs):.1f} ms")


if __name__ == "__main__":
    main()
