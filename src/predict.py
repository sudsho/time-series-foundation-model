"""
load a finetuned forecaster and produce forecasts.

usage:
    python -m src.predict --ckpt artifacts/finetune/best.pt --series ./series.npy --horizon 24
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .data import normalize
from .model import ModelConfig, TSForecaster
from .patches import patchify


def load_forecaster(ckpt_path: str, map_location: str = "cpu"
                    ) -> Tuple[TSForecaster, Dict]:
    sd = torch.load(ckpt_path, map_location=map_location)
    cfg_dict = sd["config"]
    cfg = ModelConfig(**cfg_dict)
    n_patches = sd["n_patches"]
    horizon = sd["horizon"]
    model = TSForecaster(cfg, n_patches=n_patches, horizon=horizon)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, {"n_patches": n_patches, "horizon": horizon, "patch_len": cfg.patch_len}


def forecast(model: TSForecaster, series: np.ndarray, context_length: int,
             patch_len: int, device: str = "cpu") -> np.ndarray:
    """
    series: 1D array (history). returns horizon-length forecast in original units.
    """
    if series.ndim != 1:
        raise ValueError("series must be 1D")
    if series.shape[0] < context_length:
        raise ValueError(f"need at least {context_length} points")
    ctx = series[-context_length:]
    ctx_n, mu, sigma = normalize(ctx)
    x = torch.from_numpy(ctx_n[None, :].astype(np.float32)).to(device)
    patches = patchify(x, patch_len, patch_len)
    with torch.no_grad():
        pred = model(patches).cpu().numpy()[0]
    return pred * sigma + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--series", type=str, required=True,
                    help="path to .npy with a 1D series")
    ap.add_argument("--context-length", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, meta = load_forecaster(args.ckpt, map_location=device)
    model = model.to(device)
    series = np.load(args.series).astype(np.float32)
    out = forecast(model, series, args.context_length, meta["patch_len"], device=device)
    print("forecast:", out.tolist())


if __name__ == "__main__":
    main()
