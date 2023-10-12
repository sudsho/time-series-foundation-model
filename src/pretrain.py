"""
pretraining loop: masked patch reconstruction (MPR).

we generate a big synthetic mix, patch each window, mask 40% of patches,
predict the masked patch values, MSE loss on masked positions only.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .data import make_synthetic_dataset, normalize
from .masking import apply_mask, random_patch_mask
from .model import ModelConfig, TSFoundationModel
from .patches import patchify


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_dataset(cfg: Dict) -> TensorDataset:
    dcfg = cfg["data"]
    arr = make_synthetic_dataset(
        n_series=dcfg["num_synthetic_series"],
        length=dcfg["series_length"],
        noise_std=dcfg["noise_std"],
        seed=cfg.get("seed", 0),
    )
    # normalize per series
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        out[i], _, _ = normalize(arr[i])
    L = dcfg["context_length"]
    # take last L points of each series as the training window
    windows = out[:, -L:]
    x = torch.from_numpy(windows.astype(np.float32))
    return TensorDataset(x)


def make_model(cfg: Dict, n_patches: int) -> TSFoundationModel:
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    model_cfg = ModelConfig(
        patch_len=dcfg["patch_len"],
        d_model=mcfg["d_model"],
        n_heads=mcfg["n_heads"],
        n_layers=mcfg["n_layers"],
        ffn_dim=mcfg["ffn_dim"],
        dropout=mcfg["dropout"],
        max_patches=max(n_patches, 64),
    )
    return TSFoundationModel(model_cfg)


def train_one_epoch(model, loader, optim, mask_ratio: float, device,
                    log_every: int = 50) -> float:
    model.train()
    total = 0.0
    n = 0
    for step, (xb,) in enumerate(loader):
        xb = xb.to(device)  # (B, L)
        patches = patchify(xb, patch_len=model.encoder.cfg.patch_len,
                           stride=model.encoder.cfg.patch_len)
        B, N, P = patches.shape
        mask = random_patch_mask(B, N, mask_ratio, device=device)
        masked_in = apply_mask(patches, mask)
        pred = model(masked_in)             # (B, N, P)
        # loss only on masked positions
        loss = F.mse_loss(pred[mask], patches[mask])
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += loss.item() * B
        n += B
        if step % log_every == 0:
            print(f"  step {step}: loss={loss.item():.4f}")
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = build_dataset(cfg)
    loader = DataLoader(ds, batch_size=cfg["pretrain"]["batch_size"],
                        shuffle=True, drop_last=True)

    n_patches = cfg["data"]["context_length"] // cfg["data"]["patch_len"]
    model = make_model(cfg, n_patches=n_patches).to(device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["pretrain"]["lr"],
        weight_decay=cfg["pretrain"]["weight_decay"],
    )

    ckpt_dir = Path(cfg["pretrain"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    for epoch in range(cfg["pretrain"]["epochs"]):
        loss = train_one_epoch(
            model, loader, optim,
            mask_ratio=cfg["pretrain"]["mask_ratio"],
            device=device,
            log_every=cfg["logging"]["log_every"],
        )
        print(f"epoch {epoch}: avg_loss={loss:.4f}")
        if loss < best:
            best = loss
            torch.save({"encoder": model.encoder.state_dict(),
                        "config": asdict(model.encoder.cfg)},
                       ckpt_dir / "best.pt")


if __name__ == "__main__":
    main()
