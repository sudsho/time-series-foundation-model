"""
finetune the pretrained encoder for forecasting.

approach: take last `context_length` points -> patches -> encoder ->
forecast head -> horizon predictions. eval with MASE/sMAPE on holdout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

from .data import normalize
from .m4_mini import load_m4_subset, split_train_test
from .metrics import mase, smape
from .model import ModelConfig, TSForecaster
from .patches import patchify


def load_config(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_supervised(series_list: List[np.ndarray], context_length: int,
                     horizon: int) -> Tuple[torch.Tensor, torch.Tensor,
                                            List[Tuple[float, float]]]:
    """
    build (X, y) tensors of shape (N, L) and (N, H).
    also returns per-series (mu, sigma) used for normalization, in order.
    """
    X, Y, stats = [], [], []
    for s in series_list:
        if s.shape[0] < context_length + horizon:
            continue
        train_part, test_part = split_train_test(s, horizon)
        ctx = train_part[-context_length:]
        ctx_n, mu, sigma = normalize(ctx)
        target_n = ((test_part - mu) / sigma).astype(np.float32)
        X.append(ctx_n)
        Y.append(target_n)
        stats.append((mu, sigma))
    return (torch.from_numpy(np.stack(X)),
            torch.from_numpy(np.stack(Y)),
            stats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/finetune.yaml")
    ap.add_argument("--data-dir", type=str, default=None,
                    help="optional path to M4 train CSVs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dcfg = cfg["data"]
    fcfg = cfg["finetune"]
    L = dcfg["context_length"]
    H = dcfg["horizon"]

    # build dataset across requested subsets
    all_X, all_Y, all_stats, all_train = [], [], [], []
    for sub in dcfg["m4_subset"]:
        series, _seas = load_m4_subset(sub, data_dir=args.data_dir,
                                       max_series=dcfg["max_series"])
        X, Y, stats = build_supervised(series, L, H)
        all_X.append(X)
        all_Y.append(Y)
        all_stats.extend(stats)
        all_train.extend(series)
    X = torch.cat(all_X, dim=0)
    Y = torch.cat(all_Y, dim=0)

    n = X.shape[0]
    n_val = max(1, n // 5)
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr, Y_tr = X[tr_idx], Y[tr_idx]
    X_va, Y_va = X[val_idx], Y[val_idx]

    n_patches = L // dcfg["patch_len"]
    model_cfg = ModelConfig(
        patch_len=dcfg["patch_len"],
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ffn_dim=cfg["model"]["ffn_dim"],
        dropout=cfg["model"]["dropout"],
        max_patches=max(n_patches, 64),
    )
    model = TSForecaster(model_cfg, n_patches=n_patches, horizon=H).to(device)

    pre_ckpt = fcfg.get("pretrained_ckpt")
    if pre_ckpt and Path(pre_ckpt).exists():
        print(f"loading pretrained encoder from {pre_ckpt}")
        model.load_pretrained_encoder(pre_ckpt, map_location=str(device))

    if fcfg.get("freeze_encoder"):
        for p in model.encoder.parameters():
            p.requires_grad = False

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=fcfg["lr"], weight_decay=fcfg["weight_decay"],
    )

    tr_loader = DataLoader(TensorDataset(X_tr, Y_tr),
                           batch_size=fcfg["batch_size"], shuffle=True)
    va_loader = DataLoader(TensorDataset(X_va, Y_va),
                           batch_size=fcfg["batch_size"])

    ckpt_dir = Path(fcfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    bad = 0
    patience = fcfg.get("early_stopping_patience", 5)
    for epoch in range(fcfg["epochs"]):
        model.train()
        tr_total = 0.0
        for xb, yb in tr_loader:
            xb = xb.to(device); yb = yb.to(device)
            patches = patchify(xb, dcfg["patch_len"], dcfg["patch_len"])
            pred = model(patches)
            loss = F.mse_loss(pred, yb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            tr_total += loss.item() * xb.size(0)
        tr_avg = tr_total / max(len(X_tr), 1)

        model.eval()
        va_total = 0.0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device); yb = yb.to(device)
                patches = patchify(xb, dcfg["patch_len"], dcfg["patch_len"])
                pred = model(patches)
                va_total += F.mse_loss(pred, yb).item() * xb.size(0)
        va_avg = va_total / max(len(X_va), 1)
        print(f"epoch {epoch}: train={tr_avg:.4f} val={va_avg:.4f}")
        if va_avg < best_val:
            best_val = va_avg
            bad = 0
            torch.save(
                {"state_dict": model.state_dict(),
                 "config": asdict(model_cfg),
                 "n_patches": n_patches,
                 "horizon": H},
                ckpt_dir / "best.pt",
            )
        else:
            bad += 1
            if bad > patience:
                print("early stop")
                break

    # final eval on validation in original units
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for xb, yb in va_loader:
            xb = xb.to(device); yb = yb.to(device)
            patches = patchify(xb, dcfg["patch_len"], dcfg["patch_len"])
            pred = model(patches).cpu().numpy()
            preds.append(pred)
            targets.append(yb.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    smapes = []
    mases = []
    val_stats = [all_stats[i] for i in val_idx.tolist()]
    val_train = [all_train[i] for i in val_idx.tolist()]
    for i in range(preds.shape[0]):
        mu, sigma = val_stats[i]
        yt = targets[i] * sigma + mu
        yp = preds[i] * sigma + mu
        smapes.append(smape(yt, yp))
        mases.append(mase(yt, yp, val_train[i][:-H], seasonality=1))
    print(f"val sMAPE={np.mean(smapes):.3f}, MASE={np.nanmean(mases):.3f}")


if __name__ == "__main__":
    main()
