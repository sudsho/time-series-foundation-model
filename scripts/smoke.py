"""
Offline tiny-CPU smoke for the time-series foundation model.

Proves the full patch-transformer pipeline runs end to end on CPU with no
downloads, no GPU, no pretrained weights, and no external services:

  1. build a tiny deterministic synthetic dataset (seasonal + trend), no I/O
  2. assert the patchify + patch-embedding shapes are correct
  3. masked-patch-reconstruction (MPR) pretrain a TINY encoder for a few steps
     and check the reconstruction MSE goes down
  4. attach a forecasting head, load the pretrained encoder, finetune a few
     steps and check the forecasting MSE goes down
  5. forecast a held-out series through src.predict.forecast and print
     predicted vs actual plus sMAPE / MASE
  6. assert the output shapes at every stage

Run:
    python scripts/smoke.py

The real project pretrains a bigger model (d_model=128, 4 layers) on a large
synthetic mix and finetunes on M4-style series, ideally on a GPU. This smoke
uses a deliberately tiny model and dataset so it finishes in a couple of
seconds on a laptop CPU while exercising the exact same code paths.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# make `src` importable when run as `python scripts/smoke.py` from the repo root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import SeriesConfig, gen_seasonal_trend, normalize
from src.masking import apply_mask, random_patch_mask
from src.metrics import mase, smape
from src.model import ModelConfig, TSForecaster, TSFoundationModel
from src.patches import PatchEmbedding, patchify
from src.predict import forecast

# ----------------------------------------------------------------------------
# tiny, deterministic setup
# ----------------------------------------------------------------------------
SEED = 0
CONTEXT_LENGTH = 64
PATCH_LEN = 8
HORIZON = 8
N_PATCHES = CONTEXT_LENGTH // PATCH_LEN  # 8
N_SERIES = 96
SERIES_LEN = CONTEXT_LENGTH + HORIZON  # 72


def build_seasonal_dataset(n_series: int, length: int, seed: int) -> np.ndarray:
    """A deterministic bank of seasonal+trend series with varied period/phase.

    Purely synthetic and generated in-process: no files, no network.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_series):
        cfg = SeriesConfig(length=length, noise_std=0.02,
                           seed=int(rng.integers(0, 1 << 31)))
        freq = float(rng.uniform(0.05, 0.15))
        slope = float(rng.uniform(-0.01, 0.01))
        y = gen_seasonal_trend(cfg, freq=freq, slope=slope)
        rows.append(y.astype(np.float32))
    return np.stack(rows, axis=0)


def make_supervised(arr: np.ndarray):
    """Split each series into a normalized context window and target horizon.

    Mirrors src.finetune.build_supervised: normalize the context, apply the
    same (mu, sigma) to the target so the model predicts in normalized units.
    """
    X, Y = [], []
    for s in arr:
        ctx = s[:CONTEXT_LENGTH]
        tgt = s[CONTEXT_LENGTH:CONTEXT_LENGTH + HORIZON]
        ctx_n, mu, sigma = normalize(ctx)
        tgt_n = ((tgt - mu) / sigma).astype(np.float32)
        X.append(ctx_n)
        Y.append(tgt_n)
    return (torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y)))


def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        patch_len=PATCH_LEN,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_dim=64,
        dropout=0.0,
        max_patches=max(N_PATCHES, 8),
    )


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    print("=" * 62)
    print("time-series-foundation-model :: offline tiny-CPU smoke")
    print("=" * 62)
    print(f"device={device.type}  context_length={CONTEXT_LENGTH}  "
          f"patch_len={PATCH_LEN}  n_patches={N_PATCHES}  horizon={HORIZON}")

    # ---- 1. data -----------------------------------------------------------
    arr = build_seasonal_dataset(N_SERIES, SERIES_LEN, seed=SEED)
    # hold out the last 8 series for evaluation / a single forecast demo
    train_arr, held_arr = arr[:-8], arr[-8:]
    X, Y = make_supervised(train_arr)
    print(f"\n[1] synthetic seasonal+trend dataset: {arr.shape} "
          f"(train={train_arr.shape[0]}, held={held_arr.shape[0]})")

    cfg = tiny_cfg()
    n_params = sum(p.numel()
                   for p in TSForecaster(cfg, N_PATCHES, HORIZON).parameters())
    print(f"    tiny model: d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"n_heads={cfg.n_heads} -> {n_params:,} params")

    # ---- 2. shape assertions on patchify + patch embedding -----------------
    patches = patchify(X, PATCH_LEN, PATCH_LEN)
    assert patches.shape == (X.shape[0], N_PATCHES, PATCH_LEN), patches.shape
    emb = PatchEmbedding(PATCH_LEN, cfg.d_model, cfg.max_patches, dropout=0.0)
    tokens = emb(patches)
    assert tokens.shape == (X.shape[0], N_PATCHES, cfg.d_model), tokens.shape
    print(f"\n[2] patchify  {tuple(X.shape)} -> {tuple(patches.shape)}  (B,N,P)  OK")
    print(f"    embed     {tuple(patches.shape)} -> {tuple(tokens.shape)}  (B,N,D)  OK")

    # ---- 3. MPR pretrain (few steps): reconstruction MSE should drop -------
    pre = TSFoundationModel(cfg).to(device)
    opt = torch.optim.AdamW(pre.parameters(), lr=1e-3)
    mask_ratio = 0.4
    pre.train()
    first_recon = last_recon = None
    for step in range(60):
        p = patchify(X, PATCH_LEN, PATCH_LEN).to(device)
        mask = random_patch_mask(p.shape[0], p.shape[1], mask_ratio, device)
        out = pre(apply_mask(p, mask))
        assert out.shape == p.shape, out.shape
        loss = F.mse_loss(out[mask], p[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first_recon = loss.item()
        last_recon = loss.item()
    print(f"\n[3] MPR pretrain (masked-patch recon): "
          f"MSE {first_recon:.4f} -> {last_recon:.4f}")
    assert last_recon < first_recon, "recon loss did not decrease"

    # transfer the pretrained encoder into a forecaster
    ckpt = ROOT / "artifacts" / "smoke" / "encoder.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": pre.encoder.state_dict()}, ckpt)

    # ---- 4. finetune the forecasting head: forecast MSE should drop --------
    model = TSForecaster(cfg, n_patches=N_PATCHES, horizon=HORIZON).to(device)
    model.load_pretrained_encoder(str(ckpt), map_location="cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    Xd, Yd = X.to(device), Y.to(device)
    model.train()
    first_fc = last_fc = None
    for step in range(120):
        pred = model(patchify(Xd, PATCH_LEN, PATCH_LEN))
        assert pred.shape == (Xd.shape[0], HORIZON), pred.shape
        loss = F.mse_loss(pred, Yd)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first_fc = loss.item()
        last_fc = loss.item()
    print(f"\n[4] forecast finetune (MSE on {HORIZON}-step horizon): "
          f"{first_fc:.4f} -> {last_fc:.4f}")
    assert last_fc < first_fc, "forecast loss did not decrease"
    assert last_fc < 0.5 * first_fc, "forecast loss barely moved"

    # save a finetune checkpoint in the format src.predict expects
    fc_ckpt = ROOT / "artifacts" / "smoke" / "forecaster.pt"
    torch.save({"state_dict": model.state_dict(),
                "config": {
                    "patch_len": cfg.patch_len, "d_model": cfg.d_model,
                    "n_heads": cfg.n_heads, "n_layers": cfg.n_layers,
                    "ffn_dim": cfg.ffn_dim, "dropout": cfg.dropout,
                    "max_patches": cfg.max_patches},
                "n_patches": N_PATCHES, "horizon": HORIZON}, fc_ckpt)

    # ---- 5. forecast a single held-out series ------------------------------
    model.eval()
    held = held_arr[0]                       # full length CONTEXT_LENGTH+HORIZON
    history = held[:CONTEXT_LENGTH]
    actual = held[CONTEXT_LENGTH:CONTEXT_LENGTH + HORIZON]
    pred = forecast(model, history, CONTEXT_LENGTH, PATCH_LEN, device="cpu")
    assert pred.shape == (HORIZON,), pred.shape
    sm = smape(actual, pred)
    ms = mase(actual, pred, history, seasonality=1)
    print(f"\n[5] forecast on a held-out series (predicted vs actual):")
    print(f"    idx |  actual  | predicted")
    for i in range(HORIZON):
        print(f"    {i:>3} | {actual[i]:>8.4f} | {pred[i]:>8.4f}")
    print(f"    sMAPE={sm:.3f}%   MASE={ms:.3f}")

    # ---- 6. aggregate held-out metric --------------------------------------
    sms = []
    for s in held_arr:
        h, a = s[:CONTEXT_LENGTH], s[CONTEXT_LENGTH:CONTEXT_LENGTH + HORIZON]
        sms.append(smape(a, forecast(model, h, CONTEXT_LENGTH, PATCH_LEN)))
    print(f"\n[6] mean sMAPE over {len(held_arr)} held-out series: "
          f"{float(np.mean(sms)):.3f}%")

    print("\n" + "=" * 62)
    print("SMOKE PASSED: pipeline runs offline, losses decrease, shapes OK")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
