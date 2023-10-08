"""
patch-based transformer encoder for time series foundation model.

PatchTST-ish: each patch is a token. encoder is a stack of transformer
encoder layers. heads are swappable (recon head for pretrain, forecast
head for finetune).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .patches import PatchEmbedding


@dataclass
class ModelConfig:
    patch_len: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    max_patches: int = 64


class TSEncoder(nn.Module):
    """patch transformer encoder."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = PatchEmbedding(
            patch_len=cfg.patch_len,
            d_model=cfg.d_model,
            max_patches=cfg.max_patches,
            dropout=cfg.dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, N, P)
        x = self.embed(patches)            # (B, N, D)
        x = self.encoder(x)                # (B, N, D)
        return self.norm(x)


class ReconstructionHead(nn.Module):
    """maps each token back to a patch (for masked patch reconstruction)."""

    def __init__(self, d_model: int, patch_len: int):
        super().__init__()
        self.proj = nn.Linear(d_model, patch_len)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, D) -> (B, N, P)
        return self.proj(h)


class ForecastingHead(nn.Module):
    """flattens encoder outputs and projects to a forecast horizon."""

    def __init__(self, d_model: int, n_patches: int, horizon: int,
                 dropout: float = 0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model * n_patches, horizon)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, N, D) -> (B, horizon)
        x = self.flatten(h)
        x = self.dropout(x)
        return self.proj(x)


class TSFoundationModel(nn.Module):
    """encoder + recon head; used for pretraining."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.encoder = TSEncoder(cfg)
        self.recon = ReconstructionHead(cfg.d_model, cfg.patch_len)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        h = self.encoder(patches)
        return self.recon(h)


class TSForecaster(nn.Module):
    """encoder + forecasting head; used for finetuning."""

    def __init__(self, cfg: ModelConfig, n_patches: int, horizon: int):
        super().__init__()
        self.encoder = TSEncoder(cfg)
        self.head = ForecastingHead(cfg.d_model, n_patches, horizon,
                                    dropout=cfg.dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        h = self.encoder(patches)
        return self.head(h)

    def load_pretrained_encoder(self, ckpt_path: str,
                                map_location: str = "cpu") -> None:
        sd = torch.load(ckpt_path, map_location=map_location)
        if "encoder" in sd:
            self.encoder.load_state_dict(sd["encoder"])
        else:
            # try loading a flat encoder state dict
            self.encoder.load_state_dict(sd)
