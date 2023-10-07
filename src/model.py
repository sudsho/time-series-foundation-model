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
    n_heads: 4
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
