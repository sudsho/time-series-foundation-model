"""
patch embedding for time series.

splits a univariate sequence (B, L) into non-overlapping patches
(B, N, P) and projects each patch to a d_model embedding.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    x: (B, L) univariate
    returns: (B, N, P) where N = (L - P)/stride + 1
    """
    if x.dim() != 2:
        raise ValueError(f"expected (B, L), got {tuple(x.shape)}")
    B, L = x.shape
    if L < patch_len:
        raise ValueError(f"L={L} < patch_len={patch_len}")
    # use unfold along last dim
    p = x.unfold(dimension=1, size=patch_len, step=stride)  # (B, N, P)
    return p.contiguous()


def unpatchify(patches: torch.Tensor, stride: int) -> torch.Tensor:
    """
    invert patchify when stride == patch_len (non-overlapping).
    patches: (B, N, P)
    returns: (B, N*P)
    """
    B, N, P = patches.shape
    if stride != P:
        raise ValueError("unpatchify only supports non-overlapping patches")
    return patches.reshape(B, N * P)


class PatchEmbedding(nn.Module):
    """linear projection of (P,) patches to d_model + learned position embed."""

    def __init__(self, patch_len: int, d_model: int, max_patches: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.d_model = d_model
        self.proj = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # patches: (B, N, P)
        B, N, P = patches.shape
        if P != self.patch_len:
            raise ValueError(f"expected P={self.patch_len}, got {P}")
        if N > self.pos_embed.size(1):
            raise ValueError(f"N={N} exceeds max_patches={self.pos_embed.size(1)}")
        x = self.proj(patches)  # (B, N, d_model)
        x = x + self.pos_embed[:, :N, :]
        return self.dropout(x)
