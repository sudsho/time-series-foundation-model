"""
random patch masking utilities for masked patch reconstruction (MPR).
"""
from __future__ import annotations

import torch


def random_patch_mask(B: int, N: int, mask_ratio: float,
                      device: torch.device) -> torch.Tensor:
    """
    return a bool mask of shape (B, N) where True = masked.
    each row gets exactly floor(mask_ratio * N) masked positions.
    """
    if not (0.0 <= mask_ratio <= 1.0):
        raise ValueError(f"mask_ratio out of range: {mask_ratio}")
    n_mask = int(mask_ratio * N)
    if n_mask == 0:
        return torch.zeros(B, N, dtype=torch.bool, device=device)

    noise = torch.rand(B, N, device=device)
    # take the indices of the n_mask smallest values per row.
    # this guarantees exactly n_mask masked positions per row, even when
    # there are duplicate values (rand collisions are very rare but can
    # happen and made the previous kthvalue+threshold approach flaky).
    idx = noise.argsort(dim=1)[:, :n_mask]
    mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    mask.scatter_(1, idx, True)
    return mask


def apply_mask(patches: torch.Tensor, mask: torch.Tensor,
               mask_value: float = 0.0) -> torch.Tensor:
    """
    patches: (B, N, P)
    mask:    (B, N) bool, True = masked
    returns: same shape with masked patches set to mask_value
    """
    out = patches.clone()
    out[mask] = mask_value
    return out
