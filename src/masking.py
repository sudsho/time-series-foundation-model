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
    # smallest n_mask values per row are masked
    threshold = noise.kthvalue(n_mask, dim=1, keepdim=True).values
    mask = noise <= threshold
    # ensure exactly n_mask (kthvalue ties handled coarsely)
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
