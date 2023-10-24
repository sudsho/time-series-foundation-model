import pytest
import torch

from src.masking import apply_mask, random_patch_mask
from src.patches import PatchEmbedding, patchify, unpatchify


def test_patchify_shape():
    x = torch.randn(3, 256)
    p = patchify(x, patch_len=16, stride=16)
    assert p.shape == (3, 16, 16)


def test_patchify_strided_overlap():
    x = torch.randn(2, 64)
    p = patchify(x, patch_len=16, stride=8)
    # N = (64 - 16) / 8 + 1 = 7
    assert p.shape == (2, 7, 16)


def test_patchify_rejects_short_seq():
    with pytest.raises(ValueError):
        patchify(torch.randn(1, 8), patch_len=16, stride=16)


def test_patchify_rejects_wrong_dim():
    with pytest.raises(ValueError):
        patchify(torch.randn(8), patch_len=4, stride=4)


def test_unpatchify_roundtrip_non_overlapping():
    x = torch.randn(2, 64)
    p = patchify(x, 16, 16)
    back = unpatchify(p, stride=16)
    assert torch.allclose(back, x)


def test_unpatchify_rejects_overlap():
    p = torch.randn(2, 7, 16)
    with pytest.raises(ValueError):
        unpatchify(p, stride=8)


def test_patch_embedding_shape():
    emb = PatchEmbedding(patch_len=16, d_model=32, max_patches=64, dropout=0.0)
    x = torch.randn(4, 16, 16)
    out = emb(x)
    assert out.shape == (4, 16, 32)


def test_patch_embedding_pos_added():
    torch.manual_seed(0)
    emb = PatchEmbedding(patch_len=8, d_model=16, max_patches=32, dropout=0.0)
    # the same patch at different positions should give different embeddings
    p = torch.randn(1, 1, 8)
    p2 = torch.cat([p, p], dim=1)  # (1, 2, 8)
    out = emb(p2)
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])


def test_random_patch_mask_count():
    torch.manual_seed(0)
    mask = random_patch_mask(B=8, N=20, mask_ratio=0.5, device=torch.device("cpu"))
    # exactly floor(0.5 * 20) = 10 masked per row
    counts = mask.sum(dim=1)
    assert (counts == 10).all()


def test_apply_mask_zeros_correctly():
    patches = torch.ones(2, 4, 3)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 1] = True
    mask[1, 3] = True
    out = apply_mask(patches, mask, mask_value=0.0)
    assert (out[0, 1] == 0).all()
    assert (out[1, 3] == 0).all()
    assert (out[0, 0] == 1).all()
