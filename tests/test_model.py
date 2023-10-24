import torch

from src.model import (
    ForecastingHead,
    ModelConfig,
    ReconstructionHead,
    TSEncoder,
    TSForecaster,
    TSFoundationModel,
)


def _cfg():
    return ModelConfig(patch_len=16, d_model=32, n_heads=4, n_layers=2,
                       ffn_dim=64, dropout=0.0, max_patches=32)


def test_encoder_shape():
    cfg = _cfg()
    enc = TSEncoder(cfg)
    x = torch.randn(2, 16, 16)
    out = enc(x)
    assert out.shape == (2, 16, 32)


def test_recon_head_shape():
    head = ReconstructionHead(d_model=32, patch_len=16)
    h = torch.randn(2, 16, 32)
    out = head(h)
    assert out.shape == (2, 16, 16)


def test_forecast_head_shape():
    head = ForecastingHead(d_model=32, n_patches=16, horizon=24, dropout=0.0)
    h = torch.randn(3, 16, 32)
    out = head(h)
    assert out.shape == (3, 24)


def test_foundation_model_forward():
    cfg = _cfg()
    m = TSFoundationModel(cfg)
    x = torch.randn(4, 16, 16)
    out = m(x)
    assert out.shape == (4, 16, 16)


def test_forecaster_forward():
    cfg = _cfg()
    m = TSForecaster(cfg, n_patches=16, horizon=24)
    x = torch.randn(2, 16, 16)
    out = m(x)
    assert out.shape == (2, 24)


def test_forecaster_loads_pretrained_encoder(tmp_path):
    cfg = _cfg()
    pre = TSFoundationModel(cfg)
    ckpt = tmp_path / "enc.pt"
    torch.save({"encoder": pre.encoder.state_dict()}, ckpt)

    fc = TSForecaster(cfg, n_patches=16, horizon=8)
    fc.load_pretrained_encoder(str(ckpt))
    # weights should match
    for k, v in pre.encoder.state_dict().items():
        assert torch.allclose(fc.encoder.state_dict()[k], v)


def test_param_count_small():
    cfg = ModelConfig(patch_len=16, d_model=128, n_heads=4, n_layers=4,
                      ffn_dim=256, dropout=0.1, max_patches=64)
    m = TSFoundationModel(cfg)
    n_params = sum(p.numel() for p in m.parameters())
    # sanity: small model, well under 5M params
    assert n_params < 5_000_000
