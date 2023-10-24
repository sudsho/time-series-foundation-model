import numpy as np

from src.data import (
    SeriesConfig,
    gen_ar1,
    gen_random_walk,
    gen_seasonal_trend,
    gen_sine,
    gen_trend,
    make_synthetic_dataset,
    normalize,
)


def test_sine_shape_and_dtype():
    cfg = SeriesConfig(length=128, noise_std=0.0, seed=0)
    y = gen_sine(cfg, freq=0.05)
    assert y.shape == (128,)
    assert y.dtype == np.float32


def test_trend_is_monotonic_when_no_noise():
    cfg = SeriesConfig(length=64, noise_std=0.0, seed=1)
    y = gen_trend(cfg, slope=0.05, intercept=1.0)
    assert np.all(np.diff(y) > 0)


def test_ar1_runs():
    cfg = SeriesConfig(length=200, noise_std=0.05, seed=2)
    y = gen_ar1(cfg, phi=0.6)
    assert y.shape == (200,)
    assert np.isfinite(y).all()


def test_random_walk_starts_at_step():
    cfg = SeriesConfig(length=10, noise_std=0.0, seed=3)
    y = gen_random_walk(cfg)
    # zero noise -> all zeros after cumsum
    assert np.allclose(y, 0.0)


def test_seasonal_trend_finite():
    cfg = SeriesConfig(length=128, noise_std=0.1, seed=4)
    y = gen_seasonal_trend(cfg, freq=0.05, slope=0.01)
    assert np.isfinite(y).all()


def test_make_synthetic_dataset_shape():
    arr = make_synthetic_dataset(n_series=20, length=128, noise_std=0.1, seed=0)
    assert arr.shape == (20, 128)
    assert arr.dtype == np.float32


def test_normalize_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 2.0, size=512).astype(np.float32)
    n, mu, sigma = normalize(x)
    assert abs(n.mean()) < 1e-3
    assert abs(n.std() - 1.0) < 1e-2
    assert abs(mu - 5.0) < 0.5
    assert abs(sigma - 2.0) < 0.5
