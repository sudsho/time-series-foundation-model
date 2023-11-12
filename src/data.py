"""
synthetic time series generators.

idea: pretrain on a wide mix of synthetic processes so the model sees
diverse temporal structure.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class SeriesConfig:
    length: int = 512
    noise_std: float = 0.1
    seed: Optional[int] = None


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_sine(cfg: SeriesConfig, freq: float = 0.05, amp: float = 1.0,
             phase: float = 0.0) -> np.ndarray:
    rng = _rng(cfg.seed)
    t = np.arange(cfg.length)
    y = amp * np.sin(2 * math.pi * freq * t + phase)
    y = y + rng.normal(0, cfg.noise_std, size=cfg.length)
    return y.astype(np.float32)


def gen_trend(cfg: SeriesConfig, slope: float = 0.01,
              intercept: float = 0.0) -> np.ndarray:
    rng = _rng(cfg.seed)
    t = np.arange(cfg.length)
    y = slope * t + intercept
    y = y + rng.normal(0, cfg.noise_std, size=cfg.length)
    return y.astype(np.float32)


def gen_ar1(cfg: SeriesConfig, phi: float = 0.7) -> np.ndarray:
    """simple AR(1) process."""
    rng = _rng(cfg.seed)
    y = np.zeros(cfg.length, dtype=np.float32)
    for t in range(1, cfg.length):
        y[t] = phi * y[t - 1] + rng.normal(0, cfg.noise_std)
    return y


def gen_seasonal_trend(cfg: SeriesConfig, freq: float = 0.04,
                       slope: float = 0.005) -> np.ndarray:
    s = gen_sine(cfg, freq=freq, amp=1.0)
    t = np.arange(cfg.length, dtype=np.float32) * slope
    return (s + t).astype(np.float32)


def gen_random_walk(cfg: SeriesConfig) -> np.ndarray:
    rng = _rng(cfg.seed)
    steps = rng.normal(0, cfg.noise_std, size=cfg.length)
    return np.cumsum(steps).astype(np.float32)


GENERATORS = {
    "sine": gen_sine,
    "trend": gen_trend,
    "ar1": gen_ar1,
    "seasonal_trend": gen_seasonal_trend,
    "rw": gen_random_walk,
}


def make_synthetic_dataset(n_series: int, length: int = 512,
                           noise_std: float = 0.1,
                           seed: int = 0) -> np.ndarray:
    """build a [n_series, length] array with a mix of processes."""
    rng = _rng(seed)
    series = []
    kinds = list(GENERATORS.keys())
    for i in range(n_series):
        kind = kinds[i % len(kinds)]
        sub_seed = int(rng.integers(0, 1 << 31))
        cfg = SeriesConfig(length=length, noise_std=noise_std, seed=sub_seed)
        if kind == "sine":
            f = float(rng.uniform(0.01, 0.1))
            y = gen_sine(cfg, freq=f, amp=float(rng.uniform(0.5, 2.0)))
        elif kind == "trend":
            y = gen_trend(cfg, slope=float(rng.uniform(-0.02, 0.02)))
        elif kind == "ar1":
            y = gen_ar1(cfg, phi=float(rng.uniform(0.3, 0.95)))
        elif kind == "seasonal_trend":
            y = gen_seasonal_trend(cfg, freq=float(rng.uniform(0.02, 0.08)),
                                   slope=float(rng.uniform(-0.01, 0.01)))
        else:
            y = gen_random_walk(cfg)
        series.append(y)
    return np.stack(series, axis=0)


def normalize(x: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, float, float]:
    """per-series z-score; return normalized + mean + std for inverse."""
    mu = float(x.mean())
    sigma = float(x.std() + eps)
    return ((x - mu) / sigma).astype(np.float32), mu, sigma
