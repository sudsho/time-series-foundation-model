"""
small M4 subset loader.

if real M4 CSVs are not present locally, fall back to a synthetic surrogate
that mimics the hourly/daily structure (so eval pipelines can still run).
real M4 data: https://github.com/Mcompetitions/M4-methods/tree/master/Dataset
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data import SeriesConfig, gen_seasonal_trend, gen_sine


def _synthetic_m4_like(n_series: int, length: int, seasonality: int,
                       seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_series):
        cfg = SeriesConfig(length=length, noise_std=0.2,
                           seed=int(rng.integers(0, 1 << 31)))
        freq = 1.0 / max(seasonality, 2)
        s = gen_seasonal_trend(cfg, freq=freq,
                               slope=float(rng.uniform(-0.005, 0.01)))
        # add a multiplicative scale to look more like M4
        scale = float(rng.uniform(5.0, 20.0))
        out.append((s * scale + scale).astype(np.float32))
    return out


def load_m4_subset(name: str, data_dir: Optional[str] = None,
                   max_series: int = 200) -> Tuple[List[np.ndarray], int]:
    """
    name: 'Hourly' | 'Daily'
    returns: list of 1D series arrays, seasonality
    """
    season_map = {"Hourly": 24, "Daily": 1, "Weekly": 1, "Monthly": 12}
    if name not in season_map:
        raise ValueError(f"unsupported subset {name}")
    seasonality = season_map[name]

    if data_dir:
        train_path = Path(data_dir) / f"{name}-train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path)
            series = []
            for _, row in df.head(max_series).iterrows():
                vals = row.dropna().to_numpy()
                # first column is series id
                vals = vals[1:].astype(np.float32)
                series.append(vals)
            return series, seasonality

    # fallback synthetic surrogate
    length = 700 if name == "Hourly" else 200
    return (_synthetic_m4_like(max_series, length=length,
                               seasonality=seasonality, seed=7),
            seasonality)


def split_train_test(series: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    if series.shape[0] <= horizon:
        raise ValueError("series too short for the requested horizon")
    return series[:-horizon], series[-horizon:]
