"""
forecasting metrics: MASE and sMAPE.

MASE = mean abs error / mean abs naive seasonal error
sMAPE = symmetric MAPE
"""
from __future__ import annotations

import numpy as np


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """symmetric MAPE in percent (0..200)."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    out = np.where(denom == 0, 0.0, diff / denom)
    return float(100.0 * out.mean())


def mase(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray,
         seasonality: int = 1) -> float:
    """
    Mean absolute scaled error.
    y_train is the in-sample series used to compute the naive seasonal scale.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.size <= seasonality:
        return float("nan")
    naive_err = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    if naive_err == 0:
        return float("nan")
    err = np.mean(np.abs(y_true - y_pred))
    return float(err / naive_err)
