import math

import numpy as np
import pytest

from src.metrics import mase, smape


def test_smape_zero_when_perfect():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert smape(y, y) == 0.0


def test_smape_handles_zeros():
    # both zero in a position -> contribution should be 0 (not NaN)
    y_true = np.array([0.0, 1.0])
    y_pred = np.array([0.0, 1.0])
    assert smape(y_true, y_pred) == 0.0


def test_smape_symmetric():
    a = np.array([3.0, 4.0])
    b = np.array([4.0, 3.0])
    # smape should be symmetric in y_true / y_pred
    assert math.isclose(smape(a, b), smape(b, a), rel_tol=1e-9)


def test_smape_in_range():
    rng = np.random.default_rng(0)
    y_true = rng.normal(10, 2, size=50)
    y_pred = rng.normal(10, 2, size=50)
    v = smape(y_true, y_pred)
    assert 0.0 <= v <= 200.0


def test_mase_perfect_prediction_is_zero():
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    y_true = np.array([9.0, 10.0])
    y_pred = y_true.copy()
    assert mase(y_true, y_pred, y_train, seasonality=1) == 0.0


def test_mase_constant_train_returns_nan():
    # when in-sample naive error is 0, MASE is undefined -> NaN
    y_train = np.ones(20)
    y_true = np.array([1.5, 1.5])
    y_pred = np.array([1.6, 1.4])
    assert math.isnan(mase(y_true, y_pred, y_train, seasonality=1))


def test_mase_short_train_returns_nan():
    y_train = np.array([1.0])
    y_true = np.array([2.0])
    y_pred = np.array([1.0])
    assert math.isnan(mase(y_true, y_pred, y_train, seasonality=1))


def test_mase_seasonal_scale_uses_seasonality():
    # build a strict period-4 series; with seasonality=4 naive error is 0
    base = np.tile(np.array([1.0, 2.0, 3.0, 4.0]), 5)
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.5, 2.5])
    out = mase(y_true, y_pred, base, seasonality=4)
    assert math.isnan(out)


def test_smape_pred_double_truth():
    # known: y_true=[1], y_pred=[3] -> diff=2, denom=2, ratio=1.0, smape=100
    y_true = np.array([1.0])
    y_pred = np.array([3.0])
    assert math.isclose(smape(y_true, y_pred), 100.0, rel_tol=1e-9)


def test_smape_accepts_lists():
    out = smape([1.0, 2.0], [1.0, 2.0])
    assert out == 0.0


def test_mase_large_error():
    y_train = np.arange(10, dtype=np.float64)
    y_true = np.array([10.0, 11.0])
    # if predictions are way off, MASE should be > 1
    y_pred = np.array([0.0, 0.0])
    assert mase(y_true, y_pred, y_train, seasonality=1) > 1.0
