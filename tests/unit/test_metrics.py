"""Tests for evaluation metrics."""

import numpy as np
import pytest
from co2_ml.utils.metrics import mase, smape


def test_mase_perfect_forecast_returns_zero() -> None:
    """A perfect forecast should have MASE = 0."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mase(y, y) == 0.0


def test_mase_naive_returns_one() -> None:
    """A naive forecast (shifted by 1) should have MASE = 1."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Naive forecast: predict previous value → error = [1,1,1,1] same as naive MAE
    y_pred = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    result = mase(y_true, y_pred)
    assert abs(result - 1.0) < 1e-10, f"Expected MASE=1.0, got {result}"


def test_smape_perfect_forecast_returns_zero() -> None:
    """A perfect forecast should have sMAPE = 0."""
    y = np.array([1.0, 2.0, 3.0])
    assert smape(y, y) == 0.0


def test_smape_range() -> None:
    """sMAPE should be between 0 and 200."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([3.0, 1.0, 5.0])
    result = smape(y_true, y_pred)
    assert 0 <= result <= 200, f"sMAPE out of range: {result}"
