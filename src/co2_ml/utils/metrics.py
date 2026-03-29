"""Evaluation metrics: MASE, sMAPE, and custom scoring functions."""

from __future__ import annotations

import numpy as np


def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Scaled Error — scale-independent, ideal for cross-country comparison.

    MASE = mean(|y_true - y_pred|) / mean(|y_true[t] - y_true[t-1]|)
    A MASE < 1 means the forecast is better than a naive (random walk) baseline.
    A MASE = 1 means it's equal to the naive baseline.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n = len(y_true)
    if n < 2:
        return float("nan")

    mae = np.mean(np.abs(y_true - y_pred))
    naive_mae = np.mean(np.abs(np.diff(y_true)))

    if naive_mae == 0:
        return float("inf") if mae > 0 else 0.0

    return float(mae / naive_mae)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (0-200 scale).

    sMAPE = 100/n * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero
    mask = denominator > 0
    if not mask.any():
        return 0.0

    return float(100.0 / len(y_true) * np.sum(2 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))
