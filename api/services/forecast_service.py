"""Forecast service — loads pre-trained NeuralForecast model and predicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from api.schemas.forecast import ConformalInterval, ForecastResponse


def run_forecast(
    nf_model: Any,
    df: pd.DataFrame,
    country_iso: str,
    horizon: int,
) -> ForecastResponse:
    country_df = df[df["iso_code"] == country_iso].sort_values("year").copy()
    country_df = country_df.dropna(subset=["co2"])

    if len(country_df) < 5:
        raise ValueError(f"Insufficient data for {country_iso}: {len(country_df)} points")

    # Build NeuralForecast input format
    nf_df = pd.DataFrame(
        {
            "unique_id": country_iso,
            "ds": pd.to_datetime(country_df["year"], format="%Y"),
            "y": country_df["co2"].values,
        }
    )
    for cov in ["gdp", "primary_energy_consumption", "population"]:
        if cov in country_df.columns:
            nf_df[cov] = country_df[cov].ffill().fillna(0).values

    # Predict using pre-trained model (fixed horizon from training)
    # Force CPU on HF Space CPU Basic by hiding CUDA from PyTorch Lightning
    import torch as _torch

    _original_cuda_available = _torch.cuda.is_available
    _torch.cuda.is_available = lambda: False
    try:
        forecast_df = nf_model.predict(df=nf_df)
    finally:
        _torch.cuda.is_available = _original_cuda_available
    predictions = forecast_df["NHITS"].tolist()

    # Cap at requested horizon
    predictions = predictions[:horizon]
    actual_horizon = len(predictions)

    last_year = int(country_df["year"].max())
    prediction_years = list(range(last_year + 1, last_year + 1 + actual_horizon))

    # Conformal prediction intervals
    from co2_ml.models.forecast import apply_conformal_intervals

    y_train = country_df["co2"].values
    y_pred = np.array(predictions)
    try:
        ci = apply_conformal_intervals(y_pred, y_train, alpha=0.1)
        intervals = ConformalInterval(
            lower=ci["lower"],
            upper=ci["upper"],
            coverage=ci["coverage"],
        )
    except Exception:
        intervals = None

    return ForecastResponse(
        country=country_iso,
        model="nhits",
        horizon=actual_horizon,
        predictions=predictions,
        prediction_years=prediction_years,
        intervals=intervals,
    )
