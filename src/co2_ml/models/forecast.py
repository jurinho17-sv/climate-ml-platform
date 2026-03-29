"""Time-series forecasting: Chronos-2 (zero-shot) + N-HiTS (fine-tuned)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "model"


def _load_config(name: str) -> dict:
    with open(CONFIGS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def run_chronos_zeroshot(
    df: pd.DataFrame,
    country_iso: str,
    horizon: int = 10,
) -> dict:
    """Zero-shot forecast using Chronos-2 foundation model.

    Args:
        df: ML-ready dataframe with columns [iso_code, year, co2].
        country_iso: ISO 3166-1 alpha-3 country code.
        horizon: Number of years to forecast.

    Returns:
        Dict with country, predictions list, and model name.
    """
    from chronos import ChronosPipeline

    cfg = _load_config("chronos")
    model_id = cfg.get("model_id", "amazon/chronos-t5-tiny")
    num_samples = cfg.get("num_samples", 20)

    # Filter to single country
    country_df = df[df["iso_code"] == country_iso].sort_values("year")
    co2_series = country_df["co2"].dropna().values

    if len(co2_series) < 5:
        raise ValueError(f"Insufficient data for {country_iso}: {len(co2_series)} points")

    # Load Chronos pipeline
    pipeline = ChronosPipeline.from_pretrained(
        model_id,
        device_map="cpu",  # Switch to "cuda" on DataHub
        torch_dtype=torch.float32,
    )

    context = torch.tensor(co2_series, dtype=torch.float32)
    forecast = pipeline.predict(context, prediction_length=horizon, num_samples=num_samples)

    # Median forecast
    predictions = forecast.median(dim=1).squeeze().tolist()
    if isinstance(predictions, float):
        predictions = [predictions]

    return {
        "country": country_iso,
        "predictions": predictions,
        "model": "chronos-2",
        "horizon": horizon,
    }


def run_nhits(
    df: pd.DataFrame,
    country_iso: str,
    horizon: int = 10,
) -> dict:
    """Fine-tuned N-HiTS forecast with covariates.

    # GPU TRAINING — run on DataHub L40

    Args:
        df: ML-ready dataframe.
        country_iso: ISO country code.
        horizon: Forecast horizon in years.

    Returns:
        Dict with country, predictions list, and model name.
    """
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    cfg = _load_config("nhits")
    max_steps = cfg.get("max_steps", 100)
    input_size = cfg.get("input_size", 20)
    learning_rate = cfg.get("learning_rate", 2e-5)
    seed = cfg.get("seed", 42)

    # Prepare NeuralForecast format: unique_id, ds, y + covariates
    country_df = df[df["iso_code"] == country_iso].sort_values("year").copy()
    country_df = country_df.dropna(subset=["co2"])

    nf_df = pd.DataFrame({
        "unique_id": country_iso,
        "ds": pd.to_datetime(country_df["year"], format="%Y"),
        "y": country_df["co2"].values,
    })

    # Add covariates if available
    for cov in ["gdp", "primary_energy_consumption", "population"]:
        if cov in country_df.columns:
            nf_df[cov] = country_df[cov].fillna(method="ffill").fillna(0).values

    # GPU TRAINING — run on DataHub L40
    models = [
        NHITS(
            h=horizon,
            input_size=input_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            random_seed=seed,
            accelerator="cpu",  # Switch to "gpu" on DataHub
        )
    ]

    nf = NeuralForecast(models=models, freq="YS")
    nf.fit(df=nf_df)
    forecast_df = nf.predict()

    predictions = forecast_df["NHITS"].tolist()

    return {
        "country": country_iso,
        "predictions": predictions,
        "model": "nhits",
        "horizon": horizon,
    }


def apply_conformal_intervals(
    y_pred: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 0.1,
) -> dict:
    """Apply MAPIE conformal prediction intervals.

    Args:
        y_pred: Point predictions array.
        y_train: Historical training values (used for residual calibration).
        alpha: Significance level (0.1 = 90% coverage).

    Returns:
        Dict with lower/upper bounds and nominal coverage.
    """
    from mapie.regression import MapieRegressor
    from sklearn.linear_model import LinearRegression

    # Fit a simple model on training indices to calibrate residuals
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    mapie = MapieRegressor(
        estimator=LinearRegression(),
        method="plus",
        cv=5,
    )
    mapie.fit(X_train, y_train)

    X_pred = np.arange(len(y_train), len(y_train) + len(y_pred)).reshape(-1, 1)
    _, intervals = mapie.predict(X_pred, alpha=alpha)

    return {
        "lower": intervals[:, 0, 0].tolist(),
        "upper": intervals[:, 1, 0].tolist(),
        "coverage": 1 - alpha,
    }


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluate forecast quality with MASE and sMAPE.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dict with mase and smape scores.
    """
    from co2_ml.utils.metrics import mase, smape

    return {
        "mase": mase(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
