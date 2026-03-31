"""Global CO2 Insight — FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import pandas as pd
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import anomalies, data, forecast, health, policy
from co2_ml.config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load data and models at startup."""
    root = settings.PROJECT_ROOT

    # 1. Load processed dataset
    parquet_path = root / settings.PROCESSED_PATH
    try:
        app.state.df = pd.read_parquet(parquet_path)
        logger.info("Loaded %d rows from %s", len(app.state.df), parquet_path)
    except Exception as exc:
        logger.warning("Dataset not available: %s", exc)
        app.state.df = None

    # 2. Load NeuralForecast model
    forecast_path = root / "models" / "nhits"
    try:
        from neuralforecast import NeuralForecast

        app.state.nf_model = NeuralForecast.load(path=str(forecast_path))
        logger.info("NeuralForecast model loaded from %s", forecast_path)
    except Exception as exc:
        logger.warning("Forecast model not available: %s", exc)
        app.state.nf_model = None

    # 3. Load LSTM Autoencoder
    ae_path = root / "models" / "lstm_ae" / "lstm_ae.pt"
    try:
        from co2_ml.models.anomaly import LSTMAutoencoder

        checkpoint = torch.load(ae_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["config"]
        model = LSTMAutoencoder(
            input_dim=len(checkpoint["features"]),
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        app.state.lstm_ae = model
        app.state.norm_stats = {
            "mean": checkpoint["means"],
            "std": checkpoint["stds"],
        }
        logger.info("LSTM-AE model loaded from %s", ae_path)
    except Exception as exc:
        logger.warning("Anomaly model not available: %s", exc)
        app.state.lstm_ae = None
        app.state.norm_stats = None

    yield

    logger.info("Shutting down API")


app = FastAPI(
    title="Global CO2 Insight API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "https://*.hf.space",
    ],
    allow_origin_regex=r"https://.*\.hf\.space",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecasting"])
app.include_router(anomalies.router, prefix="/anomalies", tags=["Anomaly Detection"])
app.include_router(data.router, prefix="/data", tags=["Data"])
app.include_router(policy.router, prefix="/policy_effect", tags=["Causal Inference"])


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError) -> JSONResponse:  # noqa: ARG001
    return JSONResponse(status_code=422, content={"detail": str(exc)})
