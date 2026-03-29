"""Anomaly detection: LSTM Autoencoder + Isolation Forest hybrid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs" / "model"


def _load_config() -> dict:
    with open(CONFIGS_DIR / "lstm_ae.yaml") as f:
        return yaml.safe_load(f)


class LSTMAutoencoder(nn.Module):
    """LSTM Autoencoder for temporal anomaly detection.

    Encoder compresses the input sequence into a latent representation,
    decoder reconstructs it. High reconstruction error = anomaly.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # Decode — repeat latent vector for each timestep
        seq_len = x.size(1)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        decoder_out, _ = self.decoder(decoder_input, (hidden, cell))
        reconstruction = self.output_layer(decoder_out)
        return reconstruction


def build_lstm_ae(
    input_dim: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
) -> nn.Module:
    """Build LSTM Autoencoder architecture.

    # GPU TRAINING — run on DataHub L40

    Args:
        input_dim: Number of input features per timestep.
        hidden_dim: LSTM hidden state dimension.
        num_layers: Number of stacked LSTM layers.

    Returns:
        LSTMAutoencoder model (untrained).
    """
    cfg = _load_config()
    hidden_dim = cfg.get("hidden_dim", hidden_dim)
    num_layers = cfg.get("num_layers", num_layers)

    return LSTMAutoencoder(input_dim, hidden_dim, num_layers)


def compute_reconstruction_errors(
    model: nn.Module,
    X: np.ndarray,
) -> np.ndarray:
    """Compute per-sample MSE reconstruction error.

    Args:
        model: Trained LSTMAutoencoder.
        X: Input array of shape (n_samples, seq_len, n_features).

    Returns:
        Array of reconstruction errors, shape (n_samples,).
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        X_recon = model(X_tensor)
        errors = ((X_tensor - X_recon) ** 2).mean(dim=(1, 2)).numpy()
    return errors


def run_isolation_forest(
    errors: np.ndarray,
    contamination: float = 0.05,
) -> np.ndarray:
    """Fit Isolation Forest on reconstruction errors.

    Args:
        errors: Reconstruction error array from LSTM-AE.
        contamination: Expected fraction of anomalies.

    Returns:
        Labels array: -1 = anomaly, 1 = normal.
    """
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(contamination=contamination, random_state=42)
    labels = iso.fit_predict(errors.reshape(-1, 1))
    return labels


def explain_anomalies_shap(
    iso_forest,
    X: np.ndarray,
) -> np.ndarray:
    """Explain anomaly decisions using SHAP TreeExplainer.

    Args:
        iso_forest: Fitted IsolationForest model.
        X: Feature matrix used for anomaly detection.

    Returns:
        SHAP values array, same shape as X.
    """
    import shap

    explainer = shap.TreeExplainer(iso_forest)
    shap_values = explainer.shap_values(X)
    return shap_values


# Known historical emission shock events
KNOWN_EVENTS: dict[int, str] = {
    1990: "Gulf War oil fires",
    1997: "Asian Financial Crisis",
    2008: "Global Financial Crisis",
    2020: "COVID-19 pandemic",
    2022: "Ukraine war energy shock",
}


def label_known_events(anomaly_df: pd.DataFrame) -> pd.DataFrame:
    """Add event_label column for known historical emission shocks.

    Args:
        anomaly_df: DataFrame with 'year' column and anomaly flags.

    Returns:
        DataFrame with added 'event_label' column.
    """
    df = anomaly_df.copy()
    df["event_label"] = df["year"].map(KNOWN_EVENTS).fillna("")
    return df
