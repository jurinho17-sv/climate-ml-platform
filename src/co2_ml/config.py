"""Centralized configuration constants for Global CO2 Insight."""

from pathlib import Path

# Project root (co2_ml/ → src/ → project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / "co2_emissions_kt_by_country_2023.csv"

# Analysis defaults
MIN_EMISSION_THRESHOLD: int = 10_000  # kt — minimum emissions for growth analysis significance
DEFAULT_YEAR_WINDOW: int = 10  # years — default range for sidebar slider
TOP_N_OPTIONS: list[int] = [5, 10, 15, 20]
