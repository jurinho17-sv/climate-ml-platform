"""Tests for data preprocessing pipeline."""

import pandas as pd
import pytest
from pathlib import Path

PARQUET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "ml_ready.parquet"


@pytest.fixture
def ml_ready_df() -> pd.DataFrame:
    return pd.read_parquet(PARQUET_PATH)


def test_preprocess_no_nulls_in_key_cols(ml_ready_df: pd.DataFrame) -> None:
    """Key columns (country, iso_code, year) must have zero nulls."""
    for col in ["country", "iso_code", "year"]:
        assert ml_ready_df[col].isna().sum() == 0, f"{col} has nulls"


def test_preprocess_year_range(ml_ready_df: pd.DataFrame) -> None:
    """Year range must be within 1960–2024."""
    assert ml_ready_df["year"].min() >= 1960
    assert ml_ready_df["year"].max() <= 2024


def test_preprocess_country_count(ml_ready_df: pd.DataFrame) -> None:
    """Must have at least 200 unique countries."""
    assert ml_ready_df["iso_code"].nunique() >= 200
