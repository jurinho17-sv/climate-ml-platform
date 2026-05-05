"""Pytest wrappers around the Great Expectations validation suites."""

from tests.data.ge_validation import validate_processed_parquet, validate_raw_owid


def test_raw_owid_passes_ge() -> None:
    assert validate_raw_owid("data/raw/owid-co2-data.csv") is True


def test_processed_parquet_passes_ge() -> None:
    assert validate_processed_parquet("data/processed/ml_ready.parquet") is True
