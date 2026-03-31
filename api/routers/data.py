"""Data-serving endpoints — GET /data/countries, /data/emissions, /data/metadata."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, Query

from api.dependencies import get_dataframe
from api.schemas.data import (
    CountriesResponse,
    CountryRecord,
    EmissionRecord,
    EmissionsResponse,
    MetadataResponse,
)

router = APIRouter()


@router.get("/countries", response_model=CountriesResponse)
def list_countries(df: pd.DataFrame = Depends(get_dataframe)) -> CountriesResponse:
    """Return all unique countries in the dataset."""
    countries_df = df[["country", "iso_code"]].drop_duplicates().sort_values("country")
    records = [CountryRecord(name=row["country"], iso_code=row["iso_code"]) for _, row in countries_df.iterrows()]
    return CountriesResponse(countries=records, total=len(records))


@router.get("/emissions", response_model=EmissionsResponse)
def get_emissions(
    start_year: int = Query(default=1960, ge=1750, le=2100),
    end_year: int = Query(default=2023, ge=1750, le=2100),
    countries: str | None = Query(default=None, description="Comma-separated ISO codes"),
    df: pd.DataFrame = Depends(get_dataframe),
) -> EmissionsResponse:
    """Return CO2 emissions time series, optionally filtered by year range and countries."""
    filtered = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    if countries:
        iso_list = [c.strip().upper() for c in countries.split(",") if c.strip()]
        filtered = filtered[filtered["iso_code"].isin(iso_list)]

    filtered = filtered[["country", "iso_code", "year", "co2"]].sort_values(["country", "year"])

    records = [
        EmissionRecord(
            country=row["country"],
            iso_code=row["iso_code"],
            year=int(row["year"]),
            co2=round(row["co2"], 3) if pd.notna(row["co2"]) else None,
        )
        for _, row in filtered.iterrows()
    ]
    return EmissionsResponse(data=records, total=len(records))


@router.get("/metadata", response_model=MetadataResponse)
def get_metadata(df: pd.DataFrame = Depends(get_dataframe)) -> MetadataResponse:
    """Return dataset metadata: year range and country count."""
    return MetadataResponse(
        min_year=int(df["year"].min()),
        max_year=int(df["year"].max()),
        total_countries=df["iso_code"].nunique(),
    )
