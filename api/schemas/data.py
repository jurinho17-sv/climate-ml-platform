"""Data-serving endpoint schemas."""

from __future__ import annotations

from pydantic import BaseModel


class CountryRecord(BaseModel):
    name: str
    iso_code: str


class CountriesResponse(BaseModel):
    countries: list[CountryRecord]
    total: int


class EmissionRecord(BaseModel):
    country: str
    iso_code: str
    year: int
    co2: float | None


class EmissionsResponse(BaseModel):
    data: list[EmissionRecord]
    total: int


class MetadataResponse(BaseModel):
    min_year: int
    max_year: int
    total_countries: int
