"""Data preprocessing pipeline — cleans and prepares ML-ready features."""

from pathlib import Path

import pandas as pd
import yaml

CORE_COLUMNS = [
    "country",
    "iso_code",
    "year",
    "co2",
    "co2_per_capita",
    "co2_per_gdp",
    "coal_co2",
    "oil_co2",
    "gas_co2",
    "cement_co2",
    "primary_energy_consumption",
    "energy_per_capita",
    "gdp",
    "population",
    "share_global_co2",
    "cumulative_co2",
    "methane",
    "nitrous_oxide",
    "total_ghg",
]


def preprocess(input_path: str, output_path: str, min_year: int = 1960) -> pd.DataFrame:
    """Load OWID CSV, filter, select core features, save as parquet."""
    df = pd.read_csv(input_path)
    print(f"Raw data: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Drop continent aggregates (no iso_code)
    df = df.dropna(subset=["iso_code"])
    print(f"After dropping null iso_code: {len(df):,} rows")

    # Drop years before min_year
    df = df[df["year"] >= min_year]
    print(f"After filtering year >= {min_year}: {len(df):,} rows")

    # Select core columns (only those present in dataset)
    available = [c for c in CORE_COLUMNS if c in df.columns]
    missing = [c for c in CORE_COLUMNS if c not in df.columns]
    if missing:
        print(f"Warning: columns not found in dataset: {missing}")
    df = df[available].copy()

    # Save as parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    # Print stats
    print(f"\nML-ready dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Year range: {int(df['year'].min())}–{int(df['year'].max())}")
    print(f"Countries: {df['iso_code'].nunique()}")
    print(f"\nNull counts per column:")
    nulls = df.isnull().sum()
    for col in available:
        n = nulls[col]
        if n > 0:
            print(f"  {col}: {n:,} ({n/len(df)*100:.1f}%)")
        else:
            print(f"  {col}: 0")
    print(f"\nSaved to: {output_path}")
    return df


def main() -> None:
    params_path = Path(__file__).resolve().parent.parent.parent.parent / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    project_root = params_path.parent
    input_path = str(project_root / "data" / "raw" / "owid-co2-data.csv")
    output_path = str(project_root / "data" / "processed" / "ml_ready.parquet")
    min_year = params["data"]["min_year"]

    preprocess(input_path, output_path, min_year)


if __name__ == "__main__":
    main()
