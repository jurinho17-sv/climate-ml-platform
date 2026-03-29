"""Data ingestion pipeline — downloads OWID CO2 dataset."""

from pathlib import Path

import pandas as pd
import yaml


def download_owid_data(url: str, output_path: str) -> None:
    """Download OWID CO2 CSV from URL and save locally."""
    print(f"Downloading OWID CO2 data from: {url}")
    df = pd.read_csv(url)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Year range: {year_min}–{year_max}")
    print(f"  Saved to: {output_path}")


def main() -> None:
    params_path = Path(__file__).resolve().parent.parent.parent.parent / "params.yaml"
    with open(params_path) as f:
        params = yaml.safe_load(f)

    project_root = params_path.parent
    url = params["data"]["owid_url"]
    output_path = str(project_root / "data" / "raw" / "owid-co2-data.csv")

    download_owid_data(url, output_path)


if __name__ == "__main__":
    main()
