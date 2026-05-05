"""Prefect 3 orchestration layer for the OWID CO2 data pipeline."""

from __future__ import annotations

import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from co2_ml.pipelines.ingest import download_owid_data
from co2_ml.pipelines.preprocess import preprocess


@task(retries=3, retry_delay_seconds=60, log_prints=True)
def ingest_task(url: str, output_path: str) -> int:
    download_owid_data(url, output_path)
    row_count = int(pd.read_csv(output_path, usecols=["year"]).shape[0])
    print(f"Ingested {row_count:,} rows into {output_path}")

    summary = (
        f"# Ingest summary\n\n"
        f"- **Source URL:** {url}\n"
        f"- **Output path:** `{output_path}`\n"
        f"- **Rows downloaded:** {row_count:,}\n"
    )
    create_markdown_artifact(key="ingest-summary", markdown=summary)
    return row_count


@task(retries=2, log_prints=True)
def preprocess_task(input_path: str, output_path: str) -> int:
    df = preprocess(input_path, output_path)

    row_count = len(df)
    countries = int(df["iso_code"].nunique())
    year_min = int(df["year"].min())
    year_max = int(df["year"].max())
    columns = list(df.columns)

    print(
        f"Preprocessed: {row_count:,} rows, {countries} countries, "
        f"years {year_min}-{year_max}, {len(columns)} columns"
    )

    columns_md = "\n".join(f"  - `{c}`" for c in columns)
    summary = (
        f"# Preprocess summary\n\n"
        f"- **Input:** `{input_path}`\n"
        f"- **Output:** `{output_path}`\n"
        f"- **Rows:** {row_count:,}\n"
        f"- **Countries retained:** {countries}\n"
        f"- **Year range:** {year_min}-{year_max}\n"
        f"- **Columns selected ({len(columns)}):**\n{columns_md}\n"
    )
    create_markdown_artifact(key="preprocess-summary", markdown=summary)
    return row_count


@flow(name="co2-data-pipeline", log_prints=True)
def co2_pipeline(
    data_url: str = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv",
    raw_output: str = "data/raw/owid-co2-data.csv",
    processed_output: str = "data/processed/ml_ready.parquet",
) -> dict:
    raw_rows = ingest_task(data_url, raw_output)
    processed_rows = preprocess_task(raw_output, processed_output)
    return {"raw_rows": raw_rows, "processed_rows": processed_rows, "status": "success"}


if __name__ == "__main__":
    co2_pipeline.serve(
        name="co2-pipeline-dev",
        cron="0 6 * * *",
        parameters={"data_url": "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"},
    )
