# Data Lineage

End-to-end Bronze → Silver → Gold flow for the climate ML platform. Every box
is a real on-disk artifact; every arrow is a concrete DVC stage in `dvc.yaml`,
a Prefect task in `flows/co2_pipeline.py`, or a dbt model in
`warehouse/co2_warehouse/`.

```mermaid
flowchart LR
    OWID[("OWID API<br/>raw.githubusercontent.com<br/>owid/co2-data")]
    WB[("World Bank WDI API<br/>3 indicators")]
    PARIS[("UNFCCC ratification<br/>register (curated 40 emitters)")]

    OWID -->|ingest_owid<br/>ingest.py| BRONZE_OWID["data/bronze/owid_co2/<br/>ingestion_date=YYYY-MM-DD/<br/>part-*.parquet"]
    WB -->|ingest_worldbank<br/>ingest_worldbank.py| BRONZE_WDI["data/bronze/worldbank_wdi/<br/>ingestion_date=YYYY-MM-DD/<br/>part-*.parquet"]
    PARIS -->|ingest_paris<br/>ingest_paris.py| BRONZE_PARIS["data/bronze/paris_ratifications/<br/>ratification_dates.parquet"]

    BRONZE_OWID -->|silver_clean<br/>silver_clean_pandas.py (local) /<br/>silver_clean_spark.py (DataHub)| SILVER["data/silver/conformed/<br/>country_year_panel.parquet<br/>(25 cols)"]
    BRONZE_WDI --> SILVER
    BRONZE_PARIS --> SILVER

    SILVER -->|dbt_gold<br/>dbt deps + run + test| WAREHOUSE["data/warehouse/co2.duckdb<br/>(mart_ml_features, 17 cols)"]

    WAREHOUSE -->|export_gold_parquet<br/>duckdb -> pandas -> parquet| GOLD["data/gold/ml_features.parquet<br/>(17 cols, system of record for API)"]

    GOLD -->|train_nhits.py| NHITS["models/nhits/"]
    GOLD -->|train_lstm_ae.py| LSTM["models/lstm_ae/"]
    GOLD -->|run_staggered_did| CAUSAL["DoWhy + pyfixest<br/>(in-process)"]

    NHITS --> API["FastAPI<br/>/forecast /anomalies /policy_effect"]
    LSTM --> API
    CAUSAL --> API

    API --> FRONTEND["Streamlit Frontend<br/>HuggingFace Spaces"]
```

## DVC pipeline stages (six total)

| # | Stage | Inputs | Output |
|---|---|---|---|
| 1 | `ingest_owid` | OWID API | `data/bronze/owid_co2/ingestion_date=YYYY-MM-DD/part-0.parquet` |
| 2 | `ingest_worldbank` | World Bank WDI API | `data/bronze/worldbank_wdi/ingestion_date=YYYY-MM-DD/part-0.parquet` |
| 3 | `ingest_paris` | curated UNFCCC table | `data/bronze/paris_ratifications/ratification_dates.parquet` |
| 4 | `silver_clean` | all 3 Bronze parquets | `data/silver/conformed/country_year_panel.parquet` (25 cols) |
| 5 | `dbt_gold` | Silver conformed + dbt models | `data/warehouse/co2.duckdb` (`mart_ml_features` table, 17 cols) |
| 6 | `export_gold_parquet` | `co2.duckdb` | `data/gold/ml_features.parquet` (17 cols) |

The `dbt_gold` cmd is `dbt deps && dbt run && dbt test`. `--profiles-dir .`
points at the repo-root `profiles.yml`. The dbt model `mart_ml_features` is the
system of record; the flat parquet at `data/gold/` is materialized from it for
runtime use by FastAPI.

## Layer responsibilities

| Layer | Path | Producer | Consumer | Cadence |
|---|---|---|---|---|
| Bronze | `data/bronze/<source>/ingestion_date=YYYY-MM-DD/` | `ingest*.py` (one per source) | Silver job (pandas or Spark) | append-only on each pipeline run |
| Silver (cleansed) | `data/silver/cleansed/owid_co2.parquet` | `preprocess.py` (legacy local-only OWID cleansing) | not on the critical DVC path | overwrite |
| Silver (conformed) | `data/silver/conformed/country_year_panel.parquet` | `silver_clean_pandas.py` (local) or `silver_clean_spark.py` (DataHub) | dbt source `silver_conformed` | overwrite |
| Gold (DuckDB) | `data/warehouse/co2.duckdb` table `mart_ml_features` | dbt model `mart_ml_features.sql` | analytics, BI, validation tests | rebuild on each pipeline run |
| Gold (flat parquet) | `data/gold/ml_features.parquet` | `export_gold_parquet` (DuckDB → pandas → parquet) | `api/main.py` lifespan, `train_*.py`, causal models | rebuild on each pipeline run |

Both Gold representations share an identical 17-column schema (see
`schemas/gold_ml_features.yaml`).

## Provenance columns

Every Bronze parquet carries:
- `_ingested_at` (ISO-8601 UTC timestamp)
- `_source_url` (origin URL)

Silver carries the OWID `_ingested_at` and `_source_url` forward through the
join. Gold's dbt mart adds `_built_at` (the dbt run wall-clock).

## GE gates (Prefect orchestration)

When the pipeline runs through Prefect (`flows/co2_pipeline.py`), GE suites
gate the layer transitions:

- `validate_bronze` runs `raw_owid_suite` against the latest Bronze OWID
  partition (via `validate_bronze_owid_parquet`) — raises on failure.
- `validate_silver` runs `silver_conformed_suite` (25 columns required, joined
  fields existence-checked, year between 1960-2024) — raises on failure.
- The dbt build/test phase enforces uniqueness on `(iso_code, year)`,
  `not_null` on `paris_treated` and `years_since_ratification`, and a custom
  `assert_paris_join_coverage` test (≥1% treated rows; severity=error).

## dbt model graph (Gold layer)

```
silver_conformed.country_year_panel  (source — env var SILVER_CONFORMED_PATH)
            |
            v
        stg_co2  (view, type-cast and filtered)
            |
            +-> int_emissions_by_country  (view, YoY metrics)
            |              |
            |              v
            |      mart_country_emissions  (table — for analytics)
            |
            +-> mart_ml_features  (table — system of record for ML, 17 cols)
```

## DataHub PySpark execution

The Silver `silver_clean` step has two interchangeable runners that produce
**schema-identical** output:

- Locally: `python -m co2_ml.pipelines.silver_clean_pandas` (default in
  Prefect, CD pipeline, and DVC).
- On DataHub: `spark-submit --master local[*] src/co2_ml/pipelines/silver_clean_spark.py`
  (uses NVIDIA L40 48GB for scale).

To run the production Spark variant:

```bash
ssh berkeley-datahub
cd ~/climate-ml-platform
git pull origin main
pip install -e . pyspark==3.5.* --break-system-packages
spark-submit --master local[*] src/co2_ml/pipelines/silver_clean_spark.py
```
