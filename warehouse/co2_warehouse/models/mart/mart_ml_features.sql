{{ config(materialized='table') }}

WITH base AS (
    SELECT *
    FROM {{ source('silver_conformed', 'country_year_panel') }}
)

SELECT
    iso_code,
    CAST(year AS INTEGER)                                              AS year,
    CAST(co2 AS DOUBLE)                                                AS co2,
    CAST(co2_per_capita AS DOUBLE)                                     AS co2_per_capita,
    CAST(gdp AS DOUBLE)                                                AS gdp,
    CAST(population AS BIGINT)                                         AS population,
    CAST(primary_energy_consumption AS DOUBLE)                         AS primary_energy_consumption,
    CAST(gdp_growth_pct AS DOUBLE)                                     AS gdp_growth_rate,
    -- Energy-mix shares as fractions of total CO2
    CASE WHEN co2 > 0 THEN coal_co2   / co2 END                        AS coal_pct,
    CASE WHEN co2 > 0 THEN oil_co2    / co2 END                        AS oil_pct,
    CASE WHEN co2 > 0 THEN gas_co2    / co2 END                        AS gas_pct,
    CASE WHEN co2 > 0 THEN cement_co2 / co2 END                        AS cement_pct,
    -- Paris Agreement treatment (NULL/0/false until DataHub Spark fills in real values)
    CAST(ratification_year AS INTEGER)                                 AS ratification_year,
    CAST(COALESCE(paris_treated, FALSE) AS BOOLEAN)                    AS paris_treated,
    CAST(COALESCE(years_since_ratification, 0) AS INTEGER)             AS years_since_ratification,
    CURRENT_TIMESTAMP                                                  AS _built_at
FROM base
WHERE iso_code IS NOT NULL
  AND year IS NOT NULL
