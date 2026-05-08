SELECT
    CAST(year AS INTEGER)              AS year,
    iso_code,
    country                            AS country_name,
    CAST(co2 AS DOUBLE)                AS co2,
    CAST(co2_per_capita AS DOUBLE)     AS co2_per_capita,
    co2_per_gdp,
    coal_co2,
    oil_co2,
    gas_co2,
    cement_co2,
    primary_energy_consumption,
    energy_per_capita,
    gdp,
    population,
    share_global_co2,
    cumulative_co2,
    methane,
    nitrous_oxide,
    total_ghg,
    CURRENT_TIMESTAMP                  AS _loaded_at
FROM {{ source('silver_conformed', 'country_year_panel') }}
WHERE year >= 1960
  AND iso_code IS NOT NULL
  AND iso_code != ''
