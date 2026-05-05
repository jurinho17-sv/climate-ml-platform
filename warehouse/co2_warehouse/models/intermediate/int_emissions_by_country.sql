SELECT
    *,
    LAG(co2) OVER (PARTITION BY iso_code ORDER BY year) AS co2_prev_year,
    co2 - LAG(co2) OVER (PARTITION BY iso_code ORDER BY year) AS co2_yoy_change,
    (co2 - LAG(co2) OVER (PARTITION BY iso_code ORDER BY year))
        / NULLIF(LAG(co2) OVER (PARTITION BY iso_code ORDER BY year), 0) * 100 AS co2_yoy_pct
FROM {{ ref('stg_co2') }}
