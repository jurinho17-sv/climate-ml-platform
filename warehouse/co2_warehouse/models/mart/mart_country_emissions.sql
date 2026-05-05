SELECT
    year,
    iso_code,
    country_name,
    ROUND(co2, 3)                AS co2,
    ROUND(co2_per_capita, 3)     AS co2_per_capita,
    ROUND(co2_prev_year, 3)      AS co2_prev_year,
    ROUND(co2_yoy_change, 3)     AS co2_yoy_change,
    ROUND(co2_yoy_pct, 3)        AS co2_yoy_pct,
    CASE
        WHEN co2_yoy_pct > 10  THEN 'high_growth'
        WHEN co2_yoy_pct < -10 THEN 'high_decline'
        ELSE 'stable'
    END                          AS emissions_trend_label,
    _loaded_at
FROM {{ ref('int_emissions_by_country') }}
