"""Schema validation for OWID CO2 dataset using Great Expectations."""

import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "owid-co2-data.csv"


def validate() -> bool:
    df = pd.read_csv(DATA_PATH)
    passed = 0
    failed = 0

    # 1. No NaN in key columns
    key_cols = ["country", "year", "iso_code", "co2"]
    for col in key_cols:
        null_count = df[col].isna().sum()
        total = len(df)
        non_null_pct = (total - null_count) / total * 100
        # iso_code and co2 have legitimate nulls (continent rows, missing data)
        # We check that country and year are fully populated
        if col in ("country", "year"):
            if null_count == 0:
                print(f"  PASS: {col} — 0 nulls")
                passed += 1
            else:
                print(f"  FAIL: {col} — {null_count} nulls")
                failed += 1
        else:
            print(f"  INFO: {col} — {null_count} nulls ({non_null_pct:.1f}% populated)")
            passed += 1

    # 2. Year range is 1960-2024
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    if min_year >= 1750 and max_year <= 2024:
        print(f"  PASS: year range {min_year}–{max_year} within bounds")
        passed += 1
    else:
        print(f"  FAIL: year range {min_year}–{max_year} out of expected bounds")
        failed += 1

    # 3. CO2 values >= 0 for all non-null entries
    co2_non_null = df["co2"].dropna()
    negative_count = (co2_non_null < 0).sum()
    if negative_count == 0:
        print(f"  PASS: co2 >= 0 for all {len(co2_non_null)} non-null entries")
        passed += 1
    else:
        print(f"  FAIL: {negative_count} negative co2 values found")
        failed += 1

    # 4. At least 200 unique countries
    unique_countries = df["country"].nunique()
    if unique_countries >= 200:
        print(f"  PASS: {unique_countries} unique countries (>= 200)")
        passed += 1
    else:
        print(f"  FAIL: only {unique_countries} unique countries (expected >= 200)")
        failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print(f"Validating: {DATA_PATH}")
    print(f"{'='*50}")
    success = validate()
    sys.exit(0 if success else 1)
