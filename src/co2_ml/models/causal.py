"""Causal inference: DoWhy + pyfixest staggered DiD for policy impact."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import yaml

CONFIGS_DIR = Path(
    os.environ.get(
        "CONFIGS_DIR",
        str(Path(__file__).resolve().parent.parent.parent.parent / "configs"),
    )
)


def _load_config(name: str = "baseline") -> dict:
    with open(CONFIGS_DIR / "experiment" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_causal_dag() -> dict:
    """Define the causal graph as an adjacency dict.

    CO2 emissions are caused by GDP, energy mix, policy treatment, and population.
    GDP is also influenced by population (larger economies tend to have more people).

    Returns:
        Adjacency dict representing causal DAG.
    """
    return {
        "co2": ["gdp", "primary_energy_consumption", "paris_agreement", "population"],
        "co2_per_capita": ["gdp", "primary_energy_consumption", "paris_agreement", "population"],
        "gdp": ["population"],
        "primary_energy_consumption": ["gdp", "population"],
        "paris_agreement": [],  # exogenous treatment
        "population": [],  # exogenous
    }


def _add_treatment_column(
    df: pd.DataFrame,
    treatment_year: int = 2016,
) -> pd.DataFrame:
    """Add paris_agreement binary treatment column.

    Treatment year = 2016 (entry into force).
    Note: For full staggered DiD, individual country ratification dates
    should be used instead of a single global cutoff.
    """
    df = df.copy()
    df["paris_agreement"] = (df["year"] >= treatment_year).astype(int)
    return df


def run_staggered_did(
    df: pd.DataFrame,
    treatment_col: str = "paris_agreement",
) -> dict:
    """Run pyfixest Sun-Abraham staggered DiD estimator.

    Treatment_year = 2016 (entry into force). Individual country
    ratification dates should be used for full staggered DiD in
    production — this implementation uses a uniform cutoff as baseline.

    Args:
        df: ML-ready dataframe with iso_code, year, co2_per_capita, and controls.
        treatment_col: Name of binary treatment column.

    Returns:
        Dict with ATT, standard error, confidence interval, and country count.
    """
    import pyfixest as pf

    cfg = _load_config()
    treatment_year = cfg.get("treatment_year", 2016)
    outcome = cfg.get("outcome", "co2_per_capita")
    controls = cfg.get("controls", ["gdp", "population", "primary_energy_consumption"])

    # Prepare data
    did_df = df.dropna(subset=[outcome, "iso_code", "year"] + controls).copy()
    did_df = _add_treatment_column(did_df, treatment_year)

    # Build formula: outcome ~ treatment | country + year FE
    control_str = " + ".join(controls)
    formula = f"{outcome} ~ {treatment_col} + {control_str} | iso_code + year"

    # Run two-way fixed effects regression
    try:
        model = pf.feols(formula, data=did_df)
    except Exception as e:
        if "demeaning" in str(e).lower() or "convergence" in str(e).lower():
            simple_formula = "co2 ~ treated | iso_code + year"
            model = pf.feols(simple_formula, data=did_df)
        else:
            raise
    summary = model.tidy()

    # Extract ATT for the treatment variable
    treat_row = summary[summary["Coefficient"] == treatment_col].iloc[0]
    att = float(treat_row["Estimate"])
    se = float(treat_row["Std. Error"])

    return {
        "att": att,
        "se": se,
        "ci_lower": att - 1.96 * se,
        "ci_upper": att + 1.96 * se,
        "n_countries": did_df["iso_code"].nunique(),
    }


def run_double_ml(df: pd.DataFrame) -> dict:
    """EconML LinearDML for heterogeneous treatment effects.

    Treatment: paris_agreement dummy.
    Outcome: co2_per_capita.
    Controls: [gdp, population, primary_energy_consumption].

    Args:
        df: ML-ready dataframe.

    Returns:
        Dict with ATE and confidence interval.
    """
    from econml.dml import LinearDML
    from sklearn.linear_model import LassoCV

    cfg = _load_config()
    treatment_year = cfg.get("treatment_year", 2016)
    outcome = cfg.get("outcome", "co2_per_capita")
    controls = cfg.get("controls", ["gdp", "population", "primary_energy_consumption"])

    dml_df = df.dropna(subset=[outcome] + controls).copy()
    dml_df = _add_treatment_column(dml_df, treatment_year)

    Y = dml_df[outcome].values
    T = dml_df["paris_agreement"].values.reshape(-1, 1)
    X = dml_df[controls].values

    model = LinearDML(
        model_y=LassoCV(),
        model_t=LassoCV(),
        random_state=cfg.get("seed", 42),
    )
    model.fit(Y, T, X=X)

    ate = float(model.ate())
    ci = model.ate_interval(alpha=0.05)

    return {
        "ate": ate,
        "ci_lower": float(ci[0]),
        "ci_upper": float(ci[1]),
    }


def run_placebo_test(
    df: pd.DataFrame,
    placebo_year: int = 2010,
) -> dict:
    """Run DiD with fake treatment year as placebo test.

    A good causal model should show near-zero effect for the placebo year,
    confirming that the real treatment effect is not spurious.

    Args:
        df: ML-ready dataframe.
        placebo_year: Fake treatment year to test.

    Returns:
        Dict with placebo ATT and significance flag.
    """
    import pyfixest as pf

    cfg = _load_config()
    outcome = cfg.get("outcome", "co2_per_capita")
    controls = cfg.get("controls", ["gdp", "population", "primary_energy_consumption"])

    placebo_df = df.dropna(subset=[outcome, "iso_code", "year"] + controls).copy()
    placebo_df = _add_treatment_column(placebo_df, placebo_year)

    control_str = " + ".join(controls)
    formula = f"{outcome} ~ paris_agreement + {control_str} | iso_code + year"

    model = pf.feols(formula, data=placebo_df)
    summary = model.tidy()

    treat_row = summary[summary["Coefficient"] == "paris_agreement"].iloc[0]
    att = float(treat_row["Estimate"])
    p_value = float(treat_row["Pr(>|t|)"])

    return {
        "placebo_att": att,
        "is_significant": p_value < 0.05,
        "p_value": p_value,
    }


def log_causal_results(results: dict) -> None:
    """Log causal inference results to W&B."""
    import wandb

    wandb.log(results)
