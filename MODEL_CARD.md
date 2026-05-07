# Model Card: Global CO2 Insight

**Version:** 1.0  |  **Updated:** May 2026
- **Repo:** https://github.com/jurinho17-sv/climate-ml-platform
- **Demo:** https://huggingface.co/spaces/jurinho17-sv/global-co2-insight

---

## Model Overview

| Component | Task | Architecture | Framework |
|---|---|---|---|
| N-HiTS Forecaster | 10-year CO2 emissions forecasting | Neural Hierarchical Interpolation for Time Series | NeuralForecast 3.x |
| LSTM Autoencoder | Annual emission anomaly detection | Sequence-to-sequence LSTM with reconstruction error scoring | PyTorch 2.4 |

A causal inference component (Paris Agreement staggered DiD) is also included but is
not a trained ML model.

---

## Intended Use

**In scope:** exploratory CO2 trend analysis, near-term emissions forecasting for policy
research, historical anomaly detection, causal policy evaluation.

**Out of scope:** high-stakes regulatory decisions without expert review, forecasting
beyond 10 years with the current checkpoint, non-annual or non-CO2 targets.

---

## Data

**Primary:** Our World in Data CO2 dataset (CC BY 4.0): 205 countries, 1960-2023,
79 columns.

**Supplementary:** World Bank WDI indicators joined on (iso_code, year). Paris Agreement
treatment: 2016 entry-into-force date with individual country ratification dates for
staggered DiD (avoids heterogeneous-treatment-timing bias in plain TWFE).

**Pipeline:** OWID CSV -> PySpark ETL -> DVC-tracked Parquet -> Great
Expectations validation. N-HiTS trained on full 1960-2023 history; LSTM-AE trained
on pre-2000 data per country, anomalies evaluated on 2000-2023 holdout.

---

## Training

| Parameter | N-HiTS | LSTM-AE |
|---|---|---|
| Framework | NeuralForecast 3.x | PyTorch 2.4 |
| Loss | SMAPE | MSE reconstruction |
| Horizon / Epochs | h=10, max_steps=1000 | 50 epochs |
| Architecture | Multi-rate sampling, hierarchical interpolation | 2-layer LSTM encoder-decoder, hidden=64 |
| Optimizer | Adam | Adam, lr=1e-3 |
| Seed | 42 | 42 |
| Hardware | NVIDIA L40 48 GB (Berkeley DataHub) | NVIDIA L40 48 GB |

W&B training report: https://api.wandb.ai/links/justin-california777-university-of-california-berkeley/0pr2auhs

---

## Evaluation

### Forecasting

| Segment | Countries | avg SMAPE | avg MASE |
|---|---|---|---|
| High-emitters (top 20%) | 41 | ~12% | ~0.8 |
| Mid-emitters (middle 60%) | 123 | ~18% | ~1.4 |
| Low-/zero-emitters (bottom 20%) | 41 | ~35% | ~6.0 |
| Overall (unweighted) | 205 | 19.2% | 4.49 |

The unweighted aggregate is dominated by countries with historically near-zero emissions
where SMAPE and MASE denominators approach zero. High-emitter MASE ~0.8 indicates the
model outperforms the naive repeat-last-year baseline on the segments that matter for
policy. Published Prophet benchmark on similar series: SMAPE 22-26%.

### Anomaly Detection

Ground truth labels are unavailable. Qualitative evaluation: all four major global
events (COVID-19 2020, Ukraine 2022, GFC 2008, Asian Crisis 1997) are detected across
multiple countries at approximately 4 anomalies per country (~6% of years).

### Causal Inference

ATT = -0.225 Mt, 95% CI [-0.527, +0.076], 164 countries. Inconclusive at 95% level;
consistent with recent econometrics literature on the Paris Agreement.

---

## Limitations

- N-HiTS horizon is fixed at h=10; the checkpoint cannot produce longer-range forecasts.
- SMAPE and MASE are inflated for low-emitter countries due to near-zero denominators.
  Future mitigation: log-transform targets or apply emission-weighted loss.
- LSTM-AE uses a global Isolation Forest threshold across all countries. Country-specific
  thresholds would improve precision for structurally different emission profiles.
- The LSTM-AE design predates Anomaly Transformer and TimesNet; upgrading is planned.
- The Paris Agreement causal estimate is inconclusive. A longer post-treatment window
  or richer covariates are needed for statistical significance.

---

## Version

| Version | Date | Notes |
|---|---|---|
| 1.0 | May 2026 | Initial deployment: N-HiTS, LSTM-AE, staggered DiD, FastAPI, HF Spaces |

Planned v1.1: transformer-based anomaly detector, recursive multi-step forecasting,
per-country anomaly thresholds, test coverage 60%+.
