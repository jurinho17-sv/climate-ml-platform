# Model Card: Climate ML Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C.svg)](https://pytorch.org/)
[![NeuralForecast](https://img.shields.io/badge/NeuralForecast-3.x-024DFD.svg)](https://github.com/Nixtla/neuralforecast)
[![HF Spaces](https://img.shields.io/badge/HF-Spaces-FFD21E.svg)](https://huggingface.co/spaces/jurinho17-sv/global-co2-insight)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

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
| Hardware | NVIDIA L40 48GB | NVIDIA L40 48GB |

W&B training report: https://api.wandb.ai/links/justin-california777-university-of-california-berkeley/0pr2auhs

*Time-based split: 1960-2018 train, 2019-2021 validation, 2022-2023 test. All training runs are logged to W&B; exact hyperparameters, seed (42), and data version (DVC-tracked) are captured in the W&B report linked above.*

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

- **Fixed forecast horizon (h=10)**: The checkpoint cannot produce forecasts beyond 10 years. *Mitigation*: Train a recursive multi-step variant (planned v1.1).
- **Metric inflation on low-emitter countries**: SMAPE and MASE denominators approach zero for ~41 historically near-zero emitters, inflating the unweighted aggregate. *Mitigation*: Log-transform targets or apply emission-weighted loss; report segmented metrics alongside aggregates (already done in the Evaluation section).
- **Global anomaly threshold**: LSTM-AE uses one Isolation Forest decision boundary across all countries. *Mitigation*: Fit per-country thresholds for structurally different emission profiles (planned v1.1).
- **Pre-transformer anomaly architecture**: LSTM-AE design predates Anomaly Transformer and TimesNet. *Mitigation*: Upgrade to a transformer-based anomaly detector (planned v1.1).
- **Inconclusive causal estimate**: The Paris Agreement ATT 95% CI crosses zero, so an effect cannot be confirmed at conventional significance. *Mitigation*: Wait for a longer post-treatment window, add richer covariates, or apply alternative identification strategies such as synthetic controls.

---

## Ethical Considerations

- **Intended use**: Exploratory policy research, academic analysis, and educational demonstrations of ML pipeline design. Not intended for high-stakes regulatory or financial decisions without expert review.
- **Performance disparity**: The model significantly underperforms on low-emitter and rapidly industrializing nations. Users should apply segmented evaluation (see Results) rather than aggregate metrics.
- **Causal claims**: The Paris Agreement ATT estimate is inconclusive (95% CI crosses zero). Results should not be cited as evidence of policy success or failure without additional analysis.
- **Data provenance**: CO2 data sourced from Our World in Data (CC BY 4.0). World Bank WDI data is subject to their open data terms.

---

## Version

| Version | Date | Notes |
|---|---|---|
| 1.0 | May 2026 | Initial deployment: N-HiTS, LSTM-AE, staggered DiD, FastAPI, HF Spaces |

Planned v1.1: transformer-based anomaly detector, recursive multi-step forecasting,
per-country anomaly thresholds, test coverage 60%+.

---

## Citation

If you use this work, please cite:

```bibtex
@software{kim2026climateMlPlatform,
  author    = {Kim, Ju Ho},
  title     = {Climate ML Platform: End-to-End Climate Data Platform
               with Medallion Lakehouse Pipeline},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/jurinho17-sv/climate-ml-platform}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
