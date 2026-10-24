# CPCB_AQI_Forecaster

> **Operational 24-to-48-Hour Spatio-Temporal PM2.5 Forecasting & Regulatory Risk System for Delhi-NCR**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Data Source](https://img.shields.io/badge/Data-data.gov.in-orange.svg)](https://data.gov.in/)

## 1. Project Vision
Air pollution in the Indo-Gangetic Plain and Delhi-NCR represents an acute public health crisis. During winter months, shallow planetary boundary layers ($<200\text{ m}$), thermal inversions, and regional emissions drive ambient $PM_{2.5}$ beyond $250\text{--}400\ \mu\text{g/m}^3$. 

While India's **Commission for Air Quality Management (CAQM)** enforces the **Graded Response Action Plan (GRAP)**, regulatory interventions are often reactive—triggered after toxic levels are already reached.

**CPCB_AQI_Forecaster** provides a production-grade machine learning early warning system that forecasts continuous $PM_{2.5}$ concentration trajectories 24 to 48 hours in advance, providing authorities with actionable lead time to execute preventive regulatory interventions.

---

## 2. Modeling Architecture & Roadmap
The project follows a principled multi-tier modeling progression:
* **Tier 1 (Official Baseline):** Regularized Linear Autoregression (Ridge Regression) incorporating lagged telemetry, rolling window statistics, decomposed wind vectors ($U, V$), and cyclical temporal encodings.
* **Tier 2 (Primary Engine):** Multi-Horizon Quantile LightGBM optimizing Pinball Loss to generate calibrated probabilistic percentiles ($P_{10}, P_{50}, P_{90}$).
* **Tier 3 (Spatial Graph Integration):** Dynamic wind-directed spatial lag formulation capturing cross-city plume advection across Delhi's 40-station CAAQMS network.
* **Tier 4 (Policy Layer):** Conformal prediction calibration and cost-sensitive decision threshold optimization for GRAP Stage III/IV triggers.

---

## 3. Project Structure
```text
CPCB_AQI_Forecaster/
├── docs/                     # Research specifications, literature review & benchmarks
├── data/
│   ├── raw/                  # Raw sensor cache from CPCB / data.gov.in API
│   ├── processed/            # Cleaned, time-aligned, feature-engineered matrices
│   └── external/             # Meteorological and spatial coordinate datasets
├── src/
│   ├── config.py             # Global project configuration and constants
│   ├── ingestion/            # API ingestion clients and data acquisition modules
│   ├── data_quality/         # Sensor validation, health checks, and missingness rules
│   ├── features/             # Lags, rolling statistics, and wind vector transformations
│   ├── models/               # Model training (Ridge baseline, Quantile LightGBM)
│   └── evaluation/           # Metrics (MAE, RMSE, Pinball Loss, GRAP CSI score)
├── tests/                    # Unit tests for pipelines and feature transforms
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 4. Documentation
* [Problem Statement & Technical Specification](docs/problem_statement_spec.md)
* [Literature Review, Benchmarks & SOTA Gap Analysis](docs/literature_review_and_benchmarks.md)

---

## 5. Author
* **Lead Engineer:** MikhulJSR ([github.com/MikhulJSR](https://github.com/MikhulJSR))
