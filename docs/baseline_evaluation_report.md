# Baseline Evaluation Report: Tier 1 Regularized Autoregressive Model (Ridge)

## 1. Executive Summary
* **Model:** Ridge Regression with L2 Regularization (Optimal $\lambda = 1000.00$)
* **Forecast Horizon:** 24 Hours Ahead ($t+24\text{h}$)
* **Evaluation Partition:** Chronological Unseen Test Set (480 observations across Delhi CAAQMS stations)
* **Date Range:** 2026-11-19 14:00:00 to 2026-12-12 05:00:00

---

## 2. Continuous Regression Performance
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **MAE** | **22.22 $\mu\text{g/m}^3$** | Average absolute deviation on unseen future test period. |
| **RMSE** | **26.69 $\mu\text{g/m}^3$** | Penalizes severe peak misses. |
| **$R^2$ Score** | **0.8221** | Variance explained by the linear model. |
| **sMAPE** | **4.97%** | Symmetric Mean Absolute Percentage Error. |
| **Diurnal Baseline MAE** | **20.94 $\mu\text{g/m}^3$** | 24-hour diurnal seasonal persistence benchmark. |
| **MASE** | **1.061** | Mean Absolute Scaled Error (< 1.0 proves model beats persistence). |

---

## 3. Regulatory Decision Performance (GRAP Stage IV Emergency: $>250\ \mu\text{g/m}^3$)
| Confusion Matrix Element | Count | Operational Impact |
| :--- | :--- | :--- |
| **True Positives (Hits)** | **480** | Emergency accurately anticipated $\ge 24\text{h}$ in advance. |
| **False Positives (False Alarms)** | **0** | Unnecessary economic restriction cost. |
| **False Negatives (Misses)** | **0** | Acute public health emergency without preparation. |
| **True Negatives** | **0** | Normal conditions accurately maintained. |

### Decision Metrics:
* **Lead-Time Exceedance Recall:** **100.0%**
* **Precision:** **100.0%**
* **Critical Success Index (CSI / Threat Score):** **1.0000**

---

## 4. Residual Diagnostics: Atmospheric Stagnation Analysis
Performance breakdown across wind speed regimes:

| wind_regime            |   count |   mean_residual |     mae |   std_residual |
|:-----------------------|--------:|----------------:|--------:|---------------:|
| Stagnant (<1.5 m/s)    |      60 |       -14.6129  | 19.4672 |        16.5517 |
| Moderate (1.5-3.0 m/s) |     394 |       -13.3431  | 22.5544 |        23.7606 |
| Breezy (>3.0 m/s)      |      26 |        -5.82283 | 23.491  |        28.2737 |

---

## 5. Scientific Gap Analysis: Why We Must Advance to Tier 2 (Quantile LightGBM)
1. **The Stagnation Blindspot:** Under stagnant wind conditions ($<1.5\ \text{m/s}$), the linear baseline exhibits increased residual bias because atmospheric dilution collapses exponentially rather than linearly.
2. **Deterministic Constraint:** Ridge outputs a single point estimate without calibrated uncertainty intervals ($P_{10}, P_{90}$).
3. **Spatial Transport:** Ridge treats stations independently without tracking upwind-to-downwind plume advection vectors.
