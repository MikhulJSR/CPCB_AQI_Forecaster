# Feature Engineering & Coefficient Dictionary

> **Comprehensive Reference for Engineered Features, Prediction Targets, and Learned Baseline Coefficients for CPCB_AQI_Forecaster**

This document details the complete feature store architecture, physical and mathematical formulations, target variable definitions, and the exact weights learned by our **Tier 1 Ridge Baseline Model**.

---

## 1. The Prediction Target Variables

We formulate three distinct target representations for operational forecasting across a 24-hour horizon ($t+24\text{h}$):

| Target Variable | Data Type | Formula / Condition | Operational Purpose |
| :--- | :--- | :--- | :--- |
| **`target_pm25_h24`** | Float ($\mu\text{g/m}^3$) | $y_{s, t+24} = \text{shift}(-24)$ | **Continuous Forecast:** Predicts exact concentration 24 hours into the future for station $s$. |
| **`target_grap_iv_h24`** | Binary ($\{0, 1\}$) | $\mathbb{I}(y_{s, t+24} > 250.0)$ | **Regulatory Alert:** Flags whether the station will breach the **GRAP Stage IV Emergency Limit** ($>250\ \mu\text{g/m}^3$). |
| **`quantiles_h24`** *(Tier 2)* | Tuple ($P_{10}, P_{50}, P_{90}$) | Quantile Loss minimization | **Probabilistic Risk Envelope:** Captures best-case, median, and worst-case tail risk for decision-makers. |

---

## 2. Feature Catalog (24 Predictor Inputs)

All features are engineered strictly using historical observations at or before forecast time $t$ ($\le t$), guaranteeing zero lookahead leakage.

### A. Autoregressive Lags (Station-Partitioned Memory)
These capture the **physical inertia** of aerosol particles suspended over the urban airshed:
* **`pm25_lag_0` ($y_t$):** Instantaneous concentration measured at the current hour.
* **`pm25_lag_1` ($y_{t-1}$):** Reading 1 hour prior. Combined with `pm25_lag_0`, it measures the **immediate rate of change (momentum)**.
* **`pm25_lag_2` ($y_{t-2}$) & `pm25_lag_3` ($y_{t-3}$):** Rate of change over the last 2 to 3 hours.
* **`pm25_lag_6` ($y_{t-6}$):** Reading 6 hours prior. Bridges morning rush-hour peaks with afternoon solar dilution troughs.
* **`pm25_lag_12` ($y_{t-12}$):** Reading 12 hours prior (half-day diurnal balance).
* **`pm25_lag_24` ($y_{t-24}$):** Reading at the exact same hour yesterday. Captures the primary **24-hour diurnal harmonic**.
* **`pm25_lag_48` ($y_{t-48}$):** Reading 2 days ago at the same hour. Establishes multi-day cyclical baseline stability.

### B. Rolling Window Summary Statistics
These quantify the **background regional air mass** over the past 24 hours:
* **`pm25_roll_mean_24`:** $\frac{1}{24}\sum_{k=0}^{23} y_{t-k}$. The 24-hour moving average concentration.
* **`pm25_roll_std_24`:** 24-hour standard deviation. Measures short-term atmospheric volatility and turbulence.
* **`pm25_roll_min_24`:** 24-hour rolling minimum. The cleanest background baseline achievable during daytime convective mixing.
* **`pm25_roll_max_24`:** 24-hour rolling maximum. Captures recent acute emission spikes (e.g. overnight diesel truck entry).

### C. Atmospheric Physics & Meteorology
These model the atmosphere's capacity to disperse or trap ground-level emissions:
* **`wind_u`:** Orthogonal East-West wind vector ($-WS \cdot \sin(\theta)$ in $\text{m/s}$). Positive = Eastward; Negative = Westward.
* **`wind_v`:** Orthogonal North-South wind vector ($-WS \cdot \cos(\theta)$ in $\text{m/s}$). Positive = Northward; Negative = Southward.
* **`temp` ($T$ in $^\circ\text{C}$):** Surface ambient temperature. Low nighttime ground temperatures create **thermal inversions** that trap smoke.
* **`humidity` ($RH$ in $\%$):** Relative humidity. Moisture accelerates hygroscopic particle swelling and dense particulate smog formation.
* **`ventilation_index_proxy`:** $WS \times \max(1.0, T)$. Measures the effective volume of air available to flush pollutants per unit time.

### D. Cyclical Calendar Time Transforms
Trigonometric mappings prevent discontinuous jumps between 23:00 and 00:00:
* **`sin_hour` & `cos_hour`:** $\sin/\cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$. Encodes diurnal human traffic and solar radiation cycles.
* **`sin_day_of_year` & `cos_day_of_year`:** $\sin/\cos\left(\frac{2\pi \cdot \text{day}}{365.25}\right)$. Encodes annual seasonal transitions (monsoon vs. winter smog).

### E. Chemical Co-Pollutant Tracers
* **`no2`:** Nitrogen dioxide from high-temperature vehicular exhaust (real-time urban traffic density tracer).
* **`so2`:** Sulfur dioxide from industrial coal combustion and oil boilers.
* **`co`:** Carbon monoxide from incomplete combustion. Acts as a chemically stable plume tracer.

---

## 3. Computed Baseline Feature Coefficients (Tier 1 Ridge)

The table below lists the **exact learned weights** computed by our Ridge Baseline model ($\lambda = 1000.0$) trained on standardized features ($\mu=0, \sigma=1$):

| Rank | Feature Name | Learned Weight ($w_j$) | Sign | Atmospheric & Data Science Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **`pm25_lag_24`** | **$+13.25$** | Positive | **Dominant driver.** Captures 24-hour diurnal cycle (same hour yesterday). |
| **2** | **`pm25_lag_48`** | **$+12.74$** | Positive | **Multi-day memory.** Captures 48-hour periodic background cycle. |
| **3** | **`pm25_lag_0`** | **$+8.73$** | Positive | Immediate physical inertia of suspended particulates in the air. |
| **4** | **`co`** | **$+7.78$** | Positive | **Combustion tracer.** Unburned carbon monoxide indicates fresh exhaust plumes. |
| **5** | **`pm25_lag_1`** | **$+7.66$** | Positive | Short-term momentum (rate of change over the last 1 hour). |
| **6** | **`cos_day_of_year`** | **$+6.44$** | Positive | **Annual seasonal macro-trend.** Captures transition into winter smog season. |
| **7** | **`pm25_roll_min_24`** | **$+6.25$** | Positive | The background "floor"—the cleanest the air got in the past day. |
| **8** | **`pm25_lag_2`** | **$+6.17$** | Positive | 2-hour momentum memory. |
| **9** | **`pm10`** | **$+5.53$** | Positive | Coarse particulate mass; correlates with total urban aerosol load. |
| **10** | **`pm25_roll_max_24`** | **$+4.95$** | Positive | Recent peak severity threshold. |
| **11** | **`pm25_roll_mean_24`** | **$+4.72$** | Positive | 24-hour moving average representing regional background smog mass. |
| **12** | **`pm25_lag_3`** | **$+4.45$** | Positive | 3-hour lag momentum. |
| **13** | **`sin_hour`** | **$+2.90$** | Positive | Diurnal rush hour timing curve. |
| **14** | **`temp`** | **$-2.47$** | **Negative** | **Thermal Inversion:** Colder temperatures trap pollution; lower $T \implies$ Higher $PM_{2.5}$! |
| **15** | **`no2`** | **$+1.78$** | Positive | High-temperature diesel exhaust tracer. |
| **16** | **`o3`** | **$+1.32$** | Positive | Photochemical oxidant activity. |
| **17** | **`sin_day_of_year`** | **$+1.32$** | Positive | Annual calendar progression. |
| **18** | **`cos_hour`** | **$-0.85$** | Negative | Nighttime vs. daytime boundary layer cooling factor. |
| **19** | **`pm25_lag_6`** | **$+0.67$** | Positive | Intermediate morning/afternoon transition lag. |
| **20** | **`pm25_lag_12`** | **$+0.53$** | Positive | Half-day day/night balance. |
| **21** | **`humidity`** | **$+0.44$** | Positive | **Hygroscopic growth:** High humidity promotes water absorption and aerosol swelling. |
| **22** | **`wind_v`** | **$+0.34$** | Positive | North-South transport vector. |
| **23** | **`so2`** | **$+0.29$** | Positive | Industrial coal/boiler emission tracer. |
| **24** | **`ventilation_index_proxy`** | **$+0.27$** | Positive | Interaction proxy. |
| **25** | **`wind_speed`** | **$-0.06$** | **Negative** | **Dilution Physics:** Higher wind speed carries smoke away; calm winds cause accumulation. |
| **26** | **`pm25_roll_std_24`** | **$+0.05$** | Positive | Short-term atmospheric turbulence. |
| **27** | **`wind_u`** | **$-0.03$** | Negative | East-West transport vector. |
| **28** | **`wind_direction`** | **$+0.02$** | Positive | Compass direction modifier. |

---

## 4. Key Takeaways from the Learned Weights

1. **Physical Laws Verified:**
   * **Temperature (`-2.47`)** and **Wind Speed (`-0.06`)** are both negative. The model correctly learned that colder temperatures (thermal inversion) and calm winds produce severe air pollution.
2. **24h Periodic Memory Dominance:**
   * `pm25_lag_24` ($+13.25$) and `pm25_lag_48` ($+12.74$) are the two most powerful features in the entire system, proving that urban air quality in Delhi is heavily cyclical.
3. **Chemical Tracers Matter:**
   * Carbon monoxide (`co`, $+7.78$) ranks #4, confirming that tracking fresh combustion exhaust is essential for predicting next-day particulate spikes.
