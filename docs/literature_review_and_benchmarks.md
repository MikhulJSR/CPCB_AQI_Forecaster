# Comprehensive Research Review: Existing Solutions, Benchmarks & SOTA Gap Analysis

## 1. Executive Summary & Landscape Overview
Forecasting ambient particulate matter ($PM_{2.5}$) across Delhi-NCR and the Indo-Gangetic Plain is an active field spanning government operational agencies, atmospheric research institutes, and machine learning researchers. 

Existing solutions divide into three distinct paradigms:
1. **Physical Numerical Chemical Transport Models (CTMs):** The operational gold standard used by government bodies (e.g., WRF-Chem, SILAM).
2. **Classical Statistical Time-Series Models:** Historical baselines (e.g., SARIMAX, Vector Auto-Regression).
3. **Machine Learning & Deep Learning Models:** Modern data-driven architectures (e.g., Gradient Boosted Trees, LSTMs, Graph Neural Networks).

---

## 2. Deep Dive: Existing Operational Systems in India

### System A: IITM Air Quality Early Warning System (AQEWS) & Decision Support System (DSS)
* **Lead Agency:** Indian Institute of Tropical Meteorology (IITM, Pune), Ministry of Earth Sciences (MoES), in partnership with IMD and NCAR.
* **Core Technology:** **WRF-Chem (Weather Research and Forecasting coupled with Chemistry)** operating alongside **SILAM** (System for Integrated Modelling of Atmospheric Composition) and **ENFUSER** (street-level data fusion).
* **Operational Mechanism:**
  * Runs a 3-dimensional Eulerian grid simulating fluid dynamics, Navier-Stokes equations, radiative transfer, and chemical kinetic reaction mechanisms.
  * Ingests real-time satellite fire radiative power (MODIS / VIIRS on Suomi-NPP) to track crop residue burning plumes from Punjab and Haryana.
  * Generates 72-hour hourly forecasts at nested spatial resolutions (down to $400\text{ m} - 3\text{ km}$ over Delhi-NCR).
* **Critical Limitations & Failure Modes:**
  1. **Massive Computational Bottleneck:** Requires supercomputers (e.g., *Pratyush* / *Mihir* HPC clusters). Simulations take 4 to 8 hours to execute, preventing agile hourly re-forecasting.
  2. **Static Emission Inventories:** WRF-Chem relies heavily on base emission inventories (such as the SAFAR 2018 or TERI 2018 inventories). Because these inventories are static, the model cannot capture dynamic real-time traffic anomalies, local construction dust, or spontaneous biomass waste burning.
  3. **Peak Underestimation:** CTMs perform spatial grid-cell averaging (e.g., $3 \times 3\text{ km}$ cells). Consequently, they systematically underestimate hyper-local peak pollution spikes at ground monitoring stations during severe stagnation events.
  4. **Deterministic Output:** Produces a single forecast trajectory without calibrated uncertainty bounds (no probabilistic percentiles for risk management).

---

## 3. Standard Benchmarks & Published Baseline Numbers

To prove that any ML architecture deserves its place in a production portfolio, it must be benchmarked against a rigorous hierarchy of baselines.

### Benchmark Hierarchy & Typical Delhi Winter Performance

| Tier | Model | Input Architecture | Typical 24h Ahead MAE ($\mu\text{g/m}^3$) | Critical Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 0** | **Naive Persistence** | $\hat{y}_{t+24} = y_t$ | $55.0 - 75.0$ | Fails during rapid atmospheric transitions (e.g., sudden wind drops or cold fronts). |
| **Tier 0** | **Diurnal Persistence** | $\hat{y}_{t+24} = y_{t}$ (same hour yesterday) | $45.0 - 60.0$ | Misses cumulative day-to-day background accumulation. |
| **Tier 1** | **SARIMAX** | Station lag autoregression + external weather | $38.0 - 50.0$ | Assumes linear relationships; cannot model non-linear boundary layer collapses. |
| **Tier 2** | **Single-Station LightGBM / XGBoost** | Local station lags, weather, time encodings | $28.0 - 38.0$ | Treats stations in isolation; blind to spatial plume transport from upwind stations. |
| **Tier 3** | **Deep Sequence Models (LSTM / BiLSTM)** | Sliding multi-variate temporal windows | $25.0 - 34.0$ | High compute overhead; prone to overfitting on limited historical seasons; often black-box. |
| **Target SOTA** | **Spatio-Temporal Quantile Engine (Our Target)** | Spatial lags + wind vectors + quantile loss | **Target: $< 24.0$ MAE (P50)** + Calibrated P10–P90 Bounds | Delivers both low error and calibrated operational risk bounds for GRAP triggers. |

---

## 4. Thorough Review & Gap Analysis of Academic ML Literature

Reviewing published machine learning papers on Delhi air quality reveals four systematic flaws that our project will explicitly overcome:

### Flaw 1: Pervasive Data Leakage in Validation
* **The Academic Error:** A vast majority of published papers apply random $K$-fold cross-validation or randomly split tabular rows into 80% train / 20% test.
* **Why This is Fatal in Time Series:** Random splitting allows the model to train on hour $t+1$ and test on hour $t$. Because air quality exhibits massive autocorrelation, random splitting leaks the answer, inflating $R^2$ to $>0.92$ on paper while completely failing when deployed in real-time.
* **Our Rigorous Fix:** We implement **Purged Rolling-Window Time-Series Cross-Validation** with an embargo period between train and test windows to respect temporal causality.

### Flaw 2: The "Missing-Not-At-Random" (MNAR) Blindspot
* **The Academic Error:** Papers routinely state: *"Missing values were handled using forward-fill (`ffill`)"* or *"Rows with missing values were dropped."*
* **The Reality:** CPCB monitoring stations frequently lose power or suffer filter choke during the most severe pollution peaks. If a sensor drops out for 18 hours during a peak episode and you forward-fill the last pre-peak reading, you feed corrupt synthetic stability into the model.
* **Our Rigorous Fix:** Automated Data Quality Contracts (via Great Expectations / Pydantic) that detect sensor failure, track dropout duration, and use spatial neighbor imputation rather than naive forward-filling.

### Flaw 3: Deterministic Point Prediction vs. Real-World Risk
* **The Academic Error:** 95% of models predict a single continuous value $\hat{y}$ optimizing Mean Squared Error (MSE).
* **The Decision Bottleneck:** If the regulatory threshold for GRAP Stage IV (truck entry ban, school closures) is $250\ \mu\text{g/m}^3$, a point forecast of $242\ \mu\text{g/m}^3$ leads regulators to take no action. But if the 90th percentile ($P_{90}$) is $310\ \mu\text{g/m}^3$, authorities should immediately prepare.
* **Our Rigorous Fix:** Multi-horizon **Quantile Loss Optimization ($P_{10}, P_{50}, P_{90}$)** providing calibrated risk distributions.

### Flaw 4: Physics-Blind Feature Representation
* **The Academic Error:** Feeding raw wind direction ($0^\circ - 360^\circ$) into tree-based or neural models as a continuous number.
* **The Mathematical Reality:** $1^\circ$ and $359^\circ$ represent virtually identical meteorological flow (almost due North), but an algorithm sees them as extreme opposite ends of a numeric range.
* **Our Rigorous Fix:** Full trigonometric **wind vector decomposition ($U, V$ components)** combined with dynamic directional spatial lag features that track plume advection across the network.
