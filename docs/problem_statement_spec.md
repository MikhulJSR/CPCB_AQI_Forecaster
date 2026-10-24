# Formal Problem Statement: Operational 24-to-48-Hour Probabilistic PM2.5 Forecasting & Regulatory Risk System

## 1. Project Title & Metadata
* **Project Name:** VayuDrishti (वायु दृष्टि) - Spatio-Temporal Air Quality Early Warning System
* **Data Source:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest & Climate Change via [data.gov.in](https://data.gov.in/) (Resource ID: `3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69`) and CPCB Central Control Room.
* **Geographical Scope:** Delhi-National Capital Region (NCR) monitoring station network (~40 continuous stations) within the Indo-Gangetic Plain.
* **Primary Modeling Paradigm:** Spatio-Temporal Multi-Horizon Probabilistic Forecasting.

---

## 2. Real-World Societal Problem & Context
The Indo-Gangetic Plain (IGP), particularly Delhi-NCR, experiences hazardous air pollution episodes annually between October and February. 
* Atmospheric thermal inversions, shallow mixing heights ($<200\text{ m}$), and transboundary agricultural burning combine with local vehicular and industrial emissions.
* Particulate matter with diameter $\le 2.5\ \mu\text{m}$ ($PM_{2.5}$) regularly exceeds $250\text{--}400\ \mu\text{g/m}^3$ (50 to 80 times WHO guidelines).
* The human and economic costs are catastrophic: elevated pulmonary and cardiovascular morbidity, emergency hospital surges, and lost productivity.

### The Regulatory Bottleneck
The Commission for Air Quality Management (CAQM) enforces the **Graded Response Action Plan (GRAP)**. 
However, regulatory actions are frequently **reactive**—enacted only *after* monitors cross emergency thresholds. 
* Enacting restrictions retroactively fails to prevent acute exposure spikes.
* Ordering bans with zero lead time causes massive logistical paralysis.
* Over-reacting on false alarms costs hundreds of crores in halted construction and commerce.

---

## 3. Beneficiaries & The Operational Decision Loop
```
┌────────────────────────────────────────────────────────────────────────┐
│                        Operational Decision Loop                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  [CPCB Sensor Stream + Weather Telemetry]                              │
│                           │                                            │
│                           ▼                                            │
│  [24-to-48-Hour Probabilistic Engine (P10, P50, P90)]                  │
│                           │                                            │
│                           ▼                                            │
│  [Regulatory Risk Trigger: P(PM2.5 > 250 μg/m³) > τ]                   │
│                           │                                            │
│              ┌────────────┴────────────┐                               │
│              ▼                         ▼                               │
│     [CAQM & Regulators]       [Hospitals & Public]                     │
│     * Advance GRAP III/IV     * Oxygen & bed scaling                   │
│     * Reroute diesel trucks   * School outdoor activity bans           │
│     * Pause dust emission     * Vulnerable group advisories            │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Precise Analytical Question
> **Given historical pollutant concentrations across the monitoring station network, high-frequency local meteorological parameters (wind vectors, temperature, humidity), diurnal cycles, and spatial upwind lags, what is the calibrated conditional distribution (P10, P50, P90) of $PM_{2.5}$ concentration across lead times $t \in [t+1\text{h}, t+48\text{h}]$, and what is the probability of crossing the regulatory emergency threshold ($>250\ \mu\text{g/m}^3$) with at least 24 hours of operational lead time?**

---

## 5. Machine Learning Formulation

### A. Mathematical Formulation
* Let $S = \{s_1, s_2, \dots, s_M\}$ denote the spatial set of $M$ monitoring stations.
* Let $\mathbf{x}_{s, t} \in \mathbb{R}^D$ be the observed feature vector at station $s$ and time $t$ (pollutants, weather, cyclical encodings, spatial neighbor lags).
* The forecasting horizon is $H \in \{1, 2, \dots, 48\}$ hours.
* The objective is to estimate conditional quantiles $\hat{q}_{\alpha}(s, t+h)$ for quantiles $\alpha \in \{0.10, 0.50, 0.90\}$:
  $$\hat{q}_{\alpha}(s, t+h) = f_{\alpha}\Big(\{\mathbf{x}_{s', \tau} : s' \in S, \tau \le t\}\Big)$$

### B. Loss Function
The models are trained by minimizing the **Pinball / Quantile Loss**:
$$\mathcal{L}_{\alpha}(y, \hat{q}) = \max\Big(\alpha(y - \hat{q}), (1 - \alpha)(\hat{q} - y)\Big)$$

Aggregated across quantiles $\alpha \in \{0.10, 0.50, 0.90\}$ to yield a calibrated prediction interval $[\hat{q}_{0.10}, \hat{q}_{0.90}]$.

### C. Technical Justification: Why ML Over Baselines / SQL / BI?
1. **Atmospheric Non-Linearity:** Dispersion does not scale linearly. Wind stagnation below $1.5\ \text{m/s}$ causes exponential particulate buildup that linear regression cannot capture.
2. **Spatial Contagion:** Plumes transport downwind based on dynamic wind angles and inter-station Euclidean distances. Classical ARIMA/Exponential Smoothing models each station in isolation and cannot ingest multi-sensor spatial vectors.
3. **Heavy Compute Alternatives (WRF-Chem):** Full numerical chemical transport models require supercomputing clusters and detailed emissions inventories. A well-engineered ML model provides low-latency local inference in milliseconds.

---

## 6. Evaluation Framework

### A. Primary Regression & Probabilistic Metrics
1. **Pinball Loss / Quantile Loss:** Evaluates the accuracy and shape of the predicted distribution.
2. **Prediction Interval Coverage Probability (PICP):** The empirical percentage of true observations falling between $P_{10}$ and $P_{90}$ (target: $\approx 80\%$).
3. **Mean Prediction Interval Width (MPIW):** Measures the sharpness/tightness of the forecast bounds (narrower is better, conditioned on valid coverage).
4. **Continuous Ranked Probability Score (CRPS):** The gold standard for probabilistic time series.

### B. Operational / Regulatory Decision Metrics
1. **Critical Success Index (CSI / Threat Score) on GRAP Stage IV Breaches ($>250\ \mu\text{g/m}^3$):**
   $$\text{CSI} = \frac{\text{Hits}}{\text{Hits} + \text{Misses} + \text{False Alarms}}$$
2. **Lead-Time Exceedance Recall:** The proportion of severe pollution events correctly predicted with $\ge 24$ hours of advance notice.
3. **Asymmetric Utility Score:** Penalizing false negatives (missed emergencies leading to public exposure) 3× more severely than false positives (economic cost of premature restrictions).

---

## 7. Data Hygiene, Quality Risks & Causal Boundaries
1. **Missing-Not-At-Random (MNAR):** Sensor outages, power cuts, and filter tape changes cause non-random missing records. Forward-filling over hours is forbidden; strict data contracts will tag sensor dropout regimes.
2. **Sensor Calibration Drift:** Optical monitors drift over time when clogged with high dust. Baseline standardization techniques must be applied.
3. **Causal Boundary:** This system is a **predictive forecasting and early warning model**, not a causal source apportionment tool. It predicts the *occurrence* of a concentration spike based on weather and lags, but does not isolate the exact percentage of pollution attributable to a specific factory or farm without source fingerprint data.
