# The Simple English Guide: Understanding Tier 2 (Quantile LightGBM & Spatial Graphs)

> **A clear, beginner-friendly, and complete walkthrough of everything built in Tier 2: why it was needed, how each component works, and how it crushed the Tier 1 baseline.**

---

## 1. The Big Picture: Why Did We Need Tier 2?

In **Tier 1**, we built a solid baseline using **Ridge Regression** (a regularized straight-line model). 
While it gave us a clean starting point, it suffered from **three critical real-world flaws**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE 3 FATAL FLAWS OF TIER 1                               │
├─────────────────────────┬──────────────────────────────────────────────────────────────┤
│ 1. The Straight-Line Trap│ Assumed air behaves like a straight line ($y = mx + b$).     │
│                         │ When wind drops to zero, smog pools and explodes 5x faster.   │
│                         │ A straight line cannot "bend" to capture this sudden spike!  │
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 2. Isolated Island Silos│ Treated every monitoring station as if it lived on an island.│
│                         │ If a huge smoke cloud is coming from Bawana towards Anand     │
│                         │ Vihar, Anand Vihar had NO IDEA until the smoke hit its sensor.│
├─────────────────────────┼──────────────────────────────────────────────────────────────┤
│ 3. Point Estimate Only  │ Output only a single number (e.g., "265 μg/m³").             │
│                         │ Is that 265 ± 5 or 265 ± 100? Municipal ministers cannot ban │
│                         │ diesel trucks without knowing the worst-case risk range!     │
└─────────────────────────┴──────────────────────────────────────────────────────────────┘
```

**Tier 2 solves all three flaws completely** with three engineering pillars:
1. **Atmospheric Spatial Graph:** Connects the stations through wind-directed geometry.
2. **LightGBM Decision Trees:** Bends and splits feature space non-linearly to catch abrupt atmospheric stagnation.
3. **Multi-Horizon Quantile Regression:** Delivers an uncertainty risk envelope ($P_{10}, P_{50}, P_{90}$) for both 24-hour and 48-hour planning.

---

## 2. Pillar 1: The Wind-Directed Spatial Plume Graph

Delhi's air is not static. Pollution travels with the wind like an invisible river of smoke across the city.

### Delhi's Actual Geography:
```
                      DELHI'S SPATIAL POLLUTION CORRIDOR
                                
                      [Bawana (NW)]
                          \
                           \   Prevailing Winter Wind (North-West to South-East)
                            ▼
                  [Punjabi Bagh (W)]
                           │
                           ▼
                      [ITO (Center)] ──────► [Anand Vihar (E)]
                           ▲
                           │
                   [R K Puram (SW)]
```

### How the Math Works in Simple English:
In [`src/features/spatial_graph.py`](file:///C:/Users/Pradeep%20Nagaraj/CPCB_AQI_Forecaster/src/features/spatial_graph.py), we built a 4-step dynamic graph:

1. **Exact GPS Distance (Haversine Formula):**
   * Computes the exact kilometer distance between every station pair across the curved Earth.
   * *Example:* Bawana to Anand Vihar is $30.5\ \text{km}$; Punjabi Bagh to ITO is $12.0\ \text{km}$.
   * **Physics Law:** Closer stations influence each other much more than distant ones (Inverse Distance Weighting: $W \propto 1/d$).

2. **Compass Bearing Angle ($\theta_{ij}$):**
   * Calculates the exact compass degree heading from Station A to Station B.
   * *Example:* From Bawana pointing to Anand Vihar is roughly $118^\circ$ (East-Southeast).

3. **Dynamic Wind Alignment ($\cos\Delta\theta$):**
   * At every hour, weather sensors measure wind direction (where wind blows *from*).
   * We calculate the **transport direction** (where smoke travels *to*):
     $$\theta_{\text{transport}} = (\theta_{\text{wind}} + 180^\circ) \bmod 360^\circ$$
   * We check if the smoke transport direction matches the line connecting two stations using cosine:
     $$\text{Alignment} = \max\Big(0, \cos(\theta_{\text{transport}} - \theta_{ij})\Big)$$
   * **If wind blows straight from Bawana to Anand Vihar:** Alignment is **$1.0$ (100% direct hit)**.
   * **If wind blows in the opposite direction:** Alignment is **$0.0$ (Zero smoke sent)**.

4. **The 3 Spatial Features Created:**
   * **`pm25_upwind_current`:** The weighted sum of smoke currently sitting at upwind stations blowing toward you.
   * **`pm25_upwind_lag3`:** The smoke level at upwind stations 3 hours ago (accounting for the time it takes the plume to physically travel $15\text{--}30\text{ km}$ across the city).
   * **`upwind_advection_flux`:** $\text{Wind Speed} \times \text{Incoming Smoke}$ (measures the total mass flow rate of incoming pollution).

---

## 3. Pillar 2: LightGBM Gradient Boosted Decision Trees

### Why Can't Linear Models Cut It?
A linear equation ($y = w_1 x_1 + w_2 x_2 + b$) must treat every $1\ \text{m/s}$ increase in wind speed identically.
* In reality, dropping from $5\ \text{m/s} \rightarrow 4\ \text{m/s}$ has minimal impact.
* But dropping from $1.5\ \text{m/s} \rightarrow 0.5\ \text{m/s}$ triggers a **catastrophic inversion trap**, trapping vehicle exhaust at street level and sending $PM_{2.5}$ through the roof.

### How LightGBM Thinks:
LightGBM builds an ensemble of **Gradient Boosted Decision Trees** ([`src/models/quantile_lightgbm.py`](file:///C:/Users/Pradeep%20Nagaraj/CPCB_AQI_Forecaster/src/models/quantile_lightgbm.py)) using "If-Then" threshold rules:

```
                          [Wind Speed < 1.5 m/s?]
                               /          \
                             YES          NO
                             /              \
             [Hour of Day >= 20?]       [Upwind Advection Flux > 500?]
                  /         \                    /            \
                YES         NO                 YES            NO
                /             \                /                \
        [Extreme Stagnation]  [Mild Spike]  [Plume Inbound]  [Normal Flow]
         (+85 μg/m³ spike)    (+20 μg/m³)    (+45 μg/m³)      (-10 μg/m³)
```

### Key Engineering Features in Our LightGBM Pipeline:
* **Leaf-Wise Tree Splitting (`num_leaves=31`):** Splitting the leaf with maximum loss reduction yields much sharper boundaries on extreme pollution peaks than traditional level-wise trees.
* **Histogram Binning (255 Bins):** Groups continuous sensor floats into discrete bins, speeding up training by $15\times$ while acting as a natural anti-overfitting regularizer.
* **Early Stopping on Validation Set:** Halts training as soon as 15 consecutive trees fail to improve out-of-sample validation loss.

---

## 4. Pillar 3: Multi-Horizon Quantile Regression ($P_{10}, P_{50}, P_{90}$)

### What is a Quantile Forecast?
Instead of guessing just the single "average" value, Tier 2 trains **three separate models per horizon**:

```
 350 μg/m³ ─── P90 (Worst-Case Upper Bound)   ─── 90% chance pollution stays below this
           ▲
           │ Uncertainty Band (MPIW = 61.5 μg/m³)
           ▼
 288 μg/m³ ─── P50 (Expected Median Forecast) ─── Most likely scenario
           ▲
           │
           ▼
 226 μg/m³ ─── P10 (Optimistic Lower Bound)   ─── 10% chance pollution drops below this
```

### The Math: Asymmetric Pinball Loss
How do you teach a machine learning model to predict the 90th percentile instead of the average?
You use **Pinball Loss** (Tilted Absolute Loss):

$$\mathcal{L}_\alpha(y, \hat{y}) = \begin{cases} 
\alpha (y - \hat{y}) & \text{if } y \ge \hat{y} \quad \text{(Under-prediction penalty)} \\ 
(1 - \alpha) (\hat{y} - y) & \text{if } y < \hat{y} \quad \text{(Over-prediction penalty)} 
\end{cases}$$

* For **$P_{90}$ ($\alpha = 0.90$):**
  * If the model under-predicts a severe smog spike, the penalty is **$0.90 \times \text{error}$** (punished 9 times harder!).
  * If the model over-predicts, the penalty is only **$0.10 \times \text{error}$**.
  * Result: The $P_{90}$ model naturally learns to output an aggressive, safe upper ceiling!

### Quantile Crossing Prevention (Monotonic Projection):
Because $P_{10}, P_{50}$, and $P_{90}$ are trained as separate models, statistical noise could occasionally cause $P_{10}$ to predict $245$ while $P_{50}$ predicts $240$ (a physical impossibility called **Quantile Crossing**).
Our pipeline enforces **Monotonic Sorting Projection**:
$$P_{10}^{\text{clean}} \le P_{50}^{\text{clean}} \le P_{90}^{\text{clean}}$$
guaranteeing mathematically valid uncertainty envelopes every single hour.

---

## 5. Step-by-Step Tier 2 Code Execution Flow

The entire Tier 2 workflow is automated across modular scripts:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        The Tier 2 Pipeline Flow                        │
├────────────────────────────────────────────────────────────────────────┤
│ 1. spatial_graph.py      Builds inter-station distance & bearing maps  │
│ 2. feature_builder.py    Injects upwind plume & advection features     │
│ 3. train_tier2.py        Tunes LightGBM architecture on validation set │
│ 4. quantile_lightgbm.py  Fits 6 models (P10, P50, P90 for 24h & 48h)   │
│ 5. evaluate_tier2.py     Evaluates test set, generates metrics & plots │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Step 1: Spatial Graph Generation (`src/features/spatial_graph.py`):**
   Loads the 5 station GPS coordinates and computes the $5 \times 5$ distance matrix and compass bearing matrix.

2. **Step 2: Tier 2 Feature Matrix (`src/features/feature_builder.py`):**
   Executes `generate_tier2_features(df)`, creating 27 total features (24 baseline features + 3 dynamic wind plume features).

3. **Step 3: Validation Tuning (`src/models/train_tier2.py`):**
   Compares **Shallow** (leaves=15, depth=4), **Standard** (leaves=31, depth=6), and **Deep** (leaves=63, depth=8) models on the validation split (`2025-10-28` to `2025-11-19`).
   * **Selected Architecture:** `Standard` (best generalization without overfitting).

4. **Step 4: Quantile Model Fitting:**
   Trains 6 LightGBM models:
   * Horizon 24h: $P_{10}, P_{50}, P_{90}$
   * Horizon 48h: $P_{10}, P_{50}, P_{90}$
   Saves models to `models/quantile_lgbm_h24.pkl` and `models/quantile_lgbm_h48.pkl`.

5. **Step 5: Rigorous Out-of-Sample Evaluation (`src/evaluation/evaluate_tier2.py`):**
   Evaluates predictions on the unseen test set (`2025-11-19` to `2025-12-12`).
   Exports:
   * `data/processed/tier2_test_predictions_h24.csv`
   * `data/processed/tier2_test_predictions_h48.csv`
   * `docs/tier2_vs_baseline_report.md`

---

## 6. How to Run Tier 2 Yourself

You can train and evaluate the full Tier 2 pipeline from the terminal in seconds:

```powershell
# 1. Run all unit tests to verify integrity (25 tests pass)
python -m pytest tests/

# 2. Train the multi-horizon quantile LightGBM models
python src/models/train_tier2.py

# 3. Run the comprehensive evaluation and benchmark comparison
python src/evaluation/evaluate_tier2.py
```

---

## 7. What's Next: The Bridge to Tier 3

While Tier 2 dramatically reduced prediction error, two crucial production challenges remain for **Tier 3**:
1. **Conformal Uncertainty Calibration:**
   * Tier 2's uncalibrated $[P_{10}, P_{90}]$ interval achieves **$66.9\%$** empirical coverage.
   * In Tier 3, we will apply **Conformal Prediction** to stretch the intervals so they mathematically guarantee **$\ge 80.0\%$ coverage** with zero distributional assumptions.
2. **Cost-Sensitive Bayesian Policy Thresholding:**
   * Missing an emergency is $3\times$ worse than a false alarm.
   * Tier 3 will optimize the exact decision threshold to minimize municipal economic and human health loss.
