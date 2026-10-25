# The Simple English Guide: Understanding Tier 1 and Tier 2

> **A clear, beginner-friendly, and complete walkthrough of what we have built so far (Tier 1) and what we are upgrading next (Tier 2).**

---

## 1. The Big Picture: What Problem Are We Solving?

Every winter, Delhi and Northern India experience some of the most dangerous air pollution on Earth. 

* The air fills with tiny toxic particles called **$PM_{2.5}$** (particles so small they enter your lungs and bloodstream).
* Normal safe air has a $PM_{2.5}$ level below **$30\ \mu\text{g/m}^3$**.
* During Delhi winters, $PM_{2.5}$ regularly explodes past **$250\text{--}400\ \mu\text{g/m}^3$** (creating a health emergency).

### The Real-World Dilemma:
The Government of India has an emergency action plan called **GRAP (Graded Response Action Plan)**. Under Stage IV of this plan, authorities can ban diesel trucks from entering Delhi and shut down schools and construction.

**The Problem:** Today, the government acts **reactively**—they wait until the monitors are already reading toxic air before ordering bans. By then, millions of people have already breathed the smoke.

**Our Goal:** Build an intelligent Machine Learning system that can predict air quality **24 to 48 hours in advance**, giving authorities the advance notice they need to stop emergencies before they happen.

---

## 2. Part 1: What We Did in Tier 1 (The Baseline)

In Tier 1, our goal was to build a clean, working, and honest **Baseline Model**. 

Think of a baseline as the "score to beat." Before building anything fancy, we must build a simple, reliable model to see how well standard statistics perform.

Here is the step-by-step story of Tier 1:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        The 7 Steps of Tier 1                           │
├────────────────────────────────────────────────────────────────────────┤
│  Step 1: Project Setup (Git, Clean Folders, Test Suite)                │
│  Step 2: Data Ingestion (Collecting Delhi CPCB Sensor Readings)        │
│  Step 3: Data Quality (Cleaning Broken Sensors & Filling Small Gaps)   │
│  Step 4: Feature Engineering (Creating 24 Clues for the Model)         │
│  Step 5: Time Splitting & Scaling (Strictly Separating Past & Future)  │
│  Step 6: Model Training (Ridge Regression: Smart Straight Line)        │
│  Step 7: Evaluation (Testing on Unseen Future Data & Finding Gaps)     │
└────────────────────────────────────────────────────────────────────────┘
```

### The 5 Stations We Used
We tracked 5 major continuous monitoring stations across Delhi from **September 1 to December 31, 2025**:
1. **Bawana** (Far North-West industrial area)
2. **Punjabi Bagh** (West Delhi traffic & residential area)
3. **ITO** (Central Delhi heavy traffic intersection)
4. **R K Puram** (South Delhi residential area)
5. **Anand Vihar** (East Delhi bus terminal, very polluted)

---

### How We Cleaned the Data (Step 2 & 3)
Raw government sensors are messy. They lose power, freeze, or spit out bad numbers. We built automatic cleaning rules:
* **Purged Negative & Impossible Numbers:** $PM_{2.5}$ cannot be negative, and it cannot exceed $1,000$.
* **Fixed Broken Physics:** $PM_{2.5}$ is physically smaller than $PM_{10}$ (it is a subset). If a sensor claimed $PM_{2.5} > PM_{10}$, we fixed it.
* **Purged Frozen Sensors:** If a sensor reported the exact same number for 6 hours in a row, we flagged it as jammed/frozen and deleted the fake readings.
* **Filled Only Small Gaps:** If a sensor went down for 1 to 3 hours, we connected the dots with a smooth line. If it went down for more than 3 hours, we left it blank so we didn't invent fake data.

---

### The Clues We Gave the Model (Features in Step 4)
To predict tomorrow's air, we gave the model **24 clues** (features) based on what has happened so far:
1. **Recent Memory (Lags):** What was the pollution right now ($t$), 1 hour ago, 2 hours ago, and 6 hours ago?
2. **Yesterday's Air (`pm25_lag_24`):** What was the pollution at the exact same hour yesterday? (This turned out to be our **strongest clue**, because human traffic and solar cycles repeat every 24 hours).
3. **Two Days Ago (`pm25_lag_48`):** What was the pollution 48 hours ago?
4. **24-Hour Average & Volatility:** The average smog level over the past full day.
5. **The Weather (Wind & Temp):** 
   * Cold air traps smoke near the ground (**thermal inversion**).
   * Fast wind blows smoke away; calm wind traps it.
6. **Combustion Clues (`co`):** Carbon monoxide comes from tailpipes and burning garbage; it acts like a smoke tracer.
7. **The Clock & Calendar:** We converted the hour (0 to 23) and day of the year into smooth repeating waves ($\sin$ and $\cos$).

---

### What Was the Model Trying to Predict? (The Target)
* **The Target:** The exact $PM_{2.5}$ number **24 hours into the future** at that specific station.
* **The Emergency Flag:** Whether that future number crosses the emergency limit of **$250\ \mu\text{g/m}^3$** (GRAP Stage IV).

---

### The Model We Used in Tier 1: Ridge Regression
* What is Ridge Regression in plain English?
  * It is a **smart straight line**. It draws a straight mathematical formula connecting the 24 clues to the target.
  * The "Ridge" part (called **$L_2$ Regularization**) is a mathematical penalty that prevents any single clue from getting too extreme or crazy.
  * Because some clues are in huge numbers (PM2.5: $0\text{--}500$) and some are in tiny numbers (CO: $0\text{--}4$), we **scaled** every feature so they all have an average of 0 and a standard deviation of 1. This gave every clue a completely fair playing field.

---

### What Did Tier 1 Achieve? (The Results)
We tested our model on **unseen future test data from late November to mid-December 2025**:
* **MAE (Mean Absolute Error):** **$22.22\ \mu\text{g/m}^3$**. (On average, our 24-hour prediction was off by only 22 units).
* **$R^2$ Score:** **$0.8221$** (Our straight-line model explained **$82.2\%$** of the variation in future air quality).
* **Emergency Alert Recall:** **$100\%$** (It successfully caught every single severe emergency episode).

---

### The 3 Big Flaws of Tier 1 (Why We Cannot Stop Here)

Despite good numbers, Tier 1 has three major weaknesses that prevent it from being a true production system:

1. **Flaw 1: The Stations Are In Silos (No Communication)**
   * In Tier 1, Anand Vihar only looked at Anand Vihar. Bawana only looked at Bawana.
   * But wind blows smoke from Bawana (North-West) to Anand Vihar (East)! Tier 1 had no way of knowing a smoke cloud was traveling across the city.
2. **Flaw 2: A Straight Line Cannot Handle a Sudden Cliff**
   * When we looked at our errors (residual diagnostics), we found that whenever wind speed dropped below a slow walk ($<1.5\ \text{m/s}$), the linear model **under-predicted pollution by an average of $-14.6\ \mu\text{g/m}^3$**.
   * Why? Because calm air causes pollution to explode exponentially like a cliff. A straight line cannot bend to match an exponential explosion!
3. **Flaw 3: It Only Gives One Guess (No Risk Envelope)**
   * Tier 1 predicts a single number (e.g. *"Tomorrow will be $242\ \mu\text{g/m}^3$"*).
   * It cannot tell a government official: *"I expect 242, but there is a 10% risk it could hit 310 if the wind dies down."* Without knowing the worst-case risk, officials cannot make safe emergency decisions.

---

## 3. Part 2: What We Are Upgrading in Tier 2 (The Advancements)

In Tier 2, we fix all three flaws by introducing **The Primary Non-Linear Engine**:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        The 4 Upgrades of Tier 2                        │
├────────────────────────────────────────────────────────────────────────┤
│  Upgrade 1: The Wind-Directed Spatial Graph (Stations Talk to Each Other)│
│  Upgrade 2: Non-Linear Decision Trees (LightGBM Replaces Straight Line)│
│  Upgrade 3: Probabilistic Quantile Ranges (P10, P50, P90 Output)       │
│  Upgrade 4: Two Forecasting Horizons (24 Hours and 48 Hours Ahead)     │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Upgrade 1: Making the Stations Talk (The Wind-Directed Spatial Graph)

#### The Intuition:
Imagine someone starts a huge smoky campfire 5 kilometers to the west of your house.
* If the wind is blowing **towards the east** (straight at you), your house will fill with smoke in 30 minutes.
* If the wind is blowing **towards the north** (away from you), you will smell nothing.

#### What We Do in Tier 2:
For every station, we calculate:
1. **Who is upwind right now?** We check wind direction to see which other monitoring stations are currently blowing their air toward this station.
2. **How far away are they?** We calculate the distance in kilometers.
3. **The Upwind Clue:** We create two new features:
   * `pm25_upwind_current`: The average pollution currently being measured at all stations blowing toward you.
   * `pm25_upwind_lag3`: The pollution measured upwind 3 hours ago (the smoke that is arriving right now!).

Now, when Anand Vihar predicts its air, it doesn't just look at itself—it looks at the toxic smoke currently heading its way from Bawana and Punjabi Bagh!

---

### Upgrade 2: Switching to Decision Trees (LightGBM)

#### The Intuition:
Instead of drawing one rigid straight line (Ridge), we use **Gradient Boosted Decision Trees (LightGBM)**.

Think of a decision tree as a series of smart "IF-THEN" questions:
* *IF wind speed is $< 1.5\ \text{m/s}$...*
* *AND temperature is $< 12^\circ\text{C}$ (freezing night)...*
* *AND upwind pollution is $> 300\ \mu\text{g/m}^3$...*
* *THEN: Add $+90\ \mu\text{g/m}^3$ to the forecast immediately!*

Trees naturally capture **non-linear threshold cliffs** and solve the calm-wind under-prediction problem that broke Tier 1.

---

### Upgrade 3: Predicting a Range Instead of a Single Guess (Quantile Loss)

#### The Intuition:
When you check a weather app, you don't want it to say *"Tomorrow it will rain exactly 14.2 mm."* 
You want to know: *"There is a 90% chance of heavy rain, so take an umbrella."*

In Tier 2, instead of predicting just one number, our model predicts **three numbers at once**:
1. **$P_{10}$ (10th Percentile - Optimistic Bound):** *"Even if weather conditions clear up, pollution won't drop below this floor."*
2. **$P_{50}$ (50th Percentile - Expected Median):** *"This is the most realistic expected value."*
3. **$P_{90}$ (90th Percentile - Worst-Case Tail Risk):** *"If wind completely dies down, pollution could spike this high."*

#### Why Government Officials Need This:
If $P_{50}$ is $235\ \mu\text{g/m}^3$ (under the emergency limit), but $P_{90}$ is **$315\ \mu\text{g/m}^3$** (severe toxic zone), the government knows there is a dangerous **10% to 20% tail risk**. They can declare emergency truck bans 24 hours in advance and save lives.

---

### Upgrade 4: Looking 48 Hours Ahead (Multi-Horizon)
We train separate model heads for:
* **$t+24$ Hours Ahead:** For immediate overnight emergency alerts.
* **$t+48$ Hours Ahead:** For strategic planning (giving schools and bus companies 2 full days to prepare).

---

## 4. Part 3: Quick Summary Comparison

| Feature / Capability | Tier 1 (Baseline) | Tier 2 (Primary Engine) | Why It Matters |
| :--- | :--- | :--- | :--- |
| **Model Type** | Ridge Regression (Linear) | LightGBM (Gradient Boosted Trees) | Can handle sudden non-linear atmospheric crashes. |
| **Cross-Station Vision** | None (Stations in silos) | **Wind-Directed Spatial Graph** | Tracks smoke plumes traveling across Delhi. |
| **Output Type** | Single guess (Point Mean) | **Quantile Range ($P_{10}, P_{50}, P_{90}$)** | Gives authorities worst-case risk envelopes. |
| **Forecast Horizons** | 24 Hours only | **24 Hours AND 48 Hours** | Gives 2 full days of regulatory planning time. |
| **Stagnant Wind Error** | Severe under-prediction ($-14.6$) | **Eliminated by decision splits** | Corrects the calm-wind blindspot. |

---

## 5. Plain English Glossary

* **$PM_{2.5}$:** Microscopic toxic dust and soot smaller than $2.5$ micrometers that enters the human bloodstream.
* **CAAQMS:** Continuous Ambient Air Quality Monitoring Stations (the physical government kiosks with air sensors).
* **GRAP:** Graded Response Action Plan (India's legal rules for shutting down trucks/schools when air gets bad).
* **Thermal Inversion:** When warm air acts like a heavy blanket over cold surface air, trapping smoke at breathing level.
* **MAE (Mean Absolute Error):** The average number of points your model's prediction is off by. Lower is better.
* **$R^2$ Score:** A score from $0$ to $1$ showing how much of the air's real behavior your model explained. $1.0$ is perfect; $0.82$ is very strong.
* **Quantile Loss (Pinball Loss):** A mathematical formula that trains a model to predict percentiles ($P_{10}, P_{50}, P_{90}$) instead of just the average.
* **Data Leakage:** An engineering mistake where a model accidentally peeks at tomorrow's test data during training. We strictly prevent this.
