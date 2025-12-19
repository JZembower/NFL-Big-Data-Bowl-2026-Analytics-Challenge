# 🏈 NFL Big Data Bowl 2025: Sideline Decision System
**From Raw Tracking Data to Coach-Ready Intelligence**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![XGBoost](https://img.shields.io/badge/XGBoost-Classification-orange)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Viz-green)

## 📋 Overview
The **Sideline Decision System** bridges the gap between complex analytics and game-day execution. By processing **18,009 plays** and **4.5M+ tracking frames**, this project delivers two distinct products:
1.  **For Coaches:** A "Cheat Sheet" of high-probability play calls based on specific defensive looks.
2.  **For Data Scientists:** A robust ML pipeline featuring physics-informed feature engineering and trajectory forecasting.

---

## 🧢 For Coaches: The Game Plan
*Derived from `Redzone.ipynb`, `Two_Minute Drill.ipynb`, and `Coaching.ipynb`.*

### 🚨 Situation-Specific Alerts

#### 🔴 Red Zone Mastery
* **The "Free Yards" Look:** Attack **Cover 2 Zone** (74.6% completion rate) with seam routes and corners.
* **Play Action Boost:** use Play Action to gain **+6.3% completion** and **+0.192 EPA**.
* **Yardage Strategy:** * *Need Probability (3rd/4th Down)?* Throw Short (0-5 yds) → **76.5% Completion**.
    * *Need Value (1st/2nd Down)?* Throw Deep (16+ yds) → **0.607 EPA** (vs 0.085 for short).

#### ⏱️ Two-Minute Drill
* **The "Avoid" List:** Do NOT throw into **Cover 1 Man** (56.2% completion). If seen, call timeout.
* **Clock Management:** Sideline routes stop the clock; middle routes don't.
* **Coverage exploit:** **Cover 2 Zone** yields **75.8% completion**, the highest in hurry-up situations.

#### ⚠️ Critical Downs (3rd & 4th)
* **4th Down Cheat Code:** **Play Action** increases completion by **+14.9%** (56.5% → 71.4%) and adds **+0.84 EPA**.
* **3rd & Long:** **Play Action** actually *lowers* completion (-1.4%) but increases big-play potential (**+0.417 EPA**).

### 🧠 The "Trust" Factor
Analytics confirms chemistry beats coverage. We identified **10 QB-WR pairs** with **100% completion rates** in high-leverage Red Zone and 2-Minute situations. When in doubt, feed the primary target.

---

## 💻 For Data Scientists: The Architecture
*Derived from `02_features.ipynb`, `03_modeling_route.ipynb`, and `04_modeling_completions.ipynb`.*

### 1. Feature Engineering (140+ Features)
We moved beyond standard metrics to engineer physics-informed features:
* **Micro-Movements:** `jerk`, `directional_change_rate` to quantify route sharpness.
* **Contextual Pressure:** `defensive_box_density`, `pocket_integrity` metrics.
* **Spatial Relationships:** Voronoi tessellation volumes for receiver openness.

### 2. Modeling Pipeline
We utilized a multi-stage approach to predict play outcomes:
* **Route Classification:** PyTorch-based deep learning model (`AdamW` optimizer, `ReduceLROnPlateau`) to classify and predict receiver trajectories.
* **Completion Probability:** XGBoost Classifier achieved ~**0.82 Mean F1 Score** in Stratified K-Fold cross-validation, outperforming Random Forest and Gradient Boosting.

---

## 📂 Project Structure

| Directory / File | Description | Audience |
| :--- | :--- | :--- |
| **`notebooks/`** | | |
| `08_Final_Sideline_Suggestions.ipynb` | **Master Dashboard.** The interactive decision matrix and summary. | 🧢 Coach |
| `Redzone.ipynb` | Deep dive into Red Zone EPA, completion, and "Perfect Pairs". | 🧢 Coach |
| `Two_Minute Drill.ipynb` | Analysis of clock management and hurry-up coverage exploits. | 🧢 Coach |
| `Coaching.ipynb` | General decision-making framework (4th Down, Play Action). | 🧢 Coach |
| `01_eda.ipynb` | Data dictionary, distribution analysis, and quality checks. | 💻 Data Sci |
| `02_features.ipynb` | Engineering of 140+ features (physics, situation, spacing). | 💻 Data Sci |
| `03_modeling_route.ipynb` | PyTorch implementation for route trajectory modeling. | 💻 Data Sci |
| `04_modeling_completions.ipynb` | ML classifiers (XGBoost/RF) for catch probability. | 💻 Data Sci |
| **`src/`** | | |
| `feature_engineering.py` | Production-ready scripts for feature extraction. | 💻 Data Sci |

---

## 🚀 Quick Start

**Prerequisites:**
* Python 3.9+
* Libraries: `torch`, `xgboost`, `pandas`, `plotly`, `seaborn`

**Reproducing the Analysis:**
1.  **For the Final Report:** Run `notebooks/08_Final_Sideline_Suggestions.ipynb` for the interactive dashboard.
2.  **For Specific Scenarios:** Run `Redzone.ipynb`, `Two_Minute Drill.ipynb`, or `Coaching.ipynb` for targeted insights.
3.  **For Model Training:** Execute the pipeline: `02_features.ipynb` → `03_modeling_route.ipynb` → `04_modeling_completions.ipynb`.

---

*Built by Jonah Zembower, Brady Nolin, Alon Stein | Carnegie Mellon Heinz Sports Analytics | Dec 2025*
