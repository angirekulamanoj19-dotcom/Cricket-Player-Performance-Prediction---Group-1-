# 🏏 Cricket Player Performance Prediction (IPL)

## 📌 Project Overview
This project predicts a cricket player's **next match performance (runs scored)** using Machine Learning.  
We used IPL ball-by-ball data to engineer player match-level features and trained ML models to forecast future runs.  
A Streamlit dashboard is also built for interactive prediction.

---

## 🎯 Objective
To build a system that can:
- analyze player performance using historical IPL match data
- engineer meaningful features (recent form, strike rate, etc.)
- train ML model(s) to predict **next match runs**
- provide predictions through a **Streamlit web dashboard**

---

## 🗂 Dataset Description
We used IPL **ball-by-ball deliveries dataset**.

Main dataset file:
- `deliveries.csv` → contains ball-by-ball records

Key columns:
- `match_id`, `inning`, `over`, `ball`
- `batting_team`, `bowling_team`
- `batter`, `bowler`
- `batsman_runs`, `total_runs`
- `dismissal_kind`, `player_dismissed`

---

## ✅ Project Milestones / Tasks

### ✅ Task 1: Data Acquisition & EDA
- Loaded ball-by-ball dataset (`deliveries.csv`)
- Performed Exploratory Data Analysis:
  - runs distribution
  - wicket type distribution
  - team performance (batting & bowling)
  - venue analysis (optional)

Files:
- `eda.ipynb`
- `eda_analysis.py`

---

### ✅ Task 2: Feature Engineering & Preprocessing
Converted ball-by-ball data into **player-match level dataset**.

Created features:
- `runs` (total runs per player per match)
- `balls_faced`
- `strike_rate`
- `avg_runs_last_5` (recent form)
- `inning`
- match context (`batting_team`, `bowling_team`)
- target variable:
  - `target_runs` = next match runs (per player)

Output file:
- `processed_player_dataset.csv`

Notebook:
- `feature_engineering.ipynb`

---

### ✅ Task 3: Model Development & Evaluation
Models trained:
1. **Linear Regression (Baseline)**
2. **Random Forest Regressor**

Final selected model:
✅ **Linear Regression**, because it achieved better generalization performance.

Evaluation Metrics (Linear Regression):
- MAE: **15.25**
- RMSE: **19.80**
- R² Score: **0.084**

Notebook:
- `model_training.ipynb`

Saved Model Files:
- `models/final_model.joblib`
- `models/features.json`

---

## 🖥 Streamlit Dashboard
A Streamlit web application is built to allow interactive prediction of next match runs.

File:
- `streamlit_app.py`

User inputs used:
- Runs
- Balls faced
- Strike rate
- Average runs (last 5 matches)
- Inning

---

## ▶️ How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
