# Cricket-Player-Performance-Prediction---Group-1-
## Week 1 & Week 2 – Data Acquisition, Cleaning, and EDA

## Overview
Weeks 1 and 2 of this project focus on understanding the IPL datasets, analyzing data quality, handling missing values, and performing exploratory data analysis (EDA). The objective is to prepare clean, reliable datasets that can be used for feature engineering and machine learning in later stages.

---

## Datasets Used
- **matches.csv** – Match-level details such as season, teams, venue, toss information, and winner.
- **deliveries.csv** – Ball-by-ball data containing runs, wickets, players involved, and over/ball information.

---

## Week 1 & 2 Tasks

### 1. Data Loading and Inspection
- Loaded both datasets using pandas.
- Examined dataset shapes, column names, and data types.
- Verified consistency of `match_id` between matches and deliveries datasets.

### 2. Missing Value Analysis
- Performed column-wise missing value analysis.
- Identified expected missing values:
  - Dismissal-related fields when no wicket occurred.
  - Extras-related fields when no extras were conceded.
- Identified and handled problematic missing values in core delivery columns.

### 3. Data Cleaning

#### Matches Dataset
- Filled missing `method` values with `Normal`.
- Filled missing `city` values with `Unknown`.
- Converted `date` column to datetime format.
- Standardized team names across seasons.
- Removed matches with no result (missing `winner`).

#### Deliveries Dataset
- Filled missing `fielder` values with `Not Applicable`.
- Filled missing `extras_type` values with `None`.
- Removed rows with missing critical fields such as:
  - over, ball, batter, bowler, run values, and wicket indicator.
- Standardized team names to match the matches dataset.

Cleaned datasets were saved as:
- `matches_clean.csv`
- `deliveries_clean.csv`

### 4. Exploratory Data Analysis (EDA)
Performed EDA to understand overall patterns and distributions:
- Matches played per season
- Top venues by number of matches
- Distribution of runs scored per ball
- Distribution of wickets per match
- Top batsmen by total runs
- Top bowlers by total wickets

These analyses highlight the high variability of T20 cricket and justify the use of machine learning models for performance prediction.

---

## Files Included
- `01_EDA.ipynb` – Contains data loading, cleaning steps, EDA visualizations, and observations.
- `data_cleaning.py` – Standalone, reproducible script for applying finalized data cleaning steps.

---

## Key Outcomes
- Cleaned and standardized IPL datasets.
- Clear understanding of data quality and distributions.
- Reproducible data preprocessing pipeline established.
- Strong foundation prepared for feature engineering and modeling in upcoming weeks.

---
## Week 3 & Week 4 – Feature Engineering and Preprocessing

# Overview

Weeks 3 and 4 focus on transforming the cleaned IPL datasets into a structured, machine-learning-ready format. Raw ball-by-ball cricket data is highly granular and cannot be directly used for prediction. Therefore, this phase aggregates delivery-level data to player-match level and engineers meaningful cricket-specific features such as recent form, venue performance, and career statistics. The objective is to create reliable input features and a future-aware target variable for predictive modeling.

# 1. Data Loading and Preparation
Loaded cleaned deliveries and matches datasets using pandas.
Standardized column names for consistency.
Merged match-level information (venue, season, date) with ball-by-ball data.
Converted match dates to datetime format.
Sorted data by player name and match date to preserve chronological order and avoid data leakage.

# 2. Player-Match Level Aggregation
Aggregated ball-by-ball data to player-match level.
Computed total runs scored and balls faced by each batsman in every match.
Calculated strike rate for each player-match instance.
This aggregation converts granular delivery data into a single performance record per player per match.

# 3. Recent Form Feature Engineering
Calculated rolling averages of runs scored over the last 5 matches.
Calculated rolling averages of runs scored over the last 10 matches.
Rolling features were computed player-wise using only past matches to maintain temporal correctness.
These features capture short-term performance trends that strongly influence T20 outcomes.

# 4. Venue-Based Performance Features
Calculated average runs scored by each batsman at each venue.
Merged venue-specific averages back into the player-match dataset.
This feature helps capture ground-specific performance variations due to pitch and boundary conditions.

# 5. Career Performance Metrics
Computed expanding (cumulative) career average runs for each batsman.
This feature represents long-term player consistency and experience.
Career metrics balance recent form and prevent overfitting to short-term fluctuations.

# 6. Target Variable Construction
Defined the prediction target as runs scored in the player’s next match.
Shifted the runs column forward by one match for each player to create the target variable.
Removed records without a subsequent match to ensure valid training examples.
This approach ensures the model always predicts future performance and avoids data leakage.

# 7. Feature Selection and Dataset Finalization
Selected final input features:
Recent form (last 5 and 10 matches)
Career average runs
Venue (categorical feature)
Defined the target variable as target_runs.
Saved the final feature-engineered dataset as:
dataset.csv
Files Included

# 02_FeatureEngineering.ipynb – Feature engineering logic, aggregation steps, and dataset preparation

dataset.csv – Final machine-learning-ready dataset used for model training

### Key Outcomes
Converted raw IPL data into structured player-match level observations.
Engineered cricket-specific features capturing form, venue influence, and career performance.
Ensured time-aware feature construction to prevent data leakage.
Prepared a clean and reproducible dataset ready for machine learning models.

### Week 5 & Week 6 – Model Training and Evaluation

### Overview
Weeks 5 and 6 focus on developing, training, and evaluating machine learning models to predict individual player performance in IPL matches. Using the feature-engineered dataset prepared in the previous phase, regression models are trained to estimate the number of runs a batsman is expected to score in an upcoming match. This phase also includes baseline comparison, model evaluation using standard regression metrics, and model interpretability analysis to understand feature influence.

### Dataset Used
dataset.csv – Feature-engineered player-match level dataset containing recent form, career statistics, venue information, and target variables.

### Week 5 & 6 Tasks

## 1. Dataset Loading and Preparation
Loaded the feature-engineered dataset (dataset.csv) using pandas.
Verified dataset shape, feature columns, and target variable.
Separated input features (X) and target variable (y).
Ensured chronological order was preserved to maintain time-series integrity.

## 2. Train–Test Split (Time-Aware)
Split the dataset into training and testing sets using a time-based approach.
Disabled random shuffling to prevent future match information from leaking into the training set.
Allocated 80% of data for training and 20% for testing.

### 3. Baseline Model Establishment
Implemented a simple baseline model using the 10-match rolling average of runs.
Used baseline predictions to establish a minimum performance benchmark.
Compared advanced machine learning models against this baseline to justify their effectiveness.

## 4. Model Development
The following regression models were trained:
Random Forest Regressor
Captures non-linear relationships between features.
Robust to outliers and feature interactions.
XGBoost Regressor
Gradient boosting–based model optimized for tabular data.
Handles complex feature interactions and improves predictive accuracy.
Each model was integrated with a preprocessing pipeline to ensure consistent feature scaling and encoding.

## 5. Model Evaluation
Evaluated models using standard regression metrics:
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
R² Score
Compared model performance against the baseline.
Identified the best-performing model based on predictive accuracy and generalization ability.

## 6. Model Explainability and Interpretation
Applied SHAP (SHapley Additive exPlanations) to interpret the XGBoost model.
Analyzed feature contribution at both global and individual prediction levels.
Identified key influencing factors such as recent form and career average.
These insights improve model transparency and trustworthiness.

## 7. Model Serialization and Storage
Serialized the best-performing model using joblib.
Saved the preprocessing pipeline separately to ensure consistent transformations during inference.
Generated reusable artifacts for deployment and future retraining.
Saved files include:
xgb_model.joblib
feature_pipeline.pkl
Files Included

## 03_ModelTraining.ipynb – Model training, evaluation, and explainability analysis

xgb_model.joblib – Trained machine learning model
feature_pipeline.pkl – Feature preprocessing pipeline

###  Key Outcomes

Successfully trained and evaluated regression models for player performance prediction.
Demonstrated performance improvements over a baseline rolling-average approach.
Gained insights into feature importance using SHAP analysis.
Generated deployable model artifacts for dashboard integration.
 ## Week 7 & Week 8 – Dashboard Development, Deployment, and Finalization

## Overview

Weeks 7 and 8 focus on transforming the trained machine learning model into an end-to-end usable system by building an interactive Streamlit dashboard, integrating the serialized model and preprocessing pipeline, and finalizing the project for deployment and presentation. This phase bridges model development with real-world usability, enabling users to explore player analytics and generate match-level performance predictions.

## 1. Streamlit Dashboard Development

Developed an interactive Streamlit application to serve as the project’s front-end interface. The dashboard allows users to:
Select batter and season for performance exploration
Visualize historical runs and strike-rate trends
Compare career average vs match performance
Input match parameters (balls faced, recent form, venue average, career average)
Generate predicted runs for upcoming matches

# Key UI components implemented:
Sidebar filters for player and season selection
Dynamic tables for match-level data inspection
Plotly-based visualizations (bar charts, line charts, scatter plots)
Prediction input forms and output display
The dashboard provides both analytical insights and predictive functionality in a single interface.

## 2. Model and Pipeline Integration

Integrated the trained XGBoost / Random Forest model (xgb_model.joblib) and preprocessing pipeline (feature_pipeline.pkl) into the Streamlit application.
Steps included:

Loading serialized artifacts using joblib
Applying consistent preprocessing during inference
Passing user inputs through the feature pipeline
Generating real-time predictions from the trained model
This ensures prediction consistency between training and deployment environments.

## 3. Analytical Visualization and Explainability

Enhanced the dashboard with analytical components to support interpretability:
Player form trends over recent matches
Career average vs current match performance plots
Feature importance visualization (optional SHAP integration)
These visual elements help users understand why specific predictions are produced and which features contribute most to performance outcomes.

## 4. Deployment and Testing

Performed local deployment using Streamlit and validated application functionality:

Verified correct loading of dataset and models

Tested multiple player inputs and edge cases

Ensured prediction stability and UI responsiveness

(Optional) Prepared the application for cloud deployment via Streamlit Cloud by organizing repository structure and dependency files.

## 5. Documentation and Repository Finalization

Final project cleanup and documentation included:

Organizing notebooks, scripts, datasets, and model artifacts

Writing comprehensive README.md documentation

Adding setup instructions and project structure

Preparing final submission-ready GitHub repository

## Key deliverables:

streamlit_app.py – Interactive prediction dashboard

dataset.csv – Feature-engineered dataset

xgb_model.joblib – Trained ML model

feature_pipeline.pkl – Preprocessing pipeline

Final README.md

## Files Included

streamlit_app.py – Streamlit dashboard for analytics and prediction

xgb_model.joblib – Serialized trained model

feature_pipeline.pkl – Feature preprocessing pipeline

dataset.csv – Final ML-ready dataset

## Key Outcomes

Successfully deployed an interactive cricket performance prediction dashboard.
Integrated machine learning models into a real-time inference pipeline.
Enabled user-driven analytics and predictions through Streamlit.
Completed full ML lifecycle: data → features → model → deployment.
Delivered a reproducible, well-documented end-to-end ML project suitable for academic evaluation and portfolio presentation.
