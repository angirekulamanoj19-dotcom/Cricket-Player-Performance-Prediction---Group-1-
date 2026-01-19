# IPL Team Performance Predictor

A Streamlit dashboard for predicting IPL team scores and wins using machine learning.

## Overview
This app uses an XGBoost model trained on IPL match data to predict:
- Next-match score for Team 1.
- Win probability and outcome for Team 1.

Inputs include rolling averages, venue stats, and career metrics. Visualizations include SHAP feature importance and prediction accuracy.

## Features
- **Inputs**: Sliders for team features (e.g., rolling wins, venue averages).
- **Predictions**: Score and win probability with charts.
- **Visualizations**: SHAP plots and performance metrics.
- **Model**: XGBoost regressor/classifier with preprocessing.

## Setup
1. Install dependencies: `pip install streamlit shap joblib pandas numpy scikit-learn xgboost`.
2. Place files: `streamlit_app.py`, `xgb_model.joblib`, `feature_pipeline.pkl`, `dataset.csv` in the same directory.
3. Run: `streamlit run streamlit_app.py`.

## Files
- `streamlit_app.py`: Main app code.
- `xgb_model.joblib`: Trained model.
- `feature_pipeline.pkl`: Preprocessor (scaler/encoder).
- `dataset.csv`: Feature-engineered data.
- `README.md`: This documentation.

## Deployment (Optional)
- Push to GitHub.
- Deploy to Streamlit Cloud: Connect repo, set main file to `streamlit_app.py`.
- Access live app at the provided URL.

## Model Performance
- RMSE: ~40-50 (on test data).
- R²: ~0.4-0.6.
- Top features: Rolling scores and venue averages.

For questions, refer to the project notebooks.