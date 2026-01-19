# IPL Player Performance Prediction – Streamlit App

## Project Overview
This project predicts the number of runs a batsman is likely to score in an IPL match using machine learning.
The model is trained on historical IPL data and deployed using a Streamlit web application.

## Model Used
- XGBoost Regressor
- Trained with engineered features such as recent form, venue performance, and opponent statistics

## Files
- streamlit_app.py : Streamlit application file
- xgb_model.joblib : Trained machine learning model

## How to Run the Application

1. Install required libraries:
   pip install streamlit pandas scikit-learn xgboost joblib

2. Run the Streamlit app:
   streamlit run streamlit_app.py

## Inputs
- Venue
- Batting team
- Opponent team
- Season
- Recent batting averages (last 3, 5, 10 matches)
- Career, venue, and opponent averages

## Output
- Predicted number of runs for the batsman in the selected match context
