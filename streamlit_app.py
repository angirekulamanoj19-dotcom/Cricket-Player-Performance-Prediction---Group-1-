import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="IPL Player Performance Predictor")

@st.cache_resource
def load_model():
    model = joblib.load("xgb_model.joblib")
    preprocessor = joblib.load("feature_pipeline.pkl")
    return model, preprocessor

model, preprocessor = load_model()

st.title("🏏 IPL Player Performance Prediction")

batting_team = st.selectbox("Batting Team", ["MI", "CSK", "RCB", "KKR", "SRH", "DC", "PBKS", "RR", "GT", "LSG"])
bowling_team = st.selectbox("Bowling Team", ["MI", "CSK", "RCB", "KKR", "SRH", "DC", "PBKS", "RR", "GT", "LSG"])
venue = st.text_input("Venue")

avg_runs_last_5 = st.number_input("Average Runs (Last 5 matches)", 0.0)
avg_runs_last_10 = st.number_input("Average Runs (Last 10 matches)", 0.0)
venue_avg_runs = st.number_input("Venue Average Runs", 0.0)
pvt_avg_runs = st.number_input("Opponent Avg Runs", 0.0)
career_avg_runs = st.number_input("Career Avg Runs", 0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "avg_runs_last_5": avg_runs_last_5,
        "avg_runs_last_10": avg_runs_last_10,
        "venue_avg_runs": venue_avg_runs,
        "pvt_avg_runs": pvt_avg_runs,
        "career_avg_runs": career_avg_runs,
        "batting_team": batting_team,
        "bowling_team": bowling_team,
        "venue": venue
    }])

    X_processed = preprocessor.transform(input_df)
    prediction = model.predict(X_processed)[0]

    st.success(f"🏏 Predicted Runs Next Match: {prediction:.2f}")
