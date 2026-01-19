import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="IPL Runs Predictor", layout="centered")

st.title("🏏 IPL Batsman Runs Predictor")

# Load trained model
model = joblib.load("xgb_model.joblib")

st.subheader("Match Details")
venue = st.text_input("Venue", "Wankhede Stadium")
batting_team = st.text_input("Batting Team", "Mumbai Indians")
opponent_team = st.text_input("Opponent Team", "Chennai Super Kings")
season = st.number_input("Season", min_value=2007, max_value=2025, value=2019)

st.subheader("Player Form")
avg_last_3 = st.number_input("Average runs (last 3 matches)", 0.0, 100.0, 25.0)
avg_last_5 = st.number_input("Average runs (last 5 matches)", 0.0, 100.0, 30.0)
avg_last_10 = st.number_input("Average runs (last 10 matches)", 0.0, 100.0, 28.0)
career_avg = st.number_input("Career average", 0.0, 100.0, 32.0)
venue_avg = st.number_input("Venue average", 0.0, 100.0, 35.0)
opponent_avg = st.number_input("Opponent average", 0.0, 100.0, 27.0)

if st.button("Predict Runs"):
    input_df = pd.DataFrame([{
    "venue": venue,
    "batting_team": batting_team,
    "opponent_team": opponent_team,
    "team1": batting_team,          
    "team2": opponent_team,         
    "season": season,

    "avg_last_3": avg_last_3,
    "avg_last_5": avg_last_5,
    "avg_last_10": avg_last_10,

    "career_avg": career_avg,
    "venue_avg": venue_avg,
    "opponent_avg": opponent_avg,

    "avg_at_venue": venue_avg,      
    "avg_vs_opponent": opponent_avg 
}])

    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Runs: {prediction:.1f}")
