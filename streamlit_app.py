import streamlit as st
import pandas as pd
import joblib
import json

st.set_page_config(page_title="Cricket Runs Predictor", layout="centered")

st.title("🏏 Cricket Player Next Match Runs Predictor")
st.write("This app predicts **next match runs** using a trained ML model.")

# -----------------------------
# Load model + features
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/final_model.joblib")
    with open("models/features.json", "r") as f:
        features = json.load(f)
    return model, features

model, features = load_artifacts()

st.sidebar.header("Enter Player Match Details")

# Inputs based on your training features
runs = st.sidebar.number_input("Runs (current match)", min_value=0, max_value=200, value=30)
balls_faced = st.sidebar.number_input("Balls Faced", min_value=1, max_value=150, value=25)
strike_rate = st.sidebar.number_input("Strike Rate", min_value=0.0, max_value=400.0, value=120.0)
avg_runs_last_5 = st.sidebar.number_input("Avg Runs Last 5 Matches", min_value=0.0, max_value=200.0, value=28.0)
inning = st.sidebar.selectbox("Inning", [1, 2])

# Create input DataFrame
input_data = {
    "runs": runs,
    "balls_faced": balls_faced,
    "strike_rate": strike_rate,
    "avg_runs_last_5": avg_runs_last_5,
    "inning": inning
}
input_df = pd.DataFrame([input_data])

st.subheader("📌 Input Data")
st.dataframe(input_df, width="stretch")


# Predict
if st.button("🎯 Predict Next Match Runs"):
    X = input_df[features]
    pred = model.predict(X)[0]

    if pred < 0:
        pred = 0

    st.success(f"✅ Predicted Next Match Runs: **{pred:.2f}**")

    # Interpretation
    if pred >= 50:
        st.balloons()
        st.write("🔥 High chance of a big score!")
    elif pred >= 30:
        st.write("💪 Solid performance expected.")
    else:
        st.write("🙂 Low/average score predicted.")
