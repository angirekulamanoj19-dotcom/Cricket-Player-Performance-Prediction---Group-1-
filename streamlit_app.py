import streamlit as st
import pandas as pd
import joblib


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="IPL Batsman Runs Predictor",
    page_icon="🏏",
    layout="centered"
)

# -----------------------------
# Load model and pipeline
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("models/batsman_runs_model.pkl")
    pipeline = joblib.load("models/featurepipeline.pkl")
    return model, pipeline

model, pipeline = load_model()

# -----------------------------
# App title & description
# -----------------------------
st.title("🏏 IPL Batsman Runs Prediction App")
st.markdown(
    """
    This application predicts the **expected runs scored by a batsman**
    in the **next IPL match** using historical performance statistics.
    """
)

st.divider()

# -----------------------------
# Sidebar - User Inputs
# -----------------------------
st.sidebar.header("📊 Enter Match Statistics")

avg_runs_last_5 = st.sidebar.slider(
    "Average Runs (Last 5 Matches)",
    min_value=0.0,
    max_value=150.0,
    value=30.0,
    help="Average runs scored in the last 5 matches"
)

venue_avg_runs = st.sidebar.slider(
    "Average Runs at Venue",
    min_value=0.0,
    max_value=150.0,
    value=28.0
)

opponent_avg_runs = st.sidebar.slider(
    "Average Runs vs Opponent",
    min_value=0.0,
    max_value=150.0,
    value=25.0
)

career_avg_runs = st.sidebar.slider(
    "Career Average Runs",
    min_value=0.0,
    max_value=150.0,
    value=32.0
)

matches_played = st.sidebar.number_input(
    "Matches Played",
    min_value=1,
    max_value=500,
    value=50
)

strike_rate = st.sidebar.slider(
    "Strike Rate",
    min_value=50.0,
    max_value=250.0,
    value=135.0
)

# -----------------------------
# Main section
# -----------------------------
st.subheader("🔍 Input Summary")

input_df = pd.DataFrame([{
    "avg_runs_last_5": avg_runs_last_5,
    "venue_avg_runs": venue_avg_runs,
    "opponent_avg_runs": opponent_avg_runs,
    "career_avg_runs": career_avg_runs,
    "matches_played": matches_played,
    "strike_rate": strike_rate
}])

st.dataframe(input_df, use_container_width=True)

st.divider()

# -----------------------------
# Prediction
# -----------------------------
if st.button("🚀 Predict Runs", use_container_width=True):
    processed_data = pipeline.transform(input_df)
    prediction = model.predict(processed_data)

    st.success("✅ Prediction Successful!")

    st.markdown(
        f"""
        ### 🏆 Predicted Runs in Next Match  
        ## **{prediction[0]:.2f} runs**
        """
    )

    st.caption("⚠️ Prediction is based on historical data and model assumptions.")

from sklearn.metrics import mean_squared_error, r2_score

@st.cache_data
def load_evaluation_metrics():
    df = pd.read_csv("data/final_batsman_features.csv")

    X = df.drop("runs_next_match", axis=1)
    y = df["runs_next_match"]

    y_pred = model.predict(X)

    mse = mean_squared_error(y, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y, y_pred)

    return mse, rmse, r2

st.divider()
st.subheader("📈 Model Evaluation Metrics")

mse, rmse, r2 = load_evaluation_metrics()

col1, col2, col3 = st.columns(3)

col1.metric("MSE", f"{mse:.3f}")
col2.metric("RMSE", f"{rmse:.3f}")
col3.metric("R² Score", f"{r2:.3f}")

st.caption("Metrics calculated on the dataset used during model training.")
