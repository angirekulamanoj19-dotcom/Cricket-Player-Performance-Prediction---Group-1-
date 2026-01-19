
# Cricket Player Performance Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import os
# PAGE CONFIG
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    page_icon=" _🏏  ",
    layout="wide"
)

st.title("Cricket Player Performance Prediction")
st.caption(
    "Predict expected runs for batsmen and expected wickets for bowlers "
    "using machine learning models trained on IPL data."
)

# LOAD MODELS & PIPELINE
MODEL_DIR = "models"


@st.cache_resource
def load_models():
    batsman_model = joblib.load(os.path.join(
        MODEL_DIR, "xgb_batsman_model.joblib"))
    bowler_model = joblib.load(os.path.join(
        MODEL_DIR, "xgb_bowler_model.joblib"))
    feature_pipe = joblib.load(os.path.join(MODEL_DIR, "feature_pipeline.pkl"))
    return batsman_model, bowler_model, feature_pipe


batsman_model, bowler_model, feature_pipeline = load_models()

preprocessor_batsman = feature_pipeline["preprocessor_batsman"]
preprocessor_bowler = feature_pipeline["preprocessor_bowler"]

cat_cols_batsman = feature_pipeline["cat_cols_batsman"]
num_cols_batsman = feature_pipeline["num_cols_batsman"]

cat_cols_bowler = feature_pipeline["cat_cols_bowler"]
num_cols_bowler = feature_pipeline["num_cols_bowler"]

st.success("Models and feature pipeline loaded successfully")

# LOAD DATASETS for UI


@st.cache_data
def load_data():
    batsman_df = pd.read_csv("batsman_model_dataset.csv", parse_dates=["date"])
    bowler_df = pd.read_csv("bowler_model_dataset.csv", parse_dates=["date"])
    return batsman_df, bowler_df


batsman_df, bowler_df = load_data()
# HELPER FUNCTIONS


def normalize_text(x):
    return x.strip() if isinstance(x, str) else x


def clean_feature_name(name):
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ")
    return name.title()


# SIDEBAR INPUTS
st.sidebar.header("Prediction Inputs")

role = st.sidebar.radio("Select Role", ["Batsman", "Bowler"])

if role == "Batsman":
    df = batsman_df.copy()
    player_col = "batter"
else:
    df = bowler_df.copy()
    player_col = "bowler"

# Normalize text columns
for col in ["venue", "batting_team", "bowling_team", player_col]:
    df[col] = df[col].apply(normalize_text)

player = st.sidebar.selectbox("Select Player", sorted(df[player_col].unique()))
venue = st.sidebar.selectbox("Select Venue", sorted(df["venue"].unique()))
batting_team = st.sidebar.selectbox(
    "Batting Team", sorted(df["batting_team"].unique()))
bowling_team = st.sidebar.selectbox(
    "Bowling Team", sorted(df["bowling_team"].unique()))

show_shap = st.sidebar.checkbox("🔍 Show Feature Contribution (SHAP)")
# BUILD INPUT ROW
latest = (
    df[df[player_col] == player]
    .sort_values("date")
    .iloc[-1]
)

input_data = {
    "venue": venue,
    "batting_team": batting_team,
    "bowling_team": bowling_team,
    player_col: player
}
# Add numerical features
num_cols = num_cols_batsman if role == "Batsman" else num_cols_bowler

for col in num_cols:
    input_data[col] = latest[col] if col in latest else 0

input_row = pd.DataFrame([input_data])
# PREDICTION

predict_btn = st.button(" Predict Performance")

if predict_btn:

    if role == "Batsman":
        raw_pred = batsman_model.predict(input_row)[0]
        pred = int(round(raw_pred))

        std = df[df[player_col] == player]["runs_last_10"].std()
        std = 8 if pd.isna(std) else std

        lower = max(0, int(round(pred - std)))
        upper = int(round(pred + std))

        st.success(f"Expected Runs: **{pred}**")
        st.caption(f"Confidence Range: **{lower} – {upper} runs**")

    else:
        raw_pred = bowler_model.predict(input_row)[0]
        pred = max(0, int(round(raw_pred)))

        std = df[df[player_col] == player]["wickets_last_10"].std()
        std = 1 if pd.isna(std) else std

        lower = max(0, int(round(pred - std)))
        upper = int(round(pred + std))

        st.success(f" Expected Wickets: **{pred}**")
        st.caption(f"Confidence Range: **{lower} – {upper} wickets**")
# SHAP FEATURE CONTRIBUTION
if predict_btn and show_shap:

    st.markdown("---")
    st.subheader("Feature Contribution (SHAP)")
    st.caption("How each feature influenced the prediction")

    if role == "Batsman":
        X_trans = preprocessor_batsman.transform(input_row)
        model_only = batsman_model.named_steps["model"]
        feature_names = preprocessor_batsman.get_feature_names_out()
    else:
        X_trans = preprocessor_bowler.transform(input_row)
        model_only = bowler_model.named_steps["model"]
        feature_names = preprocessor_bowler.get_feature_names_out()

    explainer = shap.TreeExplainer(model_only)
    shap_values = explainer.shap_values(X_trans)[0]

    shap_df = pd.DataFrame({
        "Feature": [clean_feature_name(f) for f in feature_names],
        "Impact": shap_values
    }).sort_values("Impact", key=abs, ascending=False).head(10)

    st.bar_chart(shap_df.set_index("Feature")["Impact"])

    with st.expander("📄 View SHAP values"):
        st.dataframe(
            shap_df.reset_index(drop=True)
            .style.format({"Impact": "{:.2f}"})
        )
# FOOTER
st.markdown("---")
st.caption(
    "This dashboard uses XGBoost models trained on IPL ball-by-ball data "
    "with explainable AI SHAP for transparency."
)
