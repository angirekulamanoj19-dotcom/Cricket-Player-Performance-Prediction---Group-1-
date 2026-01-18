
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Cricket Performance Prediction", layout="wide")

# ✅ Update project root
PROJECT_ROOT = Path(r"D:\Cricket-Player-Performance-Prediction")

DATA_DIR = PROJECT_ROOT / "data" / "processed" / "milestone2"
MODEL_DIR = PROJECT_ROOT / "models"

bat_data_path = DATA_DIR / "batting_player_match_dataset.csv"
bowl_data_path = DATA_DIR / "bowling_player_match_dataset.csv"

bat_model_path = MODEL_DIR / "runs_next_match_model.pkl"
bowl_model_path = MODEL_DIR / "wickets_next_match_model.pkl"
config_path = MODEL_DIR / "model_config.json"

st.title("🏏 Cricket Player Performance Prediction")
st.caption("Premium Dashboard: Batter + Bowler predictions with graphs & explanations")

# Load assets
@st.cache_data
def load_data():
    bat_df = pd.read_csv(bat_data_path)
    bowl_df = pd.read_csv(bowl_data_path)
    return bat_df, bowl_df

@st.cache_resource
def load_models():
    bat_model = joblib.load(bat_model_path) if bat_model_path.exists() else None
    bowl_model = joblib.load(bowl_model_path) if bowl_model_path.exists() else None
    config = json.load(open(config_path)) if config_path.exists() else {}
    return bat_model, bowl_model, config

bat_df, bowl_df = load_data()
bat_model, bowl_model, config = load_models()

# Sidebar inputs
st.sidebar.header("Input Parameters")
player_type = st.sidebar.radio("Select Role", ["Batter", "Bowler"])

if player_type == "Batter":
    players = sorted(bat_df["batter"].unique())
    player = st.sidebar.selectbox("Select Batter", players)
    venue = st.sidebar.selectbox("Venue (optional)", ["All"] + sorted(bat_df["venue"].dropna().unique()))
else:
    players = sorted(bowl_df["bowler"].unique())
    player = st.sidebar.selectbox("Select Bowler", players)
    venue = st.sidebar.selectbox("Venue (optional)", ["All"] + sorted(bowl_df["venue"].dropna().unique()))

predict_btn = st.sidebar.button("Predict Performance")

# Filter player data
if player_type == "Batter":
    p_df = bat_df[bat_df["batter"] == player].sort_values("match_order")
    if venue != "All":
        p_df = p_df[p_df["venue"] == venue]
else:
    p_df = bowl_df[bowl_df["bowler"] == player].sort_values("match_order")
    if venue != "All":
        p_df = p_df[p_df["venue"] == venue]

# Dashboard layout
top1, top2, top3 = st.columns([1.2,1.2,1.6])

if player_type == "Batter":
    pred_value = None
    if predict_btn and bat_model is not None:
        features = config.get("bat_features", [])
        latest = p_df.iloc[-1].copy()
        X = latest[features].fillna(0).values.reshape(1,-1)
        pred_value = float(bat_model.predict(X)[0])
    top1.metric("Predicted Runs", f"{pred_value:.1f}" if pred_value is not None else "--")
    top2.metric("Prediction Confidence", "High" if pred_value is not None else "N/A")
    top3.metric("Predicted Wickets", "--")
else:
    pred_value = None
    if predict_btn and bowl_model is not None:
        features = config.get("bowl_features", [])
        latest = p_df.iloc[-1].copy()
        X = latest[features].fillna(0).values.reshape(1,-1)
        pred_value = float(bowl_model.predict(X)[0])
    top1.metric("Predicted Runs", "--")
    top2.metric("Prediction Confidence", "High" if pred_value is not None else "N/A")
    top3.metric("Predicted Wickets", f"{pred_value:.2f}" if pred_value is not None else "--")

st.divider()

mid1, mid2 = st.columns([1.4,1.6])

# Form chart
with mid1:
    st.subheader("Player Form: Last 10 Matches")
    if len(p_df) > 0:
        if player_type == "Batter":
            fig = px.line(p_df.tail(10), x="match_order", y="runs", markers=True, title=f"{player} - Runs Trend")
        else:
            fig = px.line(p_df.tail(10), x="match_order", y="wickets", markers=True, title=f"{player} - Wickets Trend")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for selected filters.")

# Feature importance
with mid2:
    st.subheader("Global Feature Importance (Model)")
    if player_type == "Batter" and bat_model is not None and hasattr(bat_model, "feature_importances_"):
        imp = pd.DataFrame({"feature": config.get("bat_features", []), "importance": bat_model.feature_importances_}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(imp.head(12), x="importance", y="feature", orientation="h", title="Top Batter Features"), use_container_width=True)
    elif player_type == "Bowler" and bowl_model is not None and hasattr(bowl_model, "feature_importances_"):
        imp = pd.DataFrame({"feature": config.get("bowl_features", []), "importance": bowl_model.feature_importances_}).sort_values("importance", ascending=False)
        st.plotly_chart(px.bar(imp.head(12), x="importance", y="feature", orientation="h", title="Top Bowler Features"), use_container_width=True)
    else:
        st.warning("Model/feature importance not available. Run Milestone-3++ to generate models.")

st.divider()

st.subheader("Outcome Testing: Top Predicted Players")
tab1, tab2 = st.tabs(["Top Batters", "Top Bowlers"])

with tab1:
    if bat_model is not None:
        features = config.get("bat_features", [])
        tmp = bat_df.copy()
        tmp["pred_runs_next"] = bat_model.predict(tmp[features].fillna(0))
        st.dataframe(tmp.sort_values("pred_runs_next", ascending=False).head(15)[["batter","venue","pred_runs_next","runs_last10_avg","strike_rate"]])
    else:
        st.info("Batter model not found.")

with tab2:
    if bowl_model is not None:
        features = config.get("bowl_features", [])
        tmp = bowl_df.copy()
        tmp["pred_wkts_next"] = bowl_model.predict(tmp[features].fillna(0))
        st.dataframe(tmp.sort_values("pred_wkts_next", ascending=False).head(15)[["bowler","venue","pred_wkts_next","wkts_last10_avg","economy"]])
    else:
        st.info("Bowler model not found.")
