import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Cricket Player Performance Prediction",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.markdown("## 🏏 CRICKET PLAYER PERFORMANCE PREDICTION")
st.write("Predict expected runs using historical IPL data and ML model")

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    matches = pd.read_csv("data/matches.csv")
    deliveries = pd.read_csv("data/deliveries.csv")

    # ✅ CORRECT MERGE (THIS FIXES match_id ERROR)
    merged_df = deliveries.merge(
        matches,
        left_on="match_id",
        right_on="id",
        how="left"
    )

    return merged_df

data = load_data()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.joblib")

model = load_model()

# -----------------------------
# SIDEBAR – INPUT PARAMETERS
# -----------------------------
st.sidebar.header("🎯 Input Parameters")

# MULTI PLAYER SELECTION
players = sorted(data["batter"].dropna().unique())
selected_players = st.sidebar.multiselect(
    "Select Player(s)",
    players,
    default=players[:1]
)

venues = sorted(data["venue"].dropna().unique())
selected_venue = st.sidebar.selectbox("Select Venue", venues)

opponents = sorted(
    pd.concat([data["team1"], data["team2"]]).dropna().unique()
)
selected_opponent = st.sidebar.selectbox("Select Opponent Team", opponents)

predict_btn = st.sidebar.button("🔮 Predict Performance")

# -----------------------------
# FEATURE ENGINEERING FUNCTION
# -----------------------------
def prepare_features(player_name):
    player_df = data[data["batter"] == player_name]

    if player_df.empty:
        return None

    recent_avg_runs = player_df.groupby("match_id")["total_runs"].sum().tail(5).mean()
    venue_avg_runs = player_df[player_df["venue"] == selected_venue]["total_runs"].mean()
    opponent_avg_runs = player_df[
        (player_df["team1"] == selected_opponent) |
        (player_df["team2"] == selected_opponent)
    ]["total_runs"].mean()
    career_avg_runs = player_df.groupby("match_id")["total_runs"].sum().mean()

    return np.array([
        recent_avg_runs if not np.isnan(recent_avg_runs) else 0,
        venue_avg_runs if not np.isnan(venue_avg_runs) else 0,
        opponent_avg_runs if not np.isnan(opponent_avg_runs) else 0,
        career_avg_runs if not np.isnan(career_avg_runs) else 0
    ]).reshape(1, -1)

# -----------------------------
# MAIN DASHBOARD
# -----------------------------
if predict_btn and selected_players:

    col1, col2, col3 = st.columns([1.5, 2, 2])

    # -----------------------------
    # PREDICTION CARDS
    # -----------------------------
    with col2:
        st.subheader("📊 Predicted Runs")

        results = []

        for player in selected_players:
            features = prepare_features(player)
            if features is None:
                continue

            prediction = model.predict(features)[0]
            results.append((player, int(prediction)))

        for r in results:
            st.success(f"**{r[0]}** → Predicted Runs: **{r[1]}**")

    # -----------------------------
    # PLAYER FORM GRAPH
    # -----------------------------
    with col3:
        st.subheader("📈 Player Form (Last Matches)")

        for player in selected_players:
            player_df = data[data["batter"] == player]
            form = player_df.groupby("match_id")["total_runs"].sum().tail(10)

            fig, ax = plt.subplots()
            ax.plot(form.values, marker="o")
            ax.set_title(player)
            ax.set_ylabel("Runs")
            ax.set_xlabel("Recent Matches")

            st.pyplot(fig)

    # -----------------------------
    # ANALYTICAL REPORT
    # -----------------------------
    st.markdown("---")
    st.subheader("📑 Analytical Output")

    report_data = []

    for r in results:
        report_data.append({
            "Player": r[0],
            "Opponent": selected_opponent,
            "Venue": selected_venue,
            "Predicted Runs": r[1],
            "Confidence": "High" if r[1] > 30 else "Medium"
        })

    report_df = pd.DataFrame(report_data)
    st.dataframe(report_df, use_container_width=True)

    # -----------------------------
    # FEATURE IMPORTANCE (MODEL-BASED)
    # -----------------------------
    st.subheader("🧠 Feature Importance")

    importance = model.feature_importances_
    features = [
        "Recent Avg Runs",
        "Venue Avg Runs",
        "Opponent Avg Runs",
        "Career Avg Runs"
    ]

    fig, ax = plt.subplots()
    ax.barh(features, importance)
    ax.set_title("Feature Importance for Prediction")

    st.pyplot(fig)

else:
    st.info("⬅️ Select players and click **Predict Performance**")
