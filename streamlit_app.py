import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Cricket Performance Analytics", layout="wide")

# ------------------------------------------------
# Load Data
# ------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

st.title("🏏 Cricket Performance Analytics & Run Prediction")

# ------------------------------------------------
# Sidebar Filters
# ------------------------------------------------
st.sidebar.header("Filters")

player = st.sidebar.selectbox("Select Batter", df["batter"].unique())
season = st.sidebar.selectbox("Select Season", sorted(df["season"].unique()))

filtered = df[(df["batter"] == player) & (df["season"] == season)]

# ------------------------------------------------
# Overview
# ------------------------------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(filtered)

col1, col2, col3 = st.columns(3)

col1.metric("Matches", len(filtered))
col2.metric("Avg Runs", round(filtered["runs"].mean(), 2))
col3.metric("Avg Strike Rate", round(filtered["strike_rate"].mean(), 2))

# ------------------------------------------------
# Visualizations
# ------------------------------------------------
st.subheader("📈 Performance Visuals")

fig1 = px.bar(filtered, x="date", y="runs", title="Runs per Match")
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(filtered, x="date", y="strike_rate", title="Strike Rate Trend")
st.plotly_chart(fig2, use_container_width=True)

fig3 = px.scatter(df, x="career_avg_runs", y="runs",
                  title="Career Avg vs Match Runs")
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------------------------
# ML Run Prediction
# ------------------------------------------------
st.subheader("🤖 Predict Runs (ML)")

features = [
    "balls",
    "avg_runs_last_5",
    "avg_runs_last_10",
    "venue_avg_runs",
    "career_avg_runs"
]

X = df[features]
y = df["target_runs"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)

st.markdown("### Enter Match Inputs")

c1, c2, c3 = st.columns(3)

balls = c1.number_input("Balls Faced", 1, 200, 20)
avg5 = c2.number_input("Avg Runs Last 5", 0.0, 100.0, 20.0)
avg10 = c3.number_input("Avg Runs Last 10", 0.0, 100.0, 25.0)

c4, c5 = st.columns(2)

venue_avg = c4.number_input("Venue Avg Runs", 0.0, 100.0, 30.0)
career_avg = c5.number_input("Career Avg Runs", 0.0, 100.0, 35.0)

if st.button("Predict Runs"):
    pred = model.predict([[balls, avg5, avg10, venue_avg, career_avg]])
    st.success(f"🏏 Predicted Runs: {int(pred[0])}")

# ------------------------------------------------
st.markdown("✅ Built with Streamlit + RandomForest ML")
