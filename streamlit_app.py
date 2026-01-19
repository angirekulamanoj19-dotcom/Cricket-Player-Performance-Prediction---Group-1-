import streamlit as st
import pandas as pd
import joblib
import random
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}
h1, h2, h3, h4, h5, h6, p, label {
    color: white !important;
}
.stButton>button {
    background-color: #ff9800;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: #ffc107;
    color: black;
}
.stSlider > div {
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="Cricket Run Predictor", layout="wide")

model = joblib.load("xgb_model.joblib")
preprocessor = joblib.load("feature_pipeline.pkl")

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("🏏 Cricket Player Run Predictor")
st.caption("Interactive AI app to predict cricket player runs")

# Sidebar controls
st.sidebar.header("⚙ Controls")
theme = st.sidebar.radio("Choose View Mode", ["Simple", "Advanced"])
auto_demo = st.sidebar.checkbox("Auto Demo Mode")

# Player Info
st.markdown("## 👤 Player Info")
player_name = st.text_input("Player Name", "Virat Kohli")

# Stats Section
st.markdown("## 📊 Performance Stats")
col1, col2, col3 = st.columns(3)

with col1:
    runs_last_5 = st.slider("Runs (Last 5)", 0, 100, 35)
    venue_avg_runs = st.slider("Venue Avg", 0, 100, 30)

with col2:
    runs_last_10 = st.slider("Runs (Last 10)", 0, 100, 40)
    opp_avg_runs = st.slider("Opponent Avg", 0, 100, 28)

with col3:
    career_avg_runs = st.slider("Career Avg", 0, 100, 38)
    career_sr = st.slider("Strike Rate", 50, 200, 130)

# Match Info
st.markdown("## 🏟 Match Info")
venue = st.selectbox("Venue", ["Mumbai", "Chennai", "Bangalore", "Delhi", "Kolkata", "Hyderabad"])
opposition = st.selectbox("Opposition", ["CSK", "MI", "RCB", "KKR", "SRH", "DC"])
team = st.selectbox("Player Team", ["MI", "CSK", "RCB", "KKR", "SRH", "DC"])

# Auto Demo
if auto_demo:
    runs_last_5 = random.randint(10, 60)
    runs_last_10 = random.randint(15, 70)
    venue_avg_runs = random.randint(10, 50)
    opp_avg_runs = random.randint(10, 50)
    career_avg_runs = random.randint(20, 60)
    career_sr = random.randint(90, 160)

input_data = pd.DataFrame([{
    "runs_last_5": runs_last_5,
    "runs_last_10": runs_last_10,
    "venue_avg_runs": venue_avg_runs,
    "opp_avg_runs": opp_avg_runs,
    "career_avg_runs": career_avg_runs,
    "career_sr": career_sr,
    "venue": venue,
    "opposition": opposition,
    "team": team
}])

st.markdown("---")
colA, colB, colC = st.columns(3)

with colA:
    predict = st.button("🔮 Predict")
with colB:
    clear = st.button("🧹 Clear History")
with colC:
    reroll = st.button("🎲 Randomize")

if reroll:
    st.experimental_rerun()

if clear:
    st.session_state.history = []

if predict:
    processed = preprocessor.transform(input_data)
    prediction = int(model.predict(processed)[0])

    st.success(f"🏆 {player_name} may score around **{prediction} runs**")

    st.session_state.history.append({
        "Player": player_name,
        "Runs": prediction,
        "Venue": venue,
        "Opponent": opposition
    })

# History Section
if st.session_state.history:
    st.markdown("## 📜 Prediction History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    st.markdown("## 📈 Prediction Graph")
    st.line_chart(hist_df["Runs"])
