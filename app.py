import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set page config
st.set_page_config(page_title="Cricket Performance AI", layout="wide")

# Load Data and Model
@st.cache_data # This keeps the app fast by loading data only once
def load_assets():
    df = pd.read_csv('dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    model = joblib.load('xgb_model_optimized.joblib')
    return df, model

df, model = load_assets()

# --- SIDEBAR INPUTS ---
st.sidebar.header("Input Parameters")

# Dynamically populate dropdowns from your CSV
all_players = sorted(df['batter'].unique())
selected_player = st.sidebar.selectbox("Select Player", all_players)

all_teams = sorted(df['bowling_team'].unique())
selected_opponent = st.sidebar.selectbox("Select Opponent", all_teams)

all_venues = sorted(df['venue'].unique())
selected_venue = st.sidebar.selectbox("Venue", all_venues)

# Get the latest stats for the selected player to set default slider values
player_data = df[df['batter'] == selected_player].sort_values('date', ascending=False)
latest_stats = player_data.iloc[0] if not player_data.empty else None

rolling_val = float(latest_stats['rolling_runs_avg']) if latest_stats is not None else 0.0
career_val = float(latest_stats['career_avg']) if latest_stats is not None else 0.0

rolling_avg = st.sidebar.slider("Recent Average Runs (Last 5)", 0.0, 100.0, rolling_val)
career_avg = st.sidebar.slider("Career Average", 0.0, 100.0, career_val)
opp_avg = st.sidebar.slider("Avg vs Opponent", 0.0, 100.0, 25.0)
venue_avg = st.sidebar.slider("Venue Average", 0.0, 100.0, 25.0)

if st.sidebar.button("PREDICT PERFORMANCE", use_container_width=True):
    # 1. Prediction Logic
    features = pd.DataFrame([[rolling_avg, venue_avg, career_avg, opp_avg]], 
                            columns=['rolling_runs_avg', 'venue_avg', 'career_avg', 'opp_avg'])
    prediction = model.predict(features)[0]
    
    # 2. Display Metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PREDICTED RUNS", f"{int(prediction)}")
    with col2:
        st.metric("CONFIDENCE", "High" if prediction > 20 else "Medium")

    # 3. Dynamic Charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader(f"{selected_player}: Last 5 Match Form")
        # Filter data for the actual selected player
        last_5 = player_data.head(5).sort_values('date')
        fig_form = px.line(last_5, x='date', y='batsman_runs', markers=True, 
                           labels={'batsman_runs': 'Runs', 'date': 'Match Date'})
        st.plotly_chart(fig_form, use_container_width=True)

    with col_right:
        st.subheader("Model Feature Importance")
        # Using your actual importance scores from the XGBoost plot
        importance_df = pd.DataFrame({
            'Feature': ['Recent_Avg', 'Career_Avg', 'Opponent_Avg', 'Venue_Avg'],
            'Importance': [761, 751, 601, 573]
        })
        fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h', color='Importance')
        st.plotly_chart(fig_imp, use_container_width=True)