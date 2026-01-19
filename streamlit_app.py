import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# Load model and preprocessor
@st.cache_resource
def load_model_and_pipeline():
    model = joblib.load('xgb_model.joblib')
    pipeline = joblib.load('feature_pipeline.pkl')
    scaler = pipeline['scaler']
    encoder = pipeline['encoder']
    return model, scaler, encoder

model, scaler, encoder = load_model_and_pipeline()

# Load sample data for SHAP (optional, for visualization)
data = pd.read_csv('dataset.csv')
features = ['team1_rolling_wins', 'team1_rolling_score', 'team2_rolling_wins', 'team2_rolling_score', 'venue_win_avg', 'venue_score_avg', 'tvt_win_avg', 'team1_career_wins', 'team1_career_score', 'team2_career_wins', 'team2_career_score']
sample_data = data[features].dropna().head(100)  # Sample for SHAP

# App title
st.title("IPL Team Performance Predictor")
st.markdown("Predict next-match team scores and wins using machine learning. Enter team stats below.")

# User inputs
st.header("Input Team Features")
col1, col2 = st.columns(2)

with col1:
    team1_rolling_wins = st.slider("Team 1 Rolling Wins (last 5 matches)", 0.0, 5.0, 2.5)
    team1_rolling_score = st.slider("Team 1 Rolling Score (last 5 matches)", 100.0, 250.0, 150.0)
    team2_rolling_wins = st.slider("Team 2 Rolling Wins (last 5 matches)", 0.0, 5.0, 2.5)
    team2_rolling_score = st.slider("Team 2 Rolling Score (last 5 matches)", 100.0, 250.0, 150.0)
    venue_win_avg = st.slider("Venue Win Average", 0.0, 1.0, 0.5)

with col2:
    venue_score_avg = st.slider("Venue Score Average", 100.0, 200.0, 150.0)
    tvt_win_avg = st.slider("Team vs Team Win Average", 0.0, 1.0, 0.5)
    team1_career_wins = st.slider("Team 1 Career Wins", 0.0, 100.0, 50.0)
    team1_career_score = st.slider("Team 1 Career Score", 100.0, 200.0, 150.0)
    team2_career_wins = st.slider("Team 2 Career Wins", 0.0, 100.0, 50.0)
    team2_career_score = st.slider("Team 2 Career Score", 100.0, 200.0, 150.0)

# Prediction logic
if st.button("Predict"):
    # Prepare input data
    input_data = pd.DataFrame({
        'team1_rolling_wins': [team1_rolling_wins],
        'team1_rolling_score': [team1_rolling_score],
        'team2_rolling_wins': [team2_rolling_wins],
        'team2_rolling_score': [team2_rolling_score],
        'venue_win_avg': [venue_win_avg],
        'venue_score_avg': [venue_score_avg],
        'tvt_win_avg': [tvt_win_avg],
        'team1_career_wins': [team1_career_wins],
        'team1_career_score': [team1_career_score],
        'team2_career_wins': [team2_career_wins],
        'team2_career_score': [team2_career_score]
    })
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predictions
    predicted_score = model.predict(input_scaled)[0]
    predicted_win_prob = model.predict_proba(input_scaled)[0][1]  # Probability of win (class 1)
    predicted_win = "Win" if predicted_win_prob > 0.5 else "Loss"
    
    # Display results
    st.header("Prediction Results")
    st.write(f"**Predicted Next Match Score for Team 1:** {predicted_score:.2f}")
    st.write(f"**Predicted Win Probability for Team 1:** {predicted_win_prob:.2f}")
    st.write(f"**Predicted Outcome:** {predicted_win}")
    
    # Visualization: Prediction chart
    fig, ax = plt.subplots()
    ax.bar(['Predicted Score', 'Win Probability'], [predicted_score, predicted_win_prob], color=['blue', 'green'])
    ax.set_title("Prediction Summary")
    st.pyplot(fig)

# Visualizations section
st.header("Model Insights")
if st.checkbox("Show SHAP Feature Importance"):
    # SHAP explainer
    explainer = shap.Explainer(model)
    shap_values = explainer(sample_data)
    st.write("SHAP Summary Plot (Feature Importance)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample_data, show=False)
    st.pyplot(fig)

if st.checkbox("Show Model Performance on Sample Data"):
    # Sample predictions
    sample_scaled = scaler.transform(sample_data)
    sample_predictions = model.predict(sample_scaled)
    true_scores = data['label_team1_score_next'].dropna().head(100)
    rmse = np.sqrt(mean_squared_error(true_scores, sample_predictions))
    r2 = r2_score(true_scores, sample_predictions)
    st.write(f"Sample RMSE: {rmse:.2f}")
    st.write(f"Sample R²: {r2:.2f}")
    
    # Scatter plot
    fig, ax = plt.subplots()
    ax.scatter(true_scores, sample_predictions, alpha=0.5)
    ax.plot([true_scores.min(), true_scores.max()], [true_scores.min(), true_scores.max()], 'r--')
    ax.set_xlabel("True Scores")
    ax.set_ylabel("Predicted Scores")
    ax.set_title("Predicted vs True Scores")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit. Model trained on IPL data for team performance prediction.")