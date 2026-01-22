# 🏏 Cricket Player Performance Prediction (IPL)

## Project Overview
This project is an end-to-end Machine Learning system designed to predict the batting performance (runs) of IPL players. It uses historical ball-by-ball data to engineer features like "Form," "Venue History," and "Opponent-specific averages" to provide real-time predictions via a Streamlit dashboard.

## Tech Stack
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Plotly
- **Tools:** Google Colab (Training), Git/GitHub (Version Control), Streamlit (Deployment)

## Data Science Lifecycle
1. **EDA:** Identified key trends in high-scoring venues and common dismissal types.
2. **Feature Engineering:** Aggregated ball-by-ball data into player-match summaries. Engineered 5-match rolling averages to capture current player "Form."
3. **Model Development:** Optimized an **XGBoost Regressor** using GridSearchCV. 
4. **Findings:** Feature importance analysis revealed that **Recent Form (761.0)** and **Career Average (751.0)** are the most critical predictors of a player's performance.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run the dashboard: `streamlit run app.py`

## Project Contributors
- **Jeevitha Divakar** (Data Science & Dashboard Development)
- Group-1 Team