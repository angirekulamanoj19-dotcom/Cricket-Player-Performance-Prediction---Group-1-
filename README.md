# 🏏 Cricket Player Performance Prediction

## 📌 Project Overview
This project predicts **expected runs for batsmen** and **expected wickets for bowlers** using
machine learning models trained on **IPL historical data (2008–2024)**.

The system provides:
- Data-driven performance predictions
- Confidence ranges for predictions
- Explainable AI using SHAP
- An interactive Streamlit dashboard

---

## 🧠 Problem Statement
Traditional cricket analysis relies heavily on intuition and basic averages.
This project introduces a **machine learning–based approach** to analyze player form,
venue conditions, and opponent impact to generate realistic predictions.

---

## 📊 Dataset
- Source: IPL Ball-by-Ball & Match datasets (Kaggle)
- Time Period: 2008–2024
- Raw data transformed into **player-match level**
- Separate datasets for batsmen and bowlers

---

## ⚙️ Methodology
1. Data Cleaning & Preprocessing  
2. Feature Engineering (form, venue, opponent, career stats)  
3. Baseline Modeling (rolling averages)  
4. ML Models: Random Forest, XGBoost, LightGBM  
5. Hyperparameter Tuning (Optuna)  
6. Model Evaluation (MAE, RMSE, R²)  
7. Deployment using Streamlit  

---

## 🧪 Models Used
- Baseline Rolling Average
- Random Forest Regressor
- XGBoost Regressor (Best performer)
- LightGBM Regressor

---

## 📈 Evaluation Metrics
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🖥️ User Interface
Built using **Streamlit**, the dashboard allows users to:
- Select player, venue, and opponent
- Predict runs or wickets
- View confidence range
- Analyze feature contribution using SHAP

---

## 🚀 How to Run the Project

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run streamlit_app.py
```

---

## 📁 Project Structure
```
project/
│
├── batsman_model_dataset.csv
├── bowler_model_dataset.csv
├── streamlit_app.py
├── requirements.txt
├── README.md
│
└── models/
    ├── xgb_batsman_model.joblib
    ├── xgb_bowler_model.joblib
    ├── feature_pipeline.pkl
```

---

## 🔮 Future Scope
- Live match predictions
- Player-vs-player analysis
- API integration
- Advanced deep learning models

---

## ✅ Conclusion
This project demonstrates an **end-to-end machine learning pipeline**
with explainable AI and real-world deployment capability for sports analytics.
