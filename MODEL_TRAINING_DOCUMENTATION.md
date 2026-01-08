# Model Training Documentation

## Overview

This document explains the model training implementation for the Cricket Score Prediction System. A comprehensive model training pipeline has been created in `03_ModelTraining.ipynb`.

---

## What Was Done

### 1. **Baseline Model Establishment**

- **Model Type:** 10-match rolling average
- **Purpose:** Establish a performance benchmark for comparison
- **Metrics Tracked:** RMSE, MAE, R²
- **Key Insight:** Provides a simple reference point to measure ML model improvements

---

### 2. **Machine Learning Models Implemented**

#### **Random Forest Regressor**

- Ensemble learning method using multiple decision trees
- Parameters: 100 estimators, max depth 15, minimum samples split 10
- Provides robust predictions with natural feature importance

#### **XGBoost Regressor**

- Gradient boosting framework optimized for speed and performance
- Initial parameters: 100 estimators, max depth 6, learning rate 0.1
- Known for handling complex patterns in data

#### **LightGBM Regressor**

- Gradient boosting framework optimized for efficiency
- Initial parameters: 100 estimators, max depth 6, learning rate 0.1
- Fast training with excellent performance on large datasets

---

### 3. **Hyperparameter Tuning**

#### **XGBoost - Optuna Optimization**

- **Tool:** Optuna (advanced hyperparameter optimization)
- **Trials:** 50 different parameter combinations
- **Parameters Tuned:**
  - Number of estimators (100-500)
  - Maximum depth (3-10)
  - Learning rate (0.01-0.3)
  - Subsample ratio (0.6-1.0)
  - Column sample ratio (0.6-1.0)
  - Minimum child weight (1-7)
  - Gamma (0-0.5)

#### **LightGBM - GridSearchCV**

- **Tool:** GridSearchCV (exhaustive parameter search)
- **Parameters Tuned:**
  - Number of estimators: [100, 200, 300]
  - Maximum depth: [5, 7, 9]
  - Learning rate: [0.05, 0.1, 0.15]
  - Subsample: [0.7, 0.8, 0.9]
  - Column sample: [0.7, 0.8, 0.9]
- **Total Combinations:** 243 parameter sets tested

---

### 4. **Evaluation Metrics**

All models are evaluated using three key metrics:

#### **RMSE (Root Mean Squared Error)**

- Measures average prediction error
- Penalizes larger errors more heavily
- Lower values indicate better performance

#### **MAE (Mean Absolute Error)**

- Average absolute difference between predictions and actuals
- More interpretable than RMSE
- Less sensitive to outliers

#### **R² Score (Coefficient of Determination)**

- Measures proportion of variance explained by the model
- Range: 0 to 1 (higher is better)
- Indicates how well the model fits the data

---

### 5. **Feature Importance Analysis with SHAP**

#### **What is SHAP?**

SHAP (SHapley Additive exPlanations) explains individual predictions by computing feature contributions.

#### **Visualizations Created:**

1. **Summary Plot:** Shows feature impact distribution across all predictions
2. **Bar Plot:** Ranks features by average importance
3. **Dependence Plots:** Shows relationship between feature values and predictions for top 4 features

#### **Benefits:**

- Understand which features drive predictions
- Identify most influential statistics (e.g., recent form, career averages)
- Validate model behavior matches cricket domain knowledge

---

### 6. **Model Comparison**

All models are compared side-by-side on:

- Training set performance
- Test set performance
- Improvement over baseline
- Visual comparisons through bar charts

**Models Compared:**

1. Baseline (10-Match Rolling Average)
2. Random Forest
3. XGBoost (default parameters)
4. LightGBM (default parameters)
5. XGBoost (Optuna tuned)
6. LightGBM (GridSearchCV tuned)

---

### 7. **Prediction Analysis**

#### **Actual vs Predicted Plots**

- Scatter plots comparing true runs vs predicted runs
- Perfect predictions fall on 45-degree line
- Shows model accuracy visually

#### **Residual Analysis**

- **Residual Plots:** Shows prediction errors vs predicted values
- **Residual Histograms:** Distribution of prediction errors
- Helps identify systematic biases or patterns in errors

---

### 8. **Saved Model Artifacts**

All trained models and results are saved for future use:

| File                   | Description                                 |
| ---------------------- | ------------------------------------------- |
| `xgb_model.joblib`     | Tuned XGBoost model (primary model)         |
| `lgb_model.joblib`     | Tuned LightGBM model                        |
| `rf_model.joblib`      | Random Forest model                         |
| `model_results.csv`    | Performance metrics for all models          |
| `shap_values.npy`      | SHAP values for feature importance          |
| `feature_pipeline.pkl` | Preprocessing pipeline (from previous step) |

---

## Workflow Summary

```
1. Load feature-engineered datasets
   ↓
2. Prepare train-test split (80-20, time-based)
   ↓
3. Encode categorical features (venue, country)
   ↓
4. Scale numerical features
   ↓
5. Train baseline model
   ↓
6. Train ML models (RF, XGBoost, LightGBM)
   ↓
7. Hyperparameter tuning (Optuna + GridSearchCV)
   ↓
8. Evaluate all models (RMSE, MAE, R²)
   ↓
9. SHAP analysis for interpretability
   ↓
10. Save best models and artifacts
```

---

## Key Features Used for Prediction

The models use 14 engineered features:

**Career Statistics:**

- Career matches played
- Career batting average
- Career strike rate

**Recent Form (Rolling Averages):**

- Last 3, 5, 10 matches average runs
- Last 3, 5, 10 matches average strike rate

**Context-Specific Stats:**

- Previous venue batting average
- Previous venue strike rate
- Previous opponent batting average

**Encoded Features:**

- Match venue (city) - encoded
- Match country - encoded

---

## Expected Outcomes

✅ **Multiple trained models** ready for deployment  
✅ **Performance benchmarks** established  
✅ **Best model identified** through comprehensive comparison  
✅ **Feature importance** understood through SHAP  
✅ **Model artifacts saved** for production use  
✅ **Detailed metrics** for model validation

---

## Next Steps

1. **Model Deployment:** Load saved models for real-time predictions
2. **API Integration:** Create prediction endpoint using best model
3. **Monitoring:** Track model performance on new data
4. **Retraining:** Update models as new match data becomes available
5. **A/B Testing:** Compare multiple models in production

---

## Technical Stack

- **Python Libraries:** scikit-learn, XGBoost, LightGBM, Optuna, SHAP
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Model Persistence:** joblib

---

## Notes

- Time-series split ensures no data leakage (past predicts future)
- All features are available before match starts (no future data)
- Models predict next match runs based on historical performance
- SHAP analysis confirms model decisions align with cricket knowledge

---

_Documentation created: January 8, 2026_
