"""
Utility functions for Cricket Score Prediction System
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import joblib

def load_pipeline():
    """Load the feature engineering pipeline"""
    return joblib.load('models/feature_pipeline.pkl')

def load_model(model_name: str = 'xgb'):
    """
    Load a trained model
    
    Args:
        model_name: 'xgb', 'lgb', or 'rf'
    
    Returns:
        Trained model object
    """
    model_paths = {
        'xgb': 'models/xgb_model.joblib',
        'lgb': 'models/lgb_model.joblib',
        'rf': 'models/rf_model.joblib'
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Model {model_name} not found. Choose from {list(model_paths.keys())}")
    
    return joblib.load(model_paths[model_name])

def create_prediction_features(
    runs_last_3: float,
    runs_last_5: float,
    runs_last_10: float,
    sr_last_3: float,
    sr_last_5: float,
    sr_last_10: float,
    total_matches: int,
    total_runs: int,
    total_innings: int,
    batting_position: int,
    venue: str = None,
    country: str = None
) -> pd.DataFrame:
    """
    Create feature vector for prediction
    
    Args:
        Various player and match statistics
    
    Returns:
        DataFrame with features ready for prediction
    """
    pipeline = load_pipeline()
    
    features = {
        'runs_last_3': runs_last_3,
        'runs_last_5': runs_last_5,
        'runs_last_10': runs_last_10,
        'sr_last_3': sr_last_3,
        'sr_last_5': sr_last_5,
        'sr_last_10': sr_last_10,
        'total_matches': total_matches,
        'total_runs': total_runs,
        'total_innings': total_innings,
        'batting_position': batting_position,
        'avg_score': total_runs / total_innings if total_innings > 0 else 0,
        'overall_sr': (total_runs / total_innings * 100) if total_innings > 0 else 0,
    }
    
    # Add venue and country if provided
    if venue and country:
        venue_encoder = pipeline['venue_encoder']
        country_encoder = pipeline['country_encoder']
        
        try:
            features['venue_encoded'] = venue_encoder.transform([venue])[0]
        except:
            features['venue_encoded'] = 0  # Unknown venue
        
        try:
            features['country_encoded'] = country_encoder.transform([country])[0]
        except:
            features['country_encoded'] = 0  # Unknown country
    
    # Fill remaining features with defaults
    batting_features = pipeline['batting_features']
    for feat in batting_features:
        if feat not in features:
            features[feat] = 0.0
    
    return pd.DataFrame([features])[batting_features]

def predict_score(
    features: pd.DataFrame,
    model_name: str = 'xgb',
    return_all_models: bool = False
) -> Dict:
    """
    Make prediction using specified model
    
    Args:
        features: Feature DataFrame
        model_name: Model to use ('xgb', 'lgb', 'rf')
        return_all_models: If True, return predictions from all models
    
    Returns:
        Dictionary with prediction(s)
    """
    if return_all_models:
        xgb_model = load_model('xgb')
        lgb_model = load_model('lgb')
        rf_model = load_model('rf')
        
        return {
            'xgb': max(0, xgb_model.predict(features)[0]),
            'lgb': max(0, lgb_model.predict(features)[0]),
            'rf': max(0, rf_model.predict(features)[0])
        }
    else:
        model = load_model(model_name)
        prediction = max(0, model.predict(features)[0])
        return {model_name: prediction}

def calculate_confidence_interval(
    prediction: float,
    confidence: float = 0.95,
    model_rmse: float = 17.3
) -> Tuple[float, float]:
    """
    Calculate confidence interval for prediction
    
    Args:
        prediction: Predicted score
        confidence: Confidence level (default 95%)
        model_rmse: Model's RMSE (default from XGBoost)
    
    Returns:
        (lower_bound, upper_bound)
    """
    from scipy import stats
    
    # Calculate margin of error
    z_score = stats.norm.ppf((1 + confidence) / 2)
    margin = z_score * model_rmse
    
    lower = max(0, prediction - margin)
    upper = prediction + margin
    
    return (lower, upper)

def categorize_score(score: float) -> Dict[str, str]:
    """
    Categorize predicted score into performance bands
    
    Args:
        score: Predicted score
    
    Returns:
        Dictionary with category and description
    """
    if score < 15:
        return {
            'category': 'Low Score',
            'emoji': '🔴',
            'description': 'Below par performance'
        }
    elif score < 30:
        return {
            'category': 'Below Average',
            'emoji': '🟠',
            'description': 'Subpar innings'
        }
    elif score < 50:
        return {
            'category': 'Average',
            'emoji': '🟡',
            'description': 'Typical T20 score'
        }
    elif score < 75:
        return {
            'category': 'Good',
            'emoji': '🟢',
            'description': 'Strong performance'
        }
    else:
        return {
            'category': 'Excellent',
            'emoji': '🟣',
            'description': 'Outstanding innings'
        }

def format_metric(value: float, metric_type: str) -> str:
    """
    Format metrics for display
    
    Args:
        value: Metric value
        metric_type: Type of metric ('rmse', 'mae', 'r2', 'score')
    
    Returns:
        Formatted string
    """
    if metric_type in ['rmse', 'mae', 'score']:
        return f"{value:.2f}"
    elif metric_type == 'r2':
        return f"{value:.4f}"
    else:
        return str(value)

def get_model_info(model_name: str) -> Dict:
    """
    Get information about a specific model
    
    Args:
        model_name: 'xgb', 'lgb', or 'rf'
    
    Returns:
        Dictionary with model information
    """
    model_info = {
        'xgb': {
            'full_name': 'XGBoost (Tuned)',
            'description': 'Gradient boosting optimized with Optuna',
            'rmse': 17.3,
            'mae': 12.9,
            'r2': 0.33,
            'training_time': '~15 min'
        },
        'lgb': {
            'full_name': 'LightGBM (Tuned)',
            'description': 'Light gradient boosting with GridSearchCV',
            'rmse': 17.5,
            'mae': 13.0,
            'r2': 0.32,
            'training_time': '~8 min'
        },
        'rf': {
            'full_name': 'Random Forest',
            'description': 'Ensemble of decision trees',
            'rmse': 18.2,
            'mae': 13.6,
            'r2': 0.29,
            'training_time': '~2 min'
        }
    }
    
    return model_info.get(model_name, {})

def validate_input(value: float, min_val: float, max_val: float, name: str) -> bool:
    """
    Validate input value is within expected range
    
    Args:
        value: Value to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        name: Name of the parameter
    
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not min_val <= value <= max_val:
        raise ValueError(f"{name} must be between {min_val} and {max_val}")
    return True

# Example usage
if __name__ == "__main__":
    # Test prediction
    features = create_prediction_features(
        runs_last_3=45.0,
        runs_last_5=38.0,
        runs_last_10=35.0,
        sr_last_3=145.0,
        sr_last_5=138.0,
        sr_last_10=132.0,
        total_matches=75,
        total_runs=2100,
        total_innings=72,
        batting_position=3
    )
    
    predictions = predict_score(features, return_all_models=True)
    print("Predictions:")
    for model, score in predictions.items():
        print(f"  {model.upper()}: {score:.0f} runs")
        category = categorize_score(score)
        print(f"    Category: {category['emoji']} {category['category']}")
