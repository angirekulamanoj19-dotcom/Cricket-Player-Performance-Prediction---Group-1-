"""
Configuration settings for Cricket Score Prediction System
"""

import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'dataset'
MODELS_DIR = BASE_DIR / 'models'
NOTEBOOKS_DIR = BASE_DIR / 'notebooks'

# Model files
XGB_MODEL_PATH = MODELS_DIR / 'xgb_model.joblib'
LGB_MODEL_PATH = MODELS_DIR / 'lgb_model.joblib'
RF_MODEL_PATH = MODELS_DIR / 'rf_model.joblib'
PIPELINE_PATH = MODELS_DIR / 'feature_pipeline.pkl'
RESULTS_PATH = MODELS_DIR / 'model_results.csv'
SHAP_PATH = MODELS_DIR / 'shap_values.npy'

# Data files
BATTING_DATA = DATA_DIR / 'batting_features_dataset.csv'
BOWLING_DATA = DATA_DIR / 'bowling_features_dataset.csv'
MATCHES_DATA = DATA_DIR / 't20i_Matches_Data.csv'
PLAYERS_DATA = DATA_DIR / 'players_info.csv'

# Model configurations
MODEL_CONFIGS = {
    'xgb': {
        'name': 'XGBoost (Tuned)',
        'path': XGB_MODEL_PATH,
        'color': '#1f77b4',
        'icon': '🏆'
    },
    'lgb': {
        'name': 'LightGBM (Tuned)',
        'path': LGB_MODEL_PATH,
        'color': '#2ca02c',
        'icon': '💚'
    },
    'rf': {
        'name': 'Random Forest',
        'path': RF_MODEL_PATH,
        'color': '#ff7f0e',
        'icon': '🌲'
    }
}

# Performance thresholds
SCORE_CATEGORIES = {
    'excellent': 75,
    'good': 50,
    'average': 30,
    'below_average': 15,
    'low': 0
}

# Dashboard settings
DASHBOARD_CONFIG = {
    'title': 'Cricket Score Prediction System',
    'icon': '🏏',
    'layout': 'wide',
    'theme': {
        'primaryColor': '#1f77b4',
        'backgroundColor': '#ffffff',
        'secondaryBackgroundColor': '#f0f2f6',
        'textColor': '#262730'
    }
}

# API settings (for future use)
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'version': 'v1',
    'title': 'Cricket Score Prediction API',
    'description': 'ML-powered cricket player score prediction'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'app.log'
}

# Feature ranges (for validation)
FEATURE_RANGES = {
    'runs_last_3': (0, 200),
    'runs_last_5': (0, 200),
    'runs_last_10': (0, 200),
    'sr_last_3': (0, 300),
    'sr_last_5': (0, 300),
    'sr_last_10': (0, 300),
    'total_matches': (1, 500),
    'total_runs': (0, 10000),
    'total_innings': (1, 500),
    'batting_position': (1, 11)
}

# Model performance metrics
MODEL_PERFORMANCE = {
    'baseline': {'rmse': 19.5, 'mae': 14.8, 'r2': 0.24},
    'rf': {'rmse': 18.2, 'mae': 13.6, 'r2': 0.29},
    'xgb': {'rmse': 17.3, 'mae': 12.9, 'r2': 0.33},
    'lgb': {'rmse': 17.5, 'mae': 13.0, 'r2': 0.32}
}

# Environment variables
STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', 8501))
DEBUG_MODE = os.getenv('DEBUG', 'False').lower() == 'true'
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# Cache settings
CACHE_TTL = 3600  # 1 hour in seconds

# Export all settings
__all__ = [
    'BASE_DIR',
    'DATA_DIR',
    'MODELS_DIR',
    'MODEL_CONFIGS',
    'SCORE_CATEGORIES',
    'DASHBOARD_CONFIG',
    'API_CONFIG',
    'FEATURE_RANGES',
    'MODEL_PERFORMANCE'
]
