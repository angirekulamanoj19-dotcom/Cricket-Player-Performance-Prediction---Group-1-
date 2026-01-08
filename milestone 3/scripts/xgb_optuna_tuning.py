from pathlib import Path
import pandas as pd
import optuna
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

# Load data once (important for speed)
df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
df['date_parsed'] = pd.to_datetime(df['date_parsed'])
df = df.sort_values('date_parsed').reset_index(drop=True)

target = 'runs'

num_features = [
    'runs_rolling_5',
    'balls_rolling_5',
    'strike_rate_rolling_5',
    'career_runs',
    'career_matches',
    'dismissals'
]

cat_features = ['venue', 'team1', 'team2', 'new_player']

X = df[num_features + cat_features]
y = df[target]
dates = df['date_parsed']

# Time-based split (fixed for fair comparison)
split_date = dates.quantile(0.80)
X_train = X[dates <= split_date]
y_train = y[dates <= split_date]
X_val   = X[dates > split_date]
y_val   = y[dates > split_date]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 700),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": -1
    }

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features)
    ])

    model = XGBRegressor(**params)

    pipe = Pipeline([
        ('prep', preprocessor),
        ('xgb', model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    rmse = root_mean_squared_error(y_val, preds)
    return rmse


def main():
    print("Starting Optuna hyperparameter tuning (XGBoost)...\n")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=15)

    print("\nBest RMSE:", study.best_value)
    print("Best parameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Train final model on best params
    best_model = XGBRegressor(
        **study.best_params,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    final_pipe = Pipeline([
        ('prep', ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), num_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), cat_features)
        ])),
        ('xgb', best_model)
    ])

    final_pipe.fit(X_train, y_train)

    ART.mkdir(exist_ok=True)
    joblib.dump(final_pipe, ART / "xgboost_tuned.pkl")

    print("\n💾 Saved tuned model to artifacts/xgboost_tuned.pkl")


if __name__ == "__main__":
    main()
