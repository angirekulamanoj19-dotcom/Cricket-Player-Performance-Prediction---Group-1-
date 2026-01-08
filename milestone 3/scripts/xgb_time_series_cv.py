# scripts/xgb_time_series_cv.py
from pathlib import Path
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def main():
    df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])

    # sort chronologically
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

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features)
    ])

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ('prep', preprocessor),
        ('xgb', model)
    ])

    # -------- Time-series CV splits (expanding window) --------
    quantiles = [0.6, 0.7, 0.8]
    results = []

    print("\n⏳ Running Time-Series Cross-Validation...\n")

    for q in quantiles:
        split_date = dates.quantile(q)

        X_train = X[dates <= split_date]
        y_train = y[dates <= split_date]
        X_val   = X[dates > split_date]
        y_val   = y[dates > split_date]

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)

        rmse = root_mean_squared_error(y_val, preds)
        mae  = mean_absolute_error(y_val, preds)
        r2   = r2_score(y_val, preds)

        results.append((q, rmse, mae, r2))

        print(f"Fold (train ≤ {int(q*100)}%)")
        print(f"RMSE: {rmse:.3f} | MAE: {mae:.3f} | R²: {r2:.3f}\n")

    print("CV Summary (mean):")
    print(f"RMSE: {np.mean([r[1] for r in results]):.3f}")
    print(f"MAE : {np.mean([r[2] for r in results]):.3f}")
    print(f"R²  : {np.mean([r[3] for r in results]):.3f}")

if __name__ == "__main__":
    main()
