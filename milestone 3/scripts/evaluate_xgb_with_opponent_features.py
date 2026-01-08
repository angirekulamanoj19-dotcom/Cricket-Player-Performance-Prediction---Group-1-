# scripts/evaluate_xgb_with_opponent_features.py
from pathlib import Path
import pandas as pd
import joblib

from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART  = ROOT / "artifacts"

def main():
    df = pd.read_csv(PROC / "batsman_match_model_ready_v2.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
    df = df.sort_values('date_parsed').reset_index(drop=True)

    target = 'runs'

    num_features = [
        'runs_rolling_5',
        'balls_rolling_5',
        'strike_rate_rolling_5',
        'career_runs',
        'career_matches',
        'dismissals',
        'vs_team_avg_runs',
        'vs_team_strike_rate',
        'vs_team_matches'
    ]

    cat_features = ['venue', 'team1', 'team2', 'new_player']

    X = df[num_features + cat_features]
    y = df[target]
    dates = df['date_parsed']

    split_date = dates.quantile(0.80)

    X_train = X[dates <= split_date]
    y_train = y[dates <= split_date]
    X_test  = X[dates > split_date]
    y_test  = y[dates > split_date]

    # best params from Optuna (fixed)
    model = XGBRegressor(
        n_estimators=414,
        max_depth=4,
        learning_rate=0.031183273264034746,
        subsample=0.6062817315297675,
        colsample_bytree=0.7636639231722119,
        min_child_weight=10,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features)
    ])

    pipe = Pipeline([
        ('prep', preprocessor),
        ('xgb', model)
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("🎯 XGBOOST + OPPONENT FEATURES (FINAL TEST)")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    # save if better
    if rmse < 20.918:
        joblib.dump(pipe, ART / "xgboost_final.pkl")
        print("\nImprovement detected — saved as xgboost_final.pkl")
    else:
        print("\n⚠️ No improvement — keeping previous final model")

if __name__ == "__main__":
    main()
