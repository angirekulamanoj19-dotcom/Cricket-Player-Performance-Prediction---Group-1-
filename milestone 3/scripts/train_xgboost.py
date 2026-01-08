# scripts/xgb_baseline.py
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
ART = ROOT / "artifacts"

def main():
    df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])

    # time-aware ordering
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

    # time-based split
    split_date = df['date_parsed'].quantile(0.80)
    X_train = X[df['date_parsed'] <= split_date]
    y_train = y[df['date_parsed'] <= split_date]
    X_test  = X[df['date_parsed'] > split_date]
    y_test  = y[df['date_parsed'] > split_date]

    preprocessor = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), num_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_features)
    ])

    # 🔹 XGBoost BASELINE (light, untuned)
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ('prep', preprocessor),
        ('xgb', model)
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("XGBOOST BASELINE")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    ART.mkdir(exist_ok=True)
    joblib.dump(pipe, ART / "xgb_baseline.pkl")
    print("\nSaved baseline model to artifacts/xgb_baseline.pkl")

if __name__ == "__main__":
    main()
