# scripts/evaluate_final_model.py
from pathlib import Path
import pandas as pd
import joblib

from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

def main():
    # load data
    df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])

    # sort chronologically
    df = df.sort_values('date_parsed').reset_index(drop=True)

    target = 'runs'

    features = [
        'runs_rolling_5',
        'balls_rolling_5',
        'strike_rate_rolling_5',
        'career_runs',
        'career_matches',
        'dismissals',
        'venue',
        'team1',
        'team2',
        'new_player'
    ]

    X = df[features]
    y = df[target]
    dates = df['date_parsed']

    # same split logic as before (important!)
    split_date = dates.quantile(0.80)

    X_test = X[dates > split_date]
    y_test = y[dates > split_date]

    # load tuned model
    model = joblib.load(ART / "xgboost_tuned.pkl")

    # predict
    preds = model.predict(X_test)

    # metrics
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("FINAL TEST SET EVALUATION (Tuned XGBoost)")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

if __name__ == "__main__":
    main()
