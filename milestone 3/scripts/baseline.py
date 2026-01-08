# scripts/baseline.py
# Baseline: 10-match rolling average for batsman runs

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def main():
    df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])

    df = df.sort_values(['batsman','date_parsed']).reset_index(drop=True)

    df['baseline_rolling_10'] = (
        df.groupby('batsman')['runs']
          .transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
    )

    baseline_df = df.dropna(subset=['baseline_rolling_10'])

    split_date = baseline_df['date_parsed'].quantile(0.80)
    test = baseline_df[baseline_df['date_parsed'] > split_date]

    y_true = test['runs']
    y_pred = test['baseline_rolling_10']

    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print("BASELINE (10-match rolling avg)")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    out = PROC / "baseline_predictions.csv"
    test[['match_id','batsman','date_parsed','runs','baseline_rolling_10']].to_csv(out, index=False)

if __name__ == "__main__":
    main()
