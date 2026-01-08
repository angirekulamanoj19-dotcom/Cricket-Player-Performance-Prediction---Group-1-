# scripts/train_random_forest.py
from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART = ROOT / "artifacts"

def main():
    df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])

    # sort for time-aware split
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

    # time-based split (80/20)
    split_date = df['date_parsed'].quantile(0.80)
    X_train = X[df['date_parsed'] <= split_date]
    y_train = y[df['date_parsed'] <= split_date]
    X_test  = X[df['date_parsed'] > split_date]
    y_test  = y[df['date_parsed'] > split_date]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipe, num_features),
        ('cat', cat_pipe, cat_features)
    ])

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ('prep', preprocessor),
        ('rf', model)
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)

    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print("RANDOM FOREST REGRESSOR")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")

    # save model
    ART.mkdir(exist_ok=True)
    joblib.dump(pipe, ART / "random_forest_runs.pkl")
    print("\nSaved model to artifacts/random_forest_runs.pkl")

if __name__ == "__main__":
    main()
