# scripts/shap_explain.py
from pathlib import Path
import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ART  = ROOT / "artifacts"
OUT  = ROOT / "artifacts" / "shap_outputs"

OUT.mkdir(exist_ok=True)

def main():
    print("🔍 Loading model and data...")
    model = joblib.load(ART / "xgboost_tuned.pkl")

    df = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])
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

    # Use a small sample for SHAP (speed + clarity)
    X_sample = X.sample(500, random_state=42)

    print("Creating SHAP explainer...")
    explainer = shap.Explainer(model.named_steps['xgb'],
                               model.named_steps['prep'].transform(X_sample))

    shap_values = explainer(model.named_steps['prep'].transform(X_sample))

    # ---- Global importance plot ----
    print("Saving global SHAP summary plot...")
    plt.figure()
    shap.summary_plot(shap_values, show=False)
    plt.savefig(OUT / "shap_summary.png", bbox_inches="tight")
    plt.close()

    # ---- Local explanation ----
    print("Saving local explanation plot...")
    plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    plt.savefig(OUT / "shap_local_example.png", bbox_inches="tight")
    plt.close()

    print("\n✅ SHAP explainability completed")
    print("Saved files:")
    print(" - shap_outputs/shap_summary.png")
    print(" - shap_outputs/shap_local_example.png")

if __name__ == "__main__":
    main()
