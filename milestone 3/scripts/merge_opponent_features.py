# scripts/merge_opponent_features.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def main():
    base = pd.read_csv(PROC / "batsman_match_model_ready.csv")
    vs   = pd.read_csv(PROC / "batsman_vs_opponent_features.csv")

    # determine opponent team for each row
    base['opponent_team'] = base.apply(
        lambda r: r['team2'] if r['team1'] == r['batting_team']
        else r['team1'],
        axis=1
    ) if 'batting_team' in base.columns else base.apply(
        lambda r: r['team2'] if r['team1'] != r['team2'] else r['team1'],
        axis=1
    )

    # merge on match_id, batsman, opponent_team
    df = base.merge(
        vs,
        on=['match_id', 'batsman', 'opponent_team'],
        how='left'
    )

    # fill cold-start cases
    df[['vs_team_avg_runs',
        'vs_team_strike_rate',
        'vs_team_matches']] = df[[
            'vs_team_avg_runs',
            'vs_team_strike_rate',
            'vs_team_matches'
        ]].fillna(0)

    # drop helper column
    df = df.drop(columns=['opponent_team'])

    out = PROC / "batsman_match_model_ready_v2.csv"
    df.to_csv(out, index=False)

    print("✅ Saved enhanced model-ready dataset to:", out)
    print("\nNew feature summary:")
    print(df[['vs_team_avg_runs','vs_team_strike_rate','vs_team_matches']].describe())

if __name__ == "__main__":
    main()
