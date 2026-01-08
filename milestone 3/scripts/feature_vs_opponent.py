# scripts/feature_vs_opponent.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def main():
    df = pd.read_csv(PROC / "batsman_match.csv")
    df['date_parsed'] = pd.to_datetime(df['date_parsed'])

    # sort chronologically (VERY IMPORTANT)
    df = df.sort_values('date_parsed').reset_index(drop=True)

    records = []
    history = {}

    for _, row in df.iterrows():
        batsman = row['batsman']
        team1 = row['team1']
        team2 = row['team2']

        # We create two perspectives: vs team1 and vs team2
        for opponent in [team1, team2]:
            key = (batsman, opponent)

            past = history.get(key, {
                'runs': 0,
                'balls': 0,
                'matches': 0
            })

            avg_runs = past['runs'] / past['matches'] if past['matches'] > 0 else 0
            strike_rate = (past['runs'] / past['balls'] * 100) if past['balls'] > 0 else 0

            records.append({
                'match_id': row['match_id'],
                'batsman': batsman,
                'opponent_team': opponent,
                'vs_team_avg_runs': avg_runs,
                'vs_team_strike_rate': strike_rate,
                'vs_team_matches': past['matches']
            })

            # update history AFTER using it
            history.setdefault(key, {'runs': 0, 'balls': 0, 'matches': 0})
            history[key]['runs'] += row['runs']
            history[key]['balls'] += row['balls']
            history[key]['matches'] += 1

    vs_df = pd.DataFrame(records)

    out = PROC / "batsman_vs_opponent_features.csv"
    vs_df.to_csv(out, index=False)

    print("✅ Saved opponent features to:", out)
    print(vs_df.head())

if __name__ == "__main__":
    main()
