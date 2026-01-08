import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

def main():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct paths relative to the script
    raw_file = os.path.join(script_dir, "..", "data", "IPL_ball_by_ball_updated.csv")
    processed_file = os.path.join(script_dir, "..", "data", "processed_deliveries.csv")
    normalized_file = os.path.join(script_dir, "..", "data", "processed_deliveries_normalized.csv")

    # Load raw CSV
    if not os.path.exists(raw_file):
        print(f"Error: {raw_file} not found!")
        return

    deliveries = pd.read_csv(raw_file)

    # Clean column names
    deliveries.columns = deliveries.columns.str.strip().str.lower().str.replace(' ', '_')

    # Handle missing values
    numeric_cols = deliveries.select_dtypes(include='number').columns
    categorical_cols = deliveries.select_dtypes(include='object').columns
    deliveries[numeric_cols] = deliveries[numeric_cols].fillna(0)
    deliveries[categorical_cols] = deliveries[categorical_cols].fillna('Unknown')

    # Feature engineering
    deliveries['total_runs'] = deliveries['runs_off_bat'] + deliveries['extras']
    deliveries['wicket'] = deliveries['player_dismissed'].notnull().astype(int)
    deliveries['cumulative_runs'] = deliveries.groupby('striker')['runs_off_bat'].cumsum()

    # Save cleaned CSV
    deliveries.to_csv(processed_file, index=False)
    print(f"Cleaned data saved to {processed_file}")

    # Normalize numeric columns
    numeric_features = ['innings', 'runs_off_bat', 'extras', 'wicket', 'total_runs', 'cumulative_runs']
    scaler = MinMaxScaler()
    deliveries[numeric_features] = scaler.fit_transform(deliveries[numeric_features])

    # Save normalized CSV
    deliveries.to_csv(normalized_file, index=False)
    print(f"Normalized data saved to {normalized_file}")

if __name__ == "__main__":
    main()
