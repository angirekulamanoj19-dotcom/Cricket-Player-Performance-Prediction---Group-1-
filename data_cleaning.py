import pandas as pd
import numpy as np

def load_and_clean_data(matches_path, deliveries_path):
    """
    Loads IPL matches and deliveries data, merges them, and cleans the dataset.
    """
    print("Loading datasets...")
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)

    # 1. Merging the datasets
    # Using 'match_id' from deliveries and 'id' from matches
    print("Merging data...")
    df = pd.merge(deliveries, matches, left_on='match_id', right_on='id')

    # 2. Data Cleaning
    print("Cleaning data...")
    
    # Drop the ghost row (the nulls we found earlier)
    df.dropna(subset=['total_runs', 'over', 'ball'], inplace=True)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Fill categorical missing values
    df['city'] = df['city'].fillna('Unknown')
    df['extras_type'] = df['extras_type'].fillna('none')
    df['player_dismissed'] = df['player_dismissed'].fillna('none')
    df['dismissal_kind'] = df['dismissal_kind'].fillna('none')
    df['fielder'] = df['fielder'].fillna('none')
    df['method'] = df['method'].fillna('standard')

    # Fill numerical missing values
    df['result_margin'] = df['result_margin'].fillna(0)

    print(f"Cleaning complete. Final dataset shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Define file paths
    MATCHES_FILE = 'matches.csv'
    DELIVERIES_FILE = 'deliveries.csv'
    OUTPUT_FILE = 'cleaned_unified_data.csv'

    # Execute the cleaning pipeline
    cleaned_df = load_and_clean_data(MATCHES_FILE, DELIVERIES_FILE)

    # Save the cleaned dataset for future modeling
    cleaned_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned data saved to {OUTPUT_FILE}")