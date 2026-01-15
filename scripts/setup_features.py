"""
Quick setup script to generate required feature datasets for the dashboard
Run this if batting_features_dataset.csv doesn't exist
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_features():
    """Create sample feature dataset for dashboard testing"""
    
    print("Generating sample batting features dataset...")
    
    # Load raw data
    batting_card = pd.read_csv('dataset/t20i_Batting_Card.csv')
    matches = pd.read_csv('dataset/t20i_Matches_Data.csv')
    
    # Convert dates - handle errors gracefully
    matches['Match Date'] = pd.to_datetime(matches['Match Date'], errors='coerce')
    
    # Merge batting with match data
    batting_merged = batting_card.merge(
        matches[['Match ID', 'Match Date', 'Match Venue (City)', 'Match Venue (Country)']],
        on='Match ID',
        how='left'
    )
    
    # Drop rows with invalid dates
    batting_merged = batting_merged.dropna(subset=['Match Date'])
    
    # Sort by player and date
    batting_merged = batting_merged.sort_values(['batsman', 'Match Date'])
    
    # Calculate rolling features for each player
    features_list = []
    
    for player in batting_merged['batsman'].unique():
        player_data = batting_merged[batting_merged['batsman'] == player].copy()
        
        if len(player_data) < 2:
            continue
        
        # Create next match target
        player_data['next_match_runs'] = player_data['runs'].shift(-1)
        
        # Rolling statistics
        player_data['runs_last_3'] = player_data['runs'].rolling(3, min_periods=1).mean()
        player_data['runs_last_5'] = player_data['runs'].rolling(5, min_periods=1).mean()
        player_data['runs_last_10'] = player_data['runs'].rolling(10, min_periods=1).mean()
        
        # Strike rate features
        player_data['strikeRate'] = pd.to_numeric(player_data['strikeRate'], errors='coerce')
        player_data['sr_last_3'] = player_data['strikeRate'].rolling(3, min_periods=1).mean()
        player_data['sr_last_5'] = player_data['strikeRate'].rolling(5, min_periods=1).mean()
        player_data['sr_last_10'] = player_data['strikeRate'].rolling(10, min_periods=1).mean()
        
        # Career stats
        player_data['total_matches'] = range(1, len(player_data) + 1)
        player_data['total_runs'] = player_data['runs'].cumsum()
        player_data['total_innings'] = player_data['total_matches']
        player_data['avg_score'] = player_data['total_runs'] / player_data['total_innings']
        player_data['overall_sr'] = player_data['strikeRate'].expanding().mean()
        
        # Batting position (innings 1 or 2)
        player_data['batting_position'] = player_data['innings']
        
        # Add to list
        features_list.append(player_data)
    
    # Combine all players
    batting_features = pd.concat(features_list, ignore_index=True)
    
    # Remove last row for each player (no next match)
    batting_features = batting_features.dropna(subset=['next_match_runs'])
    
    # Fill any remaining NaNs
    batting_features = batting_features.fillna(0)
    
    # Save to CSV
    output_path = 'dataset/batting_features_dataset.csv'
    
    # Rename columns to match expected format
    batting_features = batting_features.rename(columns={
        'batsman': 'Player Name'
    })
    
    batting_features.to_csv(output_path, index=False)
    
    print(f"✅ Created {output_path}")
    print(f"   Total records: {len(batting_features):,}")
    print(f"   Unique players: {batting_features['Player Name'].nunique():,}")
    print(f"   Features: {len(batting_features.columns)}")
    
    return batting_features

def create_bowling_features():
    """Create sample bowling features dataset"""
    
    print("\nGenerating sample bowling features dataset...")
    
    try:
        bowling_card = pd.read_csv('dataset/t20i_Bowling_Card.csv')
        matches = pd.read_csv('dataset/t20i_Matches_Data.csv')
        
        # Convert dates
        matches['Match Date'] = pd.to_datetime(matches['Match Date'])
        
        # Merge
        bowling_merged = bowling_card.merge(
            matches[['Match ID', 'Match Date', 'Match Venue (City)', 'Match Venue (Country)']],
            on='Match ID',
            how='left'
        )
        
        # Basic features - using actual column names
        if 'bowler' in bowling_merged.columns:
            bowling_merged = bowling_merged.rename(columns={'bowler': 'Player Name'})
        
        # Save
        output_path = 'dataset/bowling_features_dataset.csv'
        bowling_merged.to_csv(output_path, index=False)
        
        print(f"✅ Created {output_path}")
        print(f"   Total records: {len(bowling_merged):,}")
    except Exception as e:
        print(f"⚠️  Skipping bowling features: {str(e)}")
    
    return None

if __name__ == "__main__":
    print("="*60)
    print("Cricket Score Prediction - Feature Dataset Generator")
    print("="*60)
    
    try:
        batting_features = create_sample_features()
        bowling_features = create_bowling_features()
        
        print("\n" + "="*60)
        print("✅ SUCCESS! Feature datasets created.")
        print("="*60)
        print("\nYou can now run the dashboard:")
        print("  streamlit run app.py")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPlease ensure the following files exist in dataset/:")
        print("  - t20i_Batting_Card.csv")
        print("  - t20i_Bowling_Card.csv")
        print("  - t20i_Matches_Data.csv")
