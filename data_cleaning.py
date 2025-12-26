import pandas as pd
import numpy as np
from datetime import datetime
import os

class DataCleaner:
    def __init__(self, data_dir='dataset'):
        self.data_dir = data_dir
        self.matches = None
        self.batting = None
        self.bowling = None
        self.players = None
        self.partnership = None
        self.fow = None
        
    def load_data(self):
        self.matches = pd.read_csv(f'{self.data_dir}/t20i_Matches_Data.csv')
        self.batting = pd.read_csv(f'{self.data_dir}/t20i_Batting_Card.csv')
        self.bowling = pd.read_csv(f'{self.data_dir}/t20i_Bowling_Card.csv')
        self.players = pd.read_csv(f'{self.data_dir}/players_info.csv')
        self.partnership = pd.read_csv(f'{self.data_dir}/t20i_Partnership_Card.csv')
        self.fow = pd.read_csv(f'{self.data_dir}/t20i_Fow_Card.csv')
        
        return self
    
    def clean_matches(self):
        df = self.matches.copy()
        
        df['Match Date'] = pd.to_datetime(df['Match Date'], errors='coerce')
        
        df['Team1 Runs Scored'] = pd.to_numeric(df['Team1 Runs Scored'], errors='coerce')
        df['Team2 Runs Scored'] = pd.to_numeric(df['Team2 Runs Scored'], errors='coerce')
        df['Team1 Wickets Fell'] = pd.to_numeric(df['Team1 Wickets Fell'], errors='coerce')
        df['Team2 Wickets Fell'] = pd.to_numeric(df['Team2 Wickets Fell'], errors='coerce')
        df['Team1 Extras Rec'] = pd.to_numeric(df['Team1 Extras Rec'], errors='coerce')
        df['Team2 Extras Rec'] = pd.to_numeric(df['Team2 Extras Rec'], errors='coerce')
        
        df['Match Venue (City)'] = df['Match Venue (City)'].fillna('Unknown')
        df['Match Venue (Country)'] = df['Match Venue (Country)'].fillna('Unknown')
        df['Match Venue (Stadium)'] = df['Match Venue (Stadium)'].fillna('Unknown')
        
        df['Team1 Name'] = df['Team1 Name'].fillna('Unknown')
        df['Team2 Name'] = df['Team2 Name'].fillna('Unknown')
        df['Toss Winner'] = df['Toss Winner'].fillna('Unknown')
        df['Toss Winner Choice'] = df['Toss Winner Choice'].fillna('bat')
        df['Match Winner'] = df['Match Winner'].fillna('Unknown')
        
        df = df.drop_duplicates(subset=['Match ID'])
        
        df = df[df['Match Date'].notna()]
        df = df[df['Team1 Runs Scored'].notna() | df['Team2 Runs Scored'].notna()]
        
        self.matches = df
        return self
    
    def clean_batting(self):
        df = self.batting.copy()
        
        df['runs'] = pd.to_numeric(df['runs'], errors='coerce').fillna(0)
        df['balls'] = pd.to_numeric(df['balls'], errors='coerce').fillna(0)
        df['fours'] = pd.to_numeric(df['fours'], errors='coerce').fillna(0)
        df['sixes'] = pd.to_numeric(df['sixes'], errors='coerce').fillna(0)
        df['strikeRate'] = pd.to_numeric(df['strikeRate'], errors='coerce').fillna(0)
        
        df['batsman'] = pd.to_numeric(df['batsman'], errors='coerce')
        df['bowler'] = pd.to_numeric(df['bowler'], errors='coerce')
        
        df['isOut'] = df['isOut'].fillna(False)
        df['wicketType'] = df['wicketType'].fillna('not out')
        
        df = df[df['Match ID'].notna()]
        
        df.loc[df['balls'] > 0, 'strikeRate'] = (df['runs'] / df['balls']) * 100
        
        self.batting = df
        return self
    
    def clean_bowling(self):
        df = self.bowling.copy()
        
        df['overs'] = pd.to_numeric(df['overs'], errors='coerce').fillna(0)
        df['balls'] = pd.to_numeric(df['balls'], errors='coerce').fillna(0)
        df['maidens'] = pd.to_numeric(df['maidens'], errors='coerce').fillna(0)
        df['conceded'] = pd.to_numeric(df['conceded'], errors='coerce').fillna(0)
        df['wickets'] = pd.to_numeric(df['wickets'], errors='coerce').fillna(0)
        df['economy'] = pd.to_numeric(df['economy'], errors='coerce').fillna(0)
        df['dots'] = pd.to_numeric(df['dots'], errors='coerce').fillna(0)
        df['fours'] = pd.to_numeric(df['fours'], errors='coerce').fillna(0)
        df['sixes'] = pd.to_numeric(df['sixes'], errors='coerce').fillna(0)
        df['wides'] = pd.to_numeric(df['wides'], errors='coerce').fillna(0)
        df['noballs'] = pd.to_numeric(df['noballs'], errors='coerce').fillna(0)
        
        df['bowler id'] = pd.to_numeric(df['bowler id'], errors='coerce')
        
        df = df[df['Match ID'].notna()]
        
        df.loc[df['overs'] > 0, 'economy'] = df['conceded'] / df['overs']
        
        self.bowling = df
        return self
    
    def clean_players(self):
        if self.players is not None:
            df = self.players.copy()
            
            df['Player Name'] = df['Player Name'].fillna('Unknown')
            df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
            
            df = df.drop_duplicates(subset=['Player ID'])
            
            self.players = df
        return self
    
    def clean_partnership(self):
        if self.partnership is not None:
            df = self.partnership.copy()
            
            numeric_cols = ['runs', 'balls']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df = df[df['Match ID'].notna()]
            
            self.partnership = df
        return self
    
    def clean_fow(self):
        if self.fow is not None:
            df = self.fow.copy()
            
            numeric_cols = ['runs', 'overs']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df = df[df['Match ID'].notna()]
            
            self.fow = df
        return self
    
    def clean_all(self):
        self.clean_matches()
        self.clean_batting()
        self.clean_bowling()
        self.clean_players()
        self.clean_partnership()
        self.clean_fow()
        return self
    
    def save_cleaned_data(self, output_dir='dataset/cleaned'):
        os.makedirs(output_dir, exist_ok=True)
        
        self.matches.to_csv(f'{output_dir}/matches_cleaned.csv', index=False)
        self.batting.to_csv(f'{output_dir}/batting_cleaned.csv', index=False)
        self.bowling.to_csv(f'{output_dir}/bowling_cleaned.csv', index=False)
        
        if self.players is not None:
            self.players.to_csv(f'{output_dir}/players_cleaned.csv', index=False)
        if self.partnership is not None:
            self.partnership.to_csv(f'{output_dir}/partnership_cleaned.csv', index=False)
        if self.fow is not None:
            self.fow.to_csv(f'{output_dir}/fow_cleaned.csv', index=False)
        
        return self
    
    def get_summary(self):
        summary = {
            'matches': {
                'total_records': len(self.matches),
                'date_range': f"{self.matches['Match Date'].min()} to {self.matches['Match Date'].max()}",
                'missing_values': self.matches.isnull().sum().sum(),
                'unique_teams': self.matches['Team1 Name'].nunique() + self.matches['Team2 Name'].nunique()
            },
            'batting': {
                'total_records': len(self.batting),
                'missing_values': self.batting.isnull().sum().sum(),
                'unique_batsmen': self.batting['batsman'].nunique()
            },
            'bowling': {
                'total_records': len(self.bowling),
                'missing_values': self.bowling.isnull().sum().sum(),
                'unique_bowlers': self.bowling['bowler id'].nunique()
            }
        }
        return summary

def main():
    cleaner = DataCleaner()
    
    print("Loading data...")
    cleaner.load_data()
    
    print("Cleaning data...")
    cleaner.clean_all()
    
    print("Saving cleaned data...")
    cleaner.save_cleaned_data()
    
    print("\nData Cleaning Summary:")
    summary = cleaner.get_summary()
    for dataset, stats in summary.items():
        print(f"\n{dataset.upper()}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    print("\nData cleaning completed successfully!")

if __name__ == "__main__":
    main()
