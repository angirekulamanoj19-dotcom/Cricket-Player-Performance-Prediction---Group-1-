import pandas as pd

def load_and_clean_data(matches_path='matches.csv', deliveries_path='deliveries.csv'):
    # Load data
    matches = pd.read_csv(matches_path)
    deliveries = pd.read_csv(deliveries_path)
    
    # Handle missing values
    matches['city'].fillna('Unknown', inplace=True)
    matches.dropna(subset=['winner'], inplace=True)
    deliveries['player_dismissed'].fillna('Not Out', inplace=True)
    deliveries['dismissal_kind'].fillna('Not Out', inplace=True)
    
    # Normalize team names
    team_name_map = {
        'Delhi Daredevils': 'Delhi Capitals',
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiant': 'Rising Pune Supergiants'
    }
    matches['team1'] = matches['team1'].replace(team_name_map)
    matches['team2'] = matches['team2'].replace(team_name_map)
    matches['winner'] = matches['winner'].replace(team_name_map)
    deliveries['batting_team'] = deliveries['batting_team'].replace(team_name_map)
    deliveries['bowling_team'] = deliveries['bowling_team'].replace(team_name_map)
    
    # Ensure date formats
    matches['date'] = pd.to_datetime(matches['date'], errors='coerce')
    
    # Remove duplicates
    matches.drop_duplicates(inplace=True)
    deliveries.drop_duplicates(inplace=True)
    
    return matches, deliveries

if __name__ == "__main__":
    matches_clean, deliveries_clean = load_and_clean_data()
    print("Cleaning complete. Shapes:", matches_clean.shape, deliveries_clean.shape)