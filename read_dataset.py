import pandas as pd

# Load raw ball-by-ball data
df = pd.read_csv("deliveries.csv")

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Basic missing value handling (ONLY minimal)
df['player_dismissed'] = df['player_dismissed'].fillna('NA')
df['dismissal_kind'] = df['dismissal_kind'].fillna('NA')
df['fielder'] = df['fielder'].fillna('NA')

# Display basic info (for EDA)
print("Shape of data:", df.shape)
print("\nColumns:\n", df.columns)
print("\nSample rows:")
print(df.head())
