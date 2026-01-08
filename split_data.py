import pandas as pd

# Load your dataset
df = pd.read_csv('data/final_batsman_features.csv')  # CSV is inside data folder

# Split based on matches played
train_df = df[df['matches_played'] <= 50]  # first 50 matches for training
test_df  = df[df['matches_played'] > 50]   # rest of the matches

# Save the train and test files
train_df.to_csv('data/train.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

print("Train and Test CSV files created!")
