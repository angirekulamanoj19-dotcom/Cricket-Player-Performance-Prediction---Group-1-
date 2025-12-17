import pandas as pd

print("Starting data cleaning...")

# Load merged data
df = pd.read_csv("merge.csv", low_memory=False)
print("Before cleaning:", df.shape)

# Make column names consistent
df.columns = df.columns.str.lower()

# Fix date format
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove rows with invalid dates
df = df.dropna(subset=["date"])

# Extract date parts
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

# Handle missing text values
text_columns = [
    "venue",
    "player_dismissed",
    "dismissal_kind",
    "fielder",
    "extras_type"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# Handle missing numeric values
numeric_columns = [
    "batsman_runs",
    "extra_runs",
    "total_runs",
    "is_wicket"
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Remove logically incorrect records
df = df[df["batsman_runs"] >= 0]
df = df[df["extra_runs"] >= 0]
df = df[df["total_runs"] >= 0]
df = df[df["total_runs"] >= df["batsman_runs"]]

if "over" in df.columns:
    df = df[df["over"] <= 20]

df = df[df["is_wicket"].isin([0, 1])]

# Drop columns not useful for modeling
drop_columns = [
    "method",
    "umpire1",
    "umpire2",
    "city"
]

df = df.drop(columns=[c for c in drop_columns if c in df.columns])

# Remove duplicates
df = df.drop_duplicates()

# Final check
print("After cleaning:", df.shape)
print("\nRemaining missing values:")
print(df.isna().sum())

# Save cleaned data
df.to_csv("clean_data.csv", index=False)

print("\n Data cleaning completed successfully")
print("Saved file: clean_data.csv")
