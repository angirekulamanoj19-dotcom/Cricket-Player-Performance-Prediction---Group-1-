import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# LOAD DATA
# --------------------------------
print("Loading deliveries.csv ...")
df = pd.read_csv("deliveries.csv")

print("Total rows:", len(df))

# --------------------------------
# SAMPLE DATA FOR FAST PLOTTING
# --------------------------------
# Use sample for plots to avoid buffering
df_sample = df.sample(n=200000, random_state=42)

# --------------------------------
# RUNS DISTRIBUTION
# --------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df_sample["total_runs"], bins=25)
plt.title("Distribution of Total Runs per Ball")
plt.xlabel("Runs")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --------------------------------
# WICKETS DISTRIBUTION
# --------------------------------
plt.figure(figsize=(10, 5))
sns.countplot(
    x=df_sample["dismissal_kind"].fillna("Not Out"),
    order=df_sample["dismissal_kind"].fillna("Not Out").value_counts().index
)
plt.xticks(rotation=45)
plt.title("Wicket Types Distribution")
plt.xlabel("Dismissal Type")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --------------------------------
# TEAM PERFORMANCE (BATTING)
# --------------------------------
team_runs = df.groupby("batting_team")["total_runs"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
team_runs.plot(kind="bar")
plt.title("Total Runs by Each Team (Batting Performance)")
plt.xlabel("Team")
plt.ylabel("Runs")
plt.tight_layout()
plt.show()

# --------------------------------
# TEAM PERFORMANCE (BOWLING)
# --------------------------------
# Create wicket column safely
df["is_wicket"] = df["dismissal_kind"].notna().astype(int)

team_wickets = df.groupby("bowling_team")["is_wicket"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
team_wickets.plot(kind="bar", color="red")
plt.title("Total Wickets Taken by Each Team (Bowling Performance)")
plt.xlabel("Team")
plt.ylabel("Wickets")
plt.tight_layout()
plt.show()

# --------------------------------
# VENUE STATS (OPTIONAL)
# Requires matches.csv
# --------------------------------
try:
    matches = pd.read_csv("matches.csv")

    merged = df.merge(
        matches[["id", "venue"]],
        left_on="match_id",
        right_on="id",
        how="inner"
    )

    venue_runs = (
        merged.groupby("venue")["total_runs"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 6))
    venue_runs.plot(kind="bar")
    plt.title("Top 10 High Scoring Venues")
    plt.xlabel("Venue")
    plt.ylabel("Total Runs")
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print("matches.csv not found → Venue analysis skipped")

print("EDA completed successfully.")
