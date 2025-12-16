import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("deliveries.csv")

# 1. Remove duplicates
df.drop_duplicates(inplace=True)

# 2. Fill missing values
num_cols = ['over', 'ball', 'batsman_runs', 'extra_runs', 'total_runs']
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. Outlier capping
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# 4. Normalization
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Date column? Check before using.
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].fillna(df["date"].min())
else:
    print("No 'date' column found — skipping date cleaning.")

# 6. Display result
print(df.head())

