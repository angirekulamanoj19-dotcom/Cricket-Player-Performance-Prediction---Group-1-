# train_data.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# -------------------------------
# Step 1: Load the data
# -------------------------------
train_df = pd.read_csv('data/final_batsman_features.csv')
test_df  = pd.read_csv('data/final_batsman_features.csv')  # or your separate test.csv if available

print("Train columns:", train_df.columns)
print("Test columns:", test_df.columns)

# -------------------------------
# Step 2: Split features and target
# -------------------------------
target_col = 'runs_next_match'

X_train = train_df.drop(target_col, axis=1)
y_train = train_df[target_col]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# -------------------------------
# Step 3: Train the model
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# Step 4: Evaluate the model
# -------------------------------
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# -------------------------------
# Step 5: Save the model
# -------------------------------
joblib.dump(model, "batsman_runs_model.pkl")
print("Trained model saved as batsman_runs_model.pkl")
import joblib

# Save the trained model
joblib.dump(model, "batsman_runs_model.pkl")
print("Trained model saved as batsman_runs_model.pkl")
model = joblib.load("batsman_runs_model.pkl")
