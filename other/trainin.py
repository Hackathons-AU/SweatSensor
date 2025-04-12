import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the CSV file with biomarkers and health scores
df = pd.read_csv("synthetic_sweat_data.csv")

# Input features
X = df[["Sodium", "Glucose", "Hydration", "Lactate"]]

# Output target: Health Score
y = df["HealthScore"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler for test data

# XGBoost regression model
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,  # Increase number of estimators
    learning_rate=0.05,  # Try a lower learning rate
    max_depth=6,  # Increase depth of trees
    random_state=42
)

# Train the model
model.fit(X_train_scaled, y_train)

# Print the predictions on test data
y_pred = model.predict(X_test_scaled)
print("Predictions on test data:", y_pred)

# Evaluate the performance again
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.2f}")



# Save the model and scaler
model.get_booster().save_model("model.json")
import joblib
joblib.dump(scaler, 'scaler.pkl')  # Save the scaler to apply later in predict.py

