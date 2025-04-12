import sys
import json
import xgboost as xgb
import pandas as pd
import joblib

# Load the model
model = xgb.Booster()
model.load_model("model.json")

# Load the scaler used for training
scaler = joblib.load('scaler.pkl')

# Read input from stdin
input_json = sys.stdin.read()
input_data = json.loads(input_json)

# Create DataFrame for input data
df = pd.DataFrame([input_data])

# Scale the input data using the same scaler as training
input_data_scaled = scaler.transform(df)

# Predict
dtest = xgb.DMatrix(input_data_scaled)
prediction = model.predict(dtest)[0]

# Return result as JSON (without debug printing)
print(json.dumps({"healthScore": round(float(prediction), 2)}))

