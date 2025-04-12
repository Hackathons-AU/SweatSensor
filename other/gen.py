import numpy as np
import pandas as pd

# Set a random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for features
n_samples = 200  # Number of samples

# Sodium: Between 50 and 130, simulating normal levels of sodium in sweat
Sodium = np.random.uniform(50, 130, n_samples)

# Glucose: Between 60 and 150, simulating glucose levels
Glucose = np.random.uniform(60, 150, n_samples)

# Hydration: Between 0.3 and 3, simulating hydration levels
Hydration = np.random.uniform(0.3, 3, n_samples)

# Lactate: Between 0.5 and 5, simulating lactate levels
Lactate = np.random.uniform(0.5, 5, n_samples)

# HealthScore: A formula based on the features, with some added noise
# This is just a simple linear model for illustration
health_score = (0.3 * Sodium + 0.2 * Glucose + 10 * Hydration - 2 * Lactate + np.random.normal(0, 5, n_samples))

# Ensure that the health score is between 50 and 100
health_score = np.clip(health_score, 50, 100)

# Create a DataFrame
df = pd.DataFrame({
    'Sodium': Sodium,
    'Glucose': Glucose,
    'Hydration': Hydration,
    'Lactate': Lactate,
    'HealthScore': health_score
})

# Save to CSV
df.to_csv('synthetic_sweat_data.csv', index=False)

print("✅ Synthetic dataset generated and saved to 'synthetic_sweat_data.csv'")

