import numpy as np
import pandas as pd
import random

# Number of samples
n_samples = 2000

# Simulated features
typing_speed = np.random.normal(loc=250, scale=40, size=n_samples)  # ms between keystrokes
app_usage_time = np.random.normal(loc=300, scale=100, size=n_samples)  # seconds per app session
touch_pressure = np.random.normal(loc=0.5, scale=0.1, size=n_samples)  # 0–1 scale
swipe_length = np.random.normal(loc=150, scale=30, size=n_samples)  # pixels
accel_variance = np.random.normal(loc=0.2, scale=0.05, size=n_samples)  # motion variance
location_pattern = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  
# 0 = normal (home/work routine), 1 = unusual location

# Labels (0 = normal, 1 = anomaly)
# Inject anomalies randomly
labels = np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])

# Combine into dataframe
data = pd.DataFrame({
    "typing_speed": typing_speed,
    "app_usage_time": app_usage_time,
    "touch_pressure": touch_pressure,
    "swipe_length": swipe_length,
    "accel_variance": accel_variance,
    "location_pattern": location_pattern,
    "label": labels
})

# Save to CSV
data.to_csv("rich_user_behavior.csv", index=False)

print("✅ Rich user behavior dataset generated and saved to rich_user_behavior.csv")

