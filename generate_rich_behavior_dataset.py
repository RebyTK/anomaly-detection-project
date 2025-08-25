import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# Number of samples to simulate
N_SAMPLES = 2000

# --- Typing behavior ---
typing_speed = np.random.normal(loc=250, scale=40, size=N_SAMPLES)  # ms per key
typing_error_rate = np.random.uniform(0, 0.1, size=N_SAMPLES)       # errors per keystroke

# --- App usage ---
app_usage_time = np.random.normal(loc=300, scale=100, size=N_SAMPLES)  # seconds/session
touch_pressure = np.random.normal(loc=0.5, scale=0.1, size=N_SAMPLES)  # normalized 0–1

# --- Swipe behavior ---
swipe_length = np.random.normal(loc=150, scale=30, size=N_SAMPLES)     # pixels
swipe_angle = np.random.uniform(-np.pi, np.pi, size=N_SAMPLES)         # radians (-π to π)

# --- Sensor behavior ---
accel_variance = np.random.normal(loc=0.2, scale=0.05, size=N_SAMPLES)

# --- Location pattern (0 = normal, 1 = unusual location) ---
location_pattern = np.random.choice([0, 1], size=N_SAMPLES, p=[0.9, 0.1])

# --- Usage time (hour + night flag) ---
timestamps = [datetime.now() - timedelta(minutes=random.randint(0, 10000)) for _ in range(N_SAMPLES)]
usage_hours = [t.hour for t in timestamps]
night_usage_flag = [1 if 1 <= t.hour <= 5 else 0 for t in timestamps]

# --- Labels (mostly normal, few anomalies) ---
labels = np.random.choice([0, 1], size=N_SAMPLES, p=[0.95, 0.05])

# --- Build dataframe ---
data = pd.DataFrame({
    "typing_speed": typing_speed,
    "typing_error_rate": typing_error_rate,
    "app_usage_time": app_usage_time,
    "touch_pressure": touch_pressure,
    "swipe_length": swipe_length,
    "swipe_angle": swipe_angle,
    "accel_variance": accel_variance,
    "location_pattern": location_pattern,
    "usage_hour": usage_hours,
    "night_usage_flag": night_usage_flag,
    "label": labels
})

# Save dataset
data.to_csv("rich_user_behavior.csv", index=False)
print("✅ Dataset with extended features saved to rich_user_behavior.csv")
