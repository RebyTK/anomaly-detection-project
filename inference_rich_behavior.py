import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json

# Load model, scaler, and threshold
autoencoder = tf.keras.models.load_model("rich_behavior_model.h5", compile=False)
scaler = joblib.load("rich_scaler.pkl")

with open("rich_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# Example new data (simulate real-time user behavior)
sample = pd.DataFrame([{
    "typing_ms": 650,      # typing speed in ms
    "app_opens": 6,        # app opens per hour
    "touch_pressure": 0.55, # touch strength
    "movement_var": 1.0    # movement variance
}])

# Scale input
X_new = scaler.transform(sample)

# Predict reconstruction error
recon = autoencoder.predict(X_new)
error = np.mean((X_new - recon) ** 2)

# Output results
print("Input sample:", sample.to_dict(orient="records")[0])
print("Reconstruction Error:", error)
print("Threshold:", threshold)
print("Anomaly Detected:", error > threshold)
