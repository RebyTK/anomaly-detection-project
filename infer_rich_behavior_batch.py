import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Load model, scaler, and threshold
autoencoder = tf.keras.models.load_model("rich_behavior_model.h5", compile=False)
scaler = joblib.load("rich_scaler.pkl")

with open("rich_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# Example batch of new data (simulate real-time user behavior)
samples = pd.read_csv("rich_user_behavior.csv").drop("label", axis=1)

# Scale input
X_new = scaler.transform(samples)

# Predict reconstruction error
recon = autoencoder.predict(X_new)
errors = np.mean((X_new - recon) ** 2, axis=1)

# Add predictions back to dataframe
samples["reconstruction_error"] = errors
samples["anomaly_detected"] = (errors > threshold).astype(int)

# Output results
print("\n=== Batch Inference Results ===")
for i, row in samples.iterrows():
    print(f"Sample {i+1}: {row.to_dict()}")
print("\nThreshold:", threshold)
