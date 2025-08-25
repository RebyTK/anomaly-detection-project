import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
import time

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# Load model, scaler, and threshold
# =========================
autoencoder = tf.keras.models.load_model("rich_behavior_model.h5", compile=False)
scaler = joblib.load("rich_scaler.pkl")

with open("rich_threshold.json", "r") as f:
    threshold = json.load(f)["threshold"]

# =========================
# Load sample dataset (simulate real-time stream)
# =========================
data = pd.read_csv("rich_user_behavior.csv").drop("label", axis=1)

# =========================
# Real-time Simulation
# =========================
print("\n=== Real-time Anomaly Detection Simulation ===")

for i, sample in data.iterrows():
    X_sample = sample.values.reshape(1, -1)
    X_scaled = scaler.transform(X_sample)

    # Reconstruct
    recon = autoencoder.predict(X_scaled, verbose=0)
    error = np.mean((X_scaled - recon) ** 2)

    anomaly = int(error > threshold)

    print(f"[Sample {i+1}] Error={error:.6f} | Anomaly={anomaly}")

    # Simulate streaming delay (e.g., 0.5 sec)
    time.sleep(0.2)

print("\n✅ Real-time detection complete!")
