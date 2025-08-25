import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os

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
# Load dataset for batch inference
# =========================
data = pd.read_csv("rich_user_behavior.csv")
X = data.drop("label", axis=1).values

# Scale input
X_scaled = scaler.transform(X)

# =========================
# Predict reconstruction errors
# =========================
recon = autoencoder.predict(X_scaled, verbose=0)
errors = np.mean((X_scaled - recon) ** 2, axis=1)

# Add results to dataframe
data["reconstruction_error"] = errors
data["anomaly_detected"] = (errors > threshold).astype(int)

# =========================
# Summary Statistics
# =========================
total_samples = len(data)
total_anomalies = data["anomaly_detected"].sum()
print("\n=== Batch Inference Results ===")
print(f"Total samples: {total_samples}")
print(f"Detected anomalies: {total_anomalies}")
print(f"Threshold: {threshold:.6f}\n")

# Show some flagged samples
print("🔎 Example Anomalies:")
print(data[data["anomaly_detected"] == 1].head())

# Save results
data.to_csv("batch_inference_results.csv", index=False)
print("\n✅ Batch results saved to batch_inference_results.csv")
