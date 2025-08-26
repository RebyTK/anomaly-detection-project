import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import json
import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# =========================
# Load dataset
# =========================
data = pd.read_csv("user_behavior.csv")   # <-- basic dataset file

# Drop label column
X = data.drop("label", axis=1).values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/validation split
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

# =========================
# Autoencoder Model
# =========================
input_dim = X_train.shape[1]

encoder = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
])

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(input_dim, activation="linear"),
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer="adam", loss="mse")

# =========================
# Train
# =========================
history = autoencoder.fit(
    X_train, X_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_val, X_val),
    verbose=1
)

# =========================
# Save Model & Scaler
# =========================
autoencoder.save("behavior_model.h5")
joblib.dump(scaler, "scaler.pkl")

# =========================
# Compute Threshold
# =========================
recon = autoencoder.predict(X_val, verbose=0)
errors = np.mean((X_val - recon) ** 2, axis=1)
threshold = float(np.mean(errors) + 2 * np.std(errors))

with open("threshold.json", "w") as f:
    json.dump({"threshold": threshold}, f)

print("\n✅ Basic Training complete.")
print(f"Threshold set at: {threshold:.6f}")
