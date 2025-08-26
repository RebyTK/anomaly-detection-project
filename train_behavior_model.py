import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("user_behavior.csv")
X = df[["typing_ms", "app_opens"]].values

# Normalize
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Use only "normal" data for training
X_train = X_scaled[df["label"] == 0]

# Autoencoder
input_dim = X_train.shape[1]
autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(4, activation="relu"),
    layers.Dense(input_dim, activation="sigmoid")
])

autoencoder.compile(optimizer="adam", loss="mse")
autoencoder.fit(X_train, X_train, epochs=30, batch_size=16, verbose=1)

# Compute reconstruction error for all data
recons = autoencoder.predict(X_scaled)
errors = np.mean((X_scaled - recons) ** 2, axis=1)

# Add anomaly score
df["anomaly_score"] = errors

print(df.head())

# Save model + scaler
autoencoder.save("behavior_model.h5")
import joblib
joblib.dump(scaler, "scaler.pkl")

# Visualize anomalies
plt.scatter(df["typing_ms"], df["app_opens"], c=df["anomaly_score"], cmap="coolwarm")
plt.xlabel("Typing Rhythm (ms)")
plt.ylabel("App Opens")
plt.title("User Behavior with Anomaly Scores")
plt.colorbar(label="Anomaly Score")
plt.show()
