import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import json

# Import your generator
from data_generator import generate_user_behavior

# ====== 1. Load or Generate Data ======
try:
    # If real dataset exists, use it
    df = pd.read_csv("rich_user_behavior.csv")
    print("✅ Loaded real dataset: rich_user_behavior.csv")
except FileNotFoundError:
    # Otherwise generate synthetic data
    print("⚠️ No real dataset found. Generating synthetic dataset...")
    df = generate_user_behavior(1000)  # generate 1000 samples
    df.to_csv("rich_user_behavior.csv", index=False)
    print("✅ Saved synthetic dataset to rich_user_behavior.csv")

# Drop label (only used for validation, not training)
X = df.drop("label", axis=1).values

# ====== 2. Scale Data ======
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ====== 3. Build Autoencoder ======
input_dim = X_scaled.shape[1]
encoding_dim = 2

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation="relu")(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation="sigmoid")(encoded)

autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer="adam", loss="mse")

# ====== 4. Train Model ======
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ====== 5. Save Model & Scaler ======
autoencoder.save("rich_behavior_model.h5")
joblib.dump(scaler, "rich_scaler.pkl")

# ====== 6. Compute & Save Threshold ======
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse, 95)  # 95th percentile as anomaly cutoff

with open("rich_threshold.json", "w") as f:
    json.dump({"threshold": float(threshold)}, f)

print("✅ Training complete. Model, scaler, and threshold saved.")


