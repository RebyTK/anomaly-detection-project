import numpy as np
import pandas as pd
import random
import os
import json
import time

# -----------------------
# Event Simulation Helpers
# -----------------------

def simulate_touch_session(n_touches=10, anomaly_rate=0.1):
    events = []
    for i in range(n_touches):
        x = random.randint(0, 1080)
        y = random.randint(0, 2400)
        pressure = np.random.normal(0.5, 0.1)

        # Inject anomaly
        if random.random() < anomaly_rate:
            pressure = np.random.choice([0.05, 1.5])

        events.append({
            "event_type": "touch",
            "x": x,
            "y": y,
            "pressure": round(pressure, 3),
            "timestamp": int(time.time() * 1000) + i * 50
        })
    return events

def simulate_typing_session(n_keystrokes=10, anomaly_rate=0.1):
    events = []
    last_ts = int(time.time() * 1000)
    for i in range(n_keystrokes):
        key = random.choice("abcdefghijklmnopqrstuvwxyz ")
        ts = last_ts + int(np.random.normal(250, 40))

        # Inject anomaly: extremely long pause
        if random.random() < anomaly_rate:
            ts += random.randint(2000, 5000)

        events.append({
            "event_type": "keydown",
            "key": key,
            "timestamp": ts
        })
        events.append({
            "event_type": "keyup",
            "key": key,
            "timestamp": ts + random.randint(50, 150)
        })
        last_ts = ts
    return events

def simulate_mixed_session(n_touches=5, n_typing=5, anomaly_rate=0.1):
    session = []
    session += simulate_touch_session(n_touches, anomaly_rate)
    session += simulate_typing_session(n_typing, anomaly_rate)
    return sorted(session, key=lambda e: e["timestamp"])

# -----------------------
# Dataset Generator
# -----------------------

def generate_dataset(n_samples=2000, anomaly_rate=0.05, save_dir="synthetic_rich_behavior"):
    os.makedirs(save_dir, exist_ok=True)

    # High-level features (like your original code)
    typing_speed = np.random.normal(250, 40, size=n_samples)  # ms
    app_usage_time = np.random.normal(300, 100, size=n_samples)  # sec
    touch_pressure = np.random.normal(0.5, 0.1, size=n_samples)  # 0–1
    swipe_length = np.random.normal(150, 30, size=n_samples)  # px
    accel_variance = np.random.normal(0.2, 0.05, size=n_samples)
    location_pattern = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])  # 0=normal,1=weird

    labels = np.random.choice([0, 1], size=n_samples, p=[1-anomaly_rate, anomaly_rate])

    df = pd.DataFrame({
        "typing_speed": typing_speed,
        "app_usage_time": app_usage_time,
        "touch_pressure": touch_pressure,
        "swipe_length": swipe_length,
        "accel_variance": accel_variance,
        "location_pattern": location_pattern,
        "label": labels
    })

    # Save flat CSV
    df.to_csv(os.path.join(save_dir, "rich_user_behavior.csv"), index=False)

    # Generate per-session JSON logs
    for i in range(n_samples):
        session = simulate_mixed_session(
            n_touches=random.randint(3, 7),
            n_typing=random.randint(5, 12),
            anomaly_rate=anomaly_rate
        )
        with open(os.path.join(save_dir, f"session_{i:04d}.json"), "w") as f:
            json.dump(session, f, indent=2)

    print(f"✅ Generated {n_samples} samples with {anomaly_rate*100:.1f}% anomaly rate")
    print(f"   → CSV: {save_dir}/rich_user_behavior.csv")
    print(f"   → JSON session logs: {save_dir}/session_xxxx.json")

# -----------------------
# Run Example
# -----------------------
if __name__ == "__main__":
    generate_dataset(n_samples=500, anomaly_rate=0.1)
