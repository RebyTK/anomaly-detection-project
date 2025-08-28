"""
Simple, import-friendly version of the federated client
This version avoids any syntax issues and can be safely imported
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
import os
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class SimpleFederatedClient:
    """Simplified federated client for training and inference"""
    
    def __init__(self, client_id: str, data_path: str = None):
        self.client_id = client_id
        self.data_path = data_path
        self.local_model = None
        self.local_data = None
        self.scaler = None
        self.feature_columns = self._get_feature_columns()
        
    def _get_feature_columns(self):
        """Get the standard feature columns (24 total)"""
        base_features = [
            'typing_speed', 'typing_error_rate', 'touch_pressure',
            'swipe_length', 'accel_variance', 'session_duration',
            'interactions_per_minute', 'scroll_velocity', 'scroll_frequency',
            'battery_level', 'device_temperature', 'app_response_time',
            'network_latency', 'typing_speed_deviation', 'touch_pressure_deviation',
            'time_since_last', 'hour', 'day_of_week', 'night_usage_flag',
            'is_weekend', 'location_pattern'
        ]
        interaction_features = ['typing_efficiency', 'usage_intensity', 'device_stress']
        return base_features + interaction_features
    
    def create_sample_data(self, n_samples: int = 1000):
        """Create sample behavioral data"""
        print(f"Creating sample data for {self.client_id}...")
        
        # Use client_id hash for consistent but different data per client
        np.random.seed(hash(self.client_id) % 2147483647)
        
        data = []
        for i in range(n_samples):
            sample = {
                'typing_speed': max(50, np.random.normal(250, 40)),
                'typing_error_rate': max(0.01, np.random.normal(0.05, 0.02)),
                'touch_pressure': np.clip(np.random.normal(0.5, 0.15), 0.1, 1.0),
                'swipe_length': max(50, np.random.normal(150, 30)),
                'accel_variance': max(0.05, np.random.normal(0.2, 0.05)),
                'session_duration': max(30, np.random.normal(300, 60)),
                'interactions_per_minute': max(1, np.random.normal(15, 3)),
                'scroll_velocity': max(10, np.random.normal(200, 30)),
                'scroll_frequency': max(1, np.random.poisson(25)),
                'battery_level': np.random.uniform(0.2, 1.0),
                'device_temperature': max(20, np.random.normal(35, 3)),
                'app_response_time': max(10, np.random.normal(150, 25)),
                'network_latency': max(10, np.random.normal(100, 20)),
                'hour': np.random.randint(0, 24),
                'day_of_week': np.random.randint(0, 7),
                'night_usage_flag': np.random.choice([0, 1], p=[0.8, 0.2]),
                'is_weekend': np.random.choice([0, 1], p=[0.7, 0.3]),
                'location_pattern': np.random.choice([0, 1], p=[0.95, 0.05]),
                'typing_speed_deviation': abs(np.random.normal(0, 15)),
                'touch_pressure_deviation': abs(np.random.normal(0, 0.03)),
                'time_since_last': max(300, np.random.exponential(3600))
            }
            
            # Add label (3% anomaly rate)
            sample['label'] = np.random.choice([0, 1], p=[0.97, 0.03])
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Add interaction features
        df['typing_efficiency'] = df['typing_speed'] / (1 + df['typing_error_rate'])
        df['usage_intensity'] = df['session_duration'] * df['interactions_per_minute']
        df['device_stress'] = df['device_temperature'] * df['battery_level']
        
        self.local_data = df
        print(f"✅ Created {len(df)} samples, anomaly rate: {df['label'].mean():.3f}")
        return True
    
    def load_local_data(self):
        """Load local data or create sample data"""
        print(f"📂 Loading data for {self.client_id}...")
        
        if self.data_path and os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                self.local_data = self._ensure_features(df)
                print(f"✅ Loaded {len(self.local_data)} samples from {self.data_path}")
                return True
            except Exception as e:
                print(f"❌ Error loading data: {e}")
        
        print("Creating sample data...")
        return self.create_sample_data()
    
    def _ensure_features(self, df):
        """Ensure all required features exist in the dataframe"""
        # Default values for missing features
        defaults = {
            'typing_speed': 250.0, 'typing_error_rate': 0.05,
            'touch_pressure': 0.5, 'swipe_length': 150.0,
            'accel_variance': 0.2, 'session_duration': 300.0,
            'interactions_per_minute': 15.0, 'scroll_velocity': 200.0,
            'scroll_frequency': 25.0, 'battery_level': 0.7,
            'device_temperature': 35.0, 'app_response_time': 150.0,
            'network_latency': 100.0, 'typing_speed_deviation': 20.0,
            'touch_pressure_deviation': 0.05, 'time_since_last': 3600.0,
            'hour': 12.0, 'day_of_week': 2.0, 'night_usage_flag': 0.0,
            'is_weekend': 0.0, 'location_pattern': 0.0, 'label': 0
        }
        
        # Add missing base features
        for feature, default_value in defaults.items():
            if feature not in df.columns:
                df[feature] = default_value
        
        # Calculate interaction features
        df['typing_efficiency'] = df['typing_speed'] / (1 + df['typing_error_rate'])
        df['usage_intensity'] = df['session_duration'] * df['interactions_per_minute']
        df['device_stress'] = df['device_temperature'] * df['battery_level']
        
        # Clean up any invalid values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(defaults.get(col, 0))
            df[col] = df[col].replace([np.inf, -np.inf], defaults.get(col, 0))
        
        return df
    
    def build_local_model(self, input_dim: int = 24):
        """Build local autoencoder model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(input_dim, activation='linear')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_training_data(self):
        """Prepare data for training"""
        if self.local_data is None:
            return None, None
        
        # Extract feature columns
        X = self.local_data[self.feature_columns].values
        
        # Handle invalid values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X_scaled  # Autoencoder: input = output
    
    def train_local_model(self, global_weights: Optional[List] = None, epochs: int = 10):
        """Train the local model"""
        print(f"🏋️  Training {self.client_id}...")
        
        # Prepare data
        X_train, y_train = self.prepare_training_data()
        if X_train is None:
            print(f"❌ No training data for {self.client_id}")
            return None
        
        # Build model
        self.local_model = self.build_local_model(X_train.shape[1])
        
        # Set global weights if provided
        if global_weights is not None:
            try:
                self.local_model.set_weights(global_weights)
            except Exception as e:
                print(f"  ⚠️  Could not set global weights: {e}")
        
        # Train model
        try:
            history = self.local_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=min(32, max(1, len(X_train) // 4)),
                verbose=0,
                validation_split=0.2 if len(X_train) > 20 else 0.0
            )
            
            final_loss = history.history['loss'][-1]
            print(f"  📊 Final loss: {final_loss:.4f}")
            
            return self.local_model.get_weights()
            
        except Exception as e:
            print(f"  ❌ Training error: {e}")
            return None
    
    def save_local_model(self, save_dir: str = "local_models"):
        """Save the trained model"""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Save model
            model_path = os.path.join(save_dir, f"{self.client_id}_model.h5")
            self.local_model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(save_dir, f"{self.client_id}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            print(f"💾 Saved {self.client_id} model and scaler")
            return True
            
        except Exception as e:
            print(f"❌ Save error for {self.client_id}: {e}")
            return False


def test_simple_client():
    """Test the simple client"""
    print("🧪 Testing Simple Federated Client")
    
    client = SimpleFederatedClient("test_client")
    
    if client.load_local_data():
        weights = client.train_local_model(epochs=3)
        if weights is not None:
            client.save_local_model()
            print("✅ Client test successful!")
            return client
    
    print("❌ Client test failed!")
    return None


if __name__ == "__main__":
    test_simple_client()
