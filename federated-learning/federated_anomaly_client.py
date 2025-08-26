import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class FederatedAnomalyClient:
    def __init__(self, client_id: str, data_path: str):
        self.client_id = client_id
        self.data_path = data_path
        self.local_model = None
        self.local_data = None
        self.scaler = None
        self.training_history = []
        
    def load_local_data(self):
        """Load and prepare local client data"""
        print(f"📂 Loading data for {self.client_id}...")
        
        try:
            # Load client-specific data
            self.local_data = pd.read_csv(self.data_path)
            
            # Prepare features (same as main anomaly detection system)
            feature_cols = [
                'typing_speed', 'typing_error_rate', 'touch_pressure',
                'swipe_length', 'accel_variance', 'session_duration',
                'interactions_per_minute', 'scroll_velocity', 'scroll_frequency',
                'battery_level', 'device_temperature', 'app_response_time',
                'network_latency', 'typing_speed_deviation', 'touch_pressure_deviation',
                'time_since_last', 'hour', 'day_of_week', 'night_usage_flag',
                'is_weekend', 'location_pattern'
            ]
            
            # Add interaction features
            self.local_data['typing_efficiency'] = self.local_data['typing_speed'] / (1 + self.local_data['typing_error_rate'])
            self.local_data['usage_intensity'] = self.local_data['session_duration'] * self.local_data['interactions_per_minute']
            self.local_data['device_stress'] = self.local_data['device_temperature'] * self.local_data['battery_level']
            
            interaction_features = ['typing_efficiency', 'usage_intensity', 'device_stress']
            all_features = feature_cols + interaction_features
            
            # Ensure all features exist
            for feature in all_features:
                if feature not in self.local_data.columns:
                    self.local_data[feature] = 0  # Default value
            
            print(f"✅ Loaded {len(self.local_data)} samples for {self.client_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data for {self.client_id}: {e}")
            return False
    
    def build_local_model(self, input_dim: int = 24):
        """Build local client model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation='relu', name='dense_1'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu', name='dense_2'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu', name='dense_3'),
            tf.keras.layers.Dense(32, activation='relu', name='dense_4'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu', name='dense_5'),
            tf.keras.layers.Dense(input_dim, activation='linear', name='output')
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()], run_eagerly=False)
        return model
    
    def prepare_training_data(self):
        """Prepare training data for local model"""
        if self.local_data is None:
            return None, None
        
        # Select features (exclude label, client_id, and non-numeric columns)
        exclude_cols = ['label', 'client_id']
        # Also exclude any timestamp or date columns
        for col in self.local_data.columns:
            if 'time' in col.lower() or 'date' in col.lower() or self.local_data[col].dtype == 'object':
                exclude_cols.append(col)
        
        feature_cols = [col for col in self.local_data.columns if col not in exclude_cols]
        
        X = self.local_data[feature_cols].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X_scaled  # Autoencoder: input = output
    
    def train_local_model(self, global_weights: Optional[List] = None, epochs: int = 5):
        """Train local model with optional global weights initialization"""
        print(f"🏋️  Training local model for {self.client_id}...")
        
        # Prepare data
        X_train, y_train = self.prepare_training_data()
        if X_train is None:
            return None
        
        # Build local model
        input_dim = X_train.shape[1]
        print(f"📊 Building local model with {input_dim} input features")
        self.local_model = self.build_local_model(input_dim)
        
        # Initialize with global weights if provided
        if global_weights is not None:
            self.local_model.set_weights(global_weights)
            print(f"  📥 Initialized with global weights")
        
        # Train local model
        history = self.local_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            verbose=0,
            validation_split=0.2
        )
        
        # Store training history
        self.training_history.append({
            'client_id': self.client_id,
            'epochs': epochs,
            'final_loss': history.history['loss'][-1],
            'final_mae': history.history['mae'][-1],
            'val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
        })
        
        print(f"  📊 Final Loss: {history.history['loss'][-1]:.4f}")
        print(f"  📊 Final MAE: {history.history['mae'][-1]:.4f}")
        
        return self.local_model.get_weights()
    
    def evaluate_local_model(self, test_data_path: Optional[str] = None):
        """Evaluate local model performance"""
        print(f"🔍 Evaluating local model for {self.client_id}...")
        
        # Use local data for evaluation if no test data provided
        if test_data_path is None:
            X_test, y_test = self.prepare_training_data()
        else:
            # Load external test data
            test_df = pd.read_csv(test_data_path)
            exclude_cols = ['label', 'client_id']
            # Also exclude any timestamp or date columns
            for col in test_df.columns:
                if 'time' in col.lower() or 'date' in col.lower() or test_df[col].dtype == 'object':
                    exclude_cols.append(col)
            
            feature_cols = [col for col in test_df.columns if col not in exclude_cols]
            X_test = test_df[feature_cols].values
            # Create a new scaler for test data if needed
            test_scaler = StandardScaler()
            X_test = test_scaler.fit_transform(X_test)
            y_test = X_test  # Autoencoder
        
        # Evaluate model
        test_loss, test_mae = self.local_model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate reconstruction errors for anomaly detection
        reconstructions = self.local_model.predict(X_test, verbose=0)
        reconstruction_errors = np.mean((X_test - reconstructions) ** 2, axis=1)
        
        # Calculate threshold
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        
        print(f"  📊 Test Loss: {test_loss:.4f}")
        print(f"  📊 Test MAE: {test_mae:.4f}")
        print(f"  📊 Anomaly Threshold: {threshold:.4f}")
        
        return {
            'test_loss': test_loss,
            'test_mae': test_mae,
            'threshold': threshold,
            'reconstruction_errors': reconstruction_errors.tolist()
        }
    
    def save_local_model(self, save_dir: str = "local_models"):
        """Save local model and components"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f"{self.client_id}_model.h5")
        self.local_model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(save_dir, f"{self.client_id}_scaler.pkl")
        joblib.dump(self.scaler, scaler_path)
        
        # Save training history
        history_path = os.path.join(save_dir, f"{self.client_id}_history.json")
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"💾 Saved local model for {self.client_id}")
        print(f"  - {model_path}")
        print(f"  - {scaler_path}")
        print(f"  - {history_path}")
    
    def load_local_model(self, model_path: str, scaler_path: str):
        """Load previously saved local model"""
        try:
            self.local_model = tf.keras.models.load_model(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✅ Loaded local model for {self.client_id}")
            return True
        except Exception as e:
            print(f"❌ Error loading local model for {self.client_id}: {e}")
            return False
    
    def detect_local_anomalies(self, sample_data: Dict) -> Dict:
        """Detect anomalies using local model"""
        if self.local_model is None or self.scaler is None:
            return {'error': 'Local model not trained or loaded'}
        
        # Prepare features (same as main system)
        feature_cols = [
            'typing_speed', 'typing_error_rate', 'touch_pressure',
            'swipe_length', 'accel_variance', 'session_duration',
            'interactions_per_minute', 'scroll_velocity', 'scroll_frequency',
            'battery_level', 'device_temperature', 'app_response_time',
            'network_latency', 'typing_speed_deviation', 'touch_pressure_deviation',
            'time_since_last', 'hour', 'day_of_week', 'night_usage_flag',
            'is_weekend', 'location_pattern'
        ]
        
        # Extract features with defaults
        features = []
        for feature in feature_cols:
            features.append(sample_data.get(feature, 0))
        
        # Add interaction features
        typing_efficiency = sample_data.get('typing_speed', 250) / (1 + sample_data.get('typing_error_rate', 0.05))
        usage_intensity = sample_data.get('session_duration', 300) * sample_data.get('interactions_per_minute', 15)
        device_stress = sample_data.get('device_temperature', 35) * sample_data.get('battery_level', 0.7)
        
        features.extend([typing_efficiency, usage_intensity, device_stress])
        
        # Scale features
        X = np.array(features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get reconstruction
        reconstruction = self.local_model.predict(X_scaled, verbose=0)
        reconstruction_error = np.mean((X_scaled - reconstruction) ** 2)
        
        # Calculate threshold
        threshold = np.mean(self.local_model.predict(X_scaled, verbose=0)) + 2 * np.std(self.local_model.predict(X_scaled, verbose=0))
        
        # Determine if anomaly
        is_anomaly = reconstruction_error > threshold
        
        return {
            'client_id': self.client_id,
            'reconstruction_error': float(reconstruction_error),
            'threshold': float(threshold),
            'is_anomaly': bool(is_anomaly),
            'confidence': min(1.0, reconstruction_error / threshold) if threshold > 0 else 0.0
        }

def create_federated_clients(num_clients: int = 3) -> List[FederatedAnomalyClient]:
    """Create federated learning clients"""
    clients = []
    
    for i in range(num_clients):
        client_id = f"client_{i+1}"
        data_path = f"client_data/{client_id}_data.csv"
        
        client = FederatedAnomalyClient(client_id, data_path)
        if client.load_local_data():
            clients.append(client)
    
    return clients

def main():
    """Main function to test federated client"""
    print("🏠 Federated Anomaly Detection Client")
    print("=" * 40)
    
    # Create client
    client = FederatedAnomalyClient("test_client", "client_data/client_1_data.csv")
    
    if client.load_local_data():
        # Train local model
        weights = client.train_local_model(epochs=3)
        
        # Evaluate model
        evaluation = client.evaluate_local_model()
        
        # Save model
        client.save_local_model()
        
        # Test anomaly detection
        test_sample = {
            'typing_speed': 280,
            'typing_error_rate': 0.08,
            'touch_pressure': 0.6,
            'swipe_length': 160,
            'accel_variance': 0.3,
            'session_duration': 420,
            'interactions_per_minute': 18,
            'scroll_velocity': 220,
            'scroll_frequency': 30,
            'battery_level': 0.4,
            'device_temperature': 38,
            'app_response_time': 180,
            'network_latency': 120,
            'hour': 14,
            'day_of_week': 2,
            'night_usage_flag': 0,
            'is_weekend': 0,
            'location_pattern': 0
        }
        
        result = client.detect_local_anomalies(test_sample)
        print(f"\n🔍 Anomaly Detection Result:")
        print(f"  Anomaly: {result['is_anomaly']}")
        print(f"  Error: {result['reconstruction_error']:.4f}")
        print(f"  Confidence: {result['confidence']:.4f}")

if __name__ == "__main__":
    main()
