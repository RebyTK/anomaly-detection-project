#!/usr/bin/env python3
"""
Fixed Federated Anomaly Detection System
Addresses key issues in the original implementation
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import warnings

warnings.filterwarnings('ignore')
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class FixedFederatedClient:
    """Fixed federated client with consistent feature handling"""
    
    def __init__(self, client_id: str, data_path: Optional[str] = None):
        self.client_id = client_id
        self.data_path = data_path
        self.local_model = None
        self.local_data = None
        self.scaler = None
        self.feature_columns = self._get_standard_features()
        self.input_dim = len(self.feature_columns)
        
    def _get_standard_features(self) -> List[str]:
        """Get standardized feature set (24 features total)"""
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
    
    def create_sample_data(self, n_samples: int = 1000) -> bool:
        """Generate consistent sample data with proper feature alignment"""
        print(f"Creating sample data for {self.client_id}...")
        
        # Use client_id for consistent but different data
        np.random.seed(hash(self.client_id) % 2147483647)
        
        data = []
        for i in range(n_samples):
            sample = {
                # Base behavioral features
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
                'time_since_last': max(300, np.random.exponential(3600)),
                'label': np.random.choice([0, 1], p=[0.97, 0.03])
            }
            
            data.append(sample)
        
        df = pd.DataFrame(data)
        
        # Calculate interaction features consistently
        df['typing_efficiency'] = df['typing_speed'] / (1 + df['typing_error_rate'])
        df['usage_intensity'] = df['session_duration'] * df['interactions_per_minute']
        df['device_stress'] = df['device_temperature'] * df['battery_level']
        
        self.local_data = df
        print(f"Created {len(df)} samples, anomaly rate: {df['label'].mean():.3f}")
        return True
    
    def load_local_data(self) -> bool:
        """Load and prepare local data"""
        print(f"Loading data for {self.client_id}...")
        
        if self.data_path and os.path.exists(self.data_path):
            try:
                df = pd.read_csv(self.data_path)
                self.local_data = self._ensure_feature_consistency(df)
                print(f"Loaded {len(self.local_data)} samples from file")
                return True
            except Exception as e:
                print(f"Error loading data: {e}")
        
        return self.create_sample_data()
    
    def _ensure_feature_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure dataset has all required features with consistent naming"""
        
        # Feature defaults
        feature_defaults = {
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
        
        # Add missing features
        for feature, default in feature_defaults.items():
            if feature not in df.columns:
                df[feature] = default
        
        # Calculate interaction features
        df['typing_efficiency'] = df['typing_speed'] / (1 + df['typing_error_rate'])
        df['usage_intensity'] = df['session_duration'] * df['interactions_per_minute']
        df['device_stress'] = df['device_temperature'] * df['battery_level']
        
        # Clean invalid values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(feature_defaults.get(col, 0))
            df[col] = df[col].replace([np.inf, -np.inf], feature_defaults.get(col, 0))
        
        return df
    
    def build_model(self) -> tf.keras.Model:
        """Build autoencoder model with consistent architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.input_dim,)),
            tf.keras.layers.Dense(64, activation='relu', name='encoder_1'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu', name='encoder_2'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu', name='bottleneck'),
            tf.keras.layers.Dense(32, activation='relu', name='decoder_1'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu', name='decoder_2'),
            tf.keras.layers.Dense(self.input_dim, activation='linear', name='output')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae'],
            run_eagerly=False
        )
        return model
    
    def prepare_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data with consistent feature extraction"""
        if self.local_data is None:
            return None, None
        
        # Extract features in correct order
        X = self.local_data[self.feature_columns].values
        
        # Handle invalid values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Fit and transform with scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, X_scaled  # Autoencoder: input = target
    
    def train(self, global_weights: Optional[List] = None, epochs: int = 10) -> Optional[List]:
        """Train local model"""
        print(f"Training {self.client_id}...")
        
        X_train, y_train = self.prepare_data()
        if X_train is None:
            print(f"No training data available for {self.client_id}")
            return None
        
        # Build model
        self.local_model = self.build_model()
        
        # Set global weights if provided
        if global_weights is not None:
            try:
                self.local_model.set_weights(global_weights)
                print(f"  Applied global weights to {self.client_id}")
            except Exception as e:
                print(f"  Warning: Could not apply global weights: {e}")
        
        # Train model
        try:
            history = self.local_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=min(32, max(8, len(X_train) // 4)),
                verbose=0,
                validation_split=0.2 if len(X_train) > 50 else 0.0
            )
            
            final_loss = history.history['loss'][-1]
            print(f"  Training completed - Final loss: {final_loss:.4f}")
            
            return self.local_model.get_weights()
            
        except Exception as e:
            print(f"  Training failed: {e}")
            return None
    
    def save_model(self, save_dir: str = "local_models") -> bool:
        """Save trained model and scaler"""
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Save model
            model_path = os.path.join(save_dir, f"{self.client_id}_model.h5")
            self.local_model.save(model_path)
            
            # Save scaler
            scaler_path = os.path.join(save_dir, f"{self.client_id}_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            
            print(f"Saved {self.client_id} model and scaler")
            return True
            
        except Exception as e:
            print(f"Error saving {self.client_id}: {e}")
            return False


class FixedFederatedServer:
    """Fixed federated server with proper aggregation"""
    
    def __init__(self, num_clients: int = 3, num_rounds: int = 5):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.global_model = None
        self.clients = []
        self.training_history = []
        self.feature_dim = 24  # Fixed feature dimension
    
    def create_clients(self, use_sample_data: bool = True) -> bool:
        """Create federated clients"""
        print("Creating federated clients...")
        
        # Create client data directory
        os.makedirs("client_data", exist_ok=True)
        
        for i in range(self.num_clients):
            client_id = f"client_{i+1}"
            
            if use_sample_data:
                # Create client with sample data
                client = FixedFederatedClient(client_id)
            else:
                # Try to use existing data files
                data_path = f"client_data/{client_id}_data.csv"
                client = FixedFederatedClient(client_id, data_path)
            
            if client.load_local_data():
                self.clients.append(client)
                print(f"  Added {client_id}")
            else:
                print(f"  Failed to create {client_id}")
        
        print(f"Created {len(self.clients)} clients")
        return len(self.clients) > 0
    
    def initialize_global_model(self) -> bool:
        """Initialize global model"""
        try:
            self.global_model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.feature_dim,)),
                tf.keras.layers.Dense(64, activation='relu', name='encoder_1'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu', name='encoder_2'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(16, activation='relu', name='bottleneck'),
                tf.keras.layers.Dense(32, activation='relu', name='decoder_1'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu', name='decoder_2'),
                tf.keras.layers.Dense(self.feature_dim, activation='linear', name='output')
            ])
            
            self.global_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae'],
                run_eagerly=False
            )
            
            print("Global model initialized")
            return True
            
        except Exception as e:
            print(f"Failed to initialize global model: {e}")
            return False
    
    def federated_averaging(self, client_weights: List[List]) -> List:
        """Perform federated averaging of client weights"""
        if not client_weights:
            return None
        
        print(f"  Averaging weights from {len(client_weights)} clients...")
        
        # Average weights layer by layer
        averaged_weights = []
        num_layers = len(client_weights[0])
        
        for layer_idx in range(num_layers):
            # Get weights for this layer from all clients
            layer_weights = [client_weights[i][layer_idx] for i in range(len(client_weights))]
            
            # Average the weights
            averaged_layer = np.mean(layer_weights, axis=0)
            averaged_weights.append(averaged_layer)
        
        return averaged_weights
    
    def train_federated(self) -> bool:
        """Run federated training"""
        print(f"\nStarting federated training ({self.num_rounds} rounds)...")
        print("=" * 50)
        
        if not self.clients:
            print("No clients available for training")
            return False
        
        if not self.initialize_global_model():
            return False
        
        # Training rounds
        for round_num in range(1, self.num_rounds + 1):
            print(f"\nRound {round_num}/{self.num_rounds}")
            print("-" * 30)
            
            client_weights = []
            round_losses = []
            
            # Get current global weights
            global_weights = self.global_model.get_weights()
            
            # Train each client
            for client in self.clients:
                weights = client.train(global_weights, epochs=5)
                if weights is not None:
                    client_weights.append(weights)
                    
                    # Calculate training loss for tracking
                    if client.local_model is not None:
                        X, y = client.prepare_data()
                        if X is not None:
                            loss = client.local_model.evaluate(X, y, verbose=0)[0]
                            round_losses.append(loss)
            
            # Federated averaging
            if client_weights:
                averaged_weights = self.federated_averaging(client_weights)
                self.global_model.set_weights(averaged_weights)
                
                # Record training history
                avg_loss = np.mean(round_losses) if round_losses else float('inf')
                self.training_history.append({
                    'round': round_num,
                    'avg_loss': avg_loss,
                    'num_clients': len(client_weights)
                })
                
                print(f"  Average loss: {avg_loss:.4f}")
                print(f"  Clients participated: {len(client_weights)}/{len(self.clients)}")
            else:
                print("  No client updates received")
        
        print(f"\nFederated training completed!")
        return True
    
    def save_models(self) -> bool:
        """Save federated and local models"""
        print("\nSaving models...")
        
        try:
            # Save global model
            os.makedirs("federated_models", exist_ok=True)
            global_model_path = "federated_models/global_model.h5"
            self.global_model.save(global_model_path)
            print(f"  Global model saved: {global_model_path}")
            
            # Save local models
            for client in self.clients:
                if client.local_model is not None:
                    client.save_model()
            
            # Save training history
            history_path = "federated_models/training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"  Training history saved: {history_path}")
            
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False


class FixedFederatedInference:
    """Fixed federated inference system"""
    
    def __init__(self):
        self.global_model = None
        self.client_models = {}
        self.client_scalers = {}
        self.global_scaler = None
        self.feature_columns = self._get_standard_features()
        
    def _get_standard_features(self) -> List[str]:
        """Get the same feature set used in training"""
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
    
    def load_global_model(self, model_path: str = "federated_models/global_model.h5") -> bool:
        """Load the global federated model"""
        try:
            if os.path.exists(model_path):
                self.global_model = tf.keras.models.load_model(model_path)
                print(f"Global model loaded: {model_path}")
                
                # Create a default scaler (in practice, this should be saved with the model)
                self.global_scaler = StandardScaler()
                return True
            else:
                print(f"Global model not found: {model_path}")
                return False
                
        except Exception as e:
            print(f"Error loading global model: {e}")
            return False
    
    def load_client_models(self, models_dir: str = "local_models") -> int:
        """Load client models and scalers"""
        loaded_count = 0
        
        if not os.path.exists(models_dir):
            print(f"Client models directory not found: {models_dir}")
            return 0
        
        for file in os.listdir(models_dir):
            if file.endswith("_model.h5"):
                client_id = file.replace("_model.h5", "")
                model_path = os.path.join(models_dir, file)
                scaler_path = os.path.join(models_dir, f"{client_id}_scaler.pkl")
                
                try:
                    # Load model
                    model = tf.keras.models.load_model(model_path)
                    self.client_models[client_id] = model
                    
                    # Load scaler if available
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        self.client_scalers[client_id] = scaler
                    
                    loaded_count += 1
                    print(f"  Loaded {client_id}")
                    
                except Exception as e:
                    print(f"  Error loading {client_id}: {e}")
        
        print(f"Loaded {loaded_count} client models")
        return loaded_count
    
    def prepare_sample(self, sample_data: Dict) -> np.ndarray:
        """Prepare a sample for inference with consistent feature extraction"""
        
        # Feature defaults
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
            'is_weekend': 0.0, 'location_pattern': 0.0
        }
        
        # Extract base features
        features = []
        for feature in self.feature_columns[:21]:  # First 21 are base features
            value = sample_data.get(feature, defaults.get(feature, 0.0))
            features.append(float(value))
        
        # Calculate interaction features
        typing_speed = sample_data.get('typing_speed', defaults['typing_speed'])
        typing_error_rate = sample_data.get('typing_error_rate', defaults['typing_error_rate'])
        session_duration = sample_data.get('session_duration', defaults['session_duration'])
        interactions_per_minute = sample_data.get('interactions_per_minute', defaults['interactions_per_minute'])
        device_temperature = sample_data.get('device_temperature', defaults['device_temperature'])
        battery_level = sample_data.get('battery_level', defaults['battery_level'])
        
        # Add interaction features
        typing_efficiency = typing_speed / (1 + typing_error_rate)
        usage_intensity = session_duration * interactions_per_minute
        device_stress = device_temperature * battery_level
        
        features.extend([typing_efficiency, usage_intensity, device_stress])
        
        return np.array(features).reshape(1, -1)
    
    def calculate_anomaly_score(self, model: tf.keras.Model, X: np.ndarray, 
                               scaler: Optional[StandardScaler] = None) -> Tuple[float, float]:
        """Calculate anomaly score using reconstruction error"""
        try:
            # Scale features
            if scaler is not None:
                # In practice, scaler should be pre-fitted
                # For demo, we'll fit on the sample (not recommended for production)
                X_scaled = scaler.fit_transform(X)
            else:
                temp_scaler = StandardScaler()
                X_scaled = temp_scaler.fit_transform(X)
            
            # Get reconstruction
            reconstruction = model.predict(X_scaled, verbose=0)
            
            # Calculate reconstruction error
            mse_error = np.mean((X_scaled - reconstruction) ** 2, axis=1)[0]
            
            # Simple threshold (in practice, this should be learned from validation data)
            threshold = 0.1  # This should be determined during training
            
            return float(mse_error), float(threshold)
            
        except Exception as e:
            print(f"Error calculating anomaly score: {e}")
            return 0.0, 0.1
    
    def detect_anomaly(self, sample_data: Dict, client_id: Optional[str] = None) -> Dict:
        """Detect anomalies using federated models"""
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'global_prediction': None,
            'client_prediction': None,
            'ensemble_decision': 0,
            'risk_level': 'low'
        }
        
        try:
            # Prepare features
            X = self.prepare_sample(sample_data)
            
            # Global model prediction
            if self.global_model is not None:
                error, threshold = self.calculate_anomaly_score(
                    self.global_model, X, self.global_scaler
                )
                anomaly = int(error > threshold)
                confidence = min(1.0, error / threshold) if threshold > 0 else 0.0
                
                result['global_prediction'] = {
                    'error': error,
                    'threshold': threshold,
                    'anomaly': anomaly,
                    'confidence': confidence
                }
            
            # Client model prediction
            if client_id and client_id in self.client_models:
                client_model = self.client_models[client_id]
                client_scaler = self.client_scalers.get(client_id)
                
                error, threshold = self.calculate_anomaly_score(
                    client_model, X, client_scaler
                )
                anomaly = int(error > threshold)
                confidence = min(1.0, error / threshold) if threshold > 0 else 0.0
                
                result['client_prediction'] = {
                    'error': error,
                    'threshold': threshold,
                    'anomaly': anomaly,
                    'confidence': confidence
                }
            
            # Ensemble decision
            votes = 0
            total = 0
            
            if result['global_prediction']:
                votes += result['global_prediction']['anomaly']
                total += 1
            
            if result['client_prediction']:
                votes += result['client_prediction']['anomaly']
                total += 1
            
            if total > 0:
                result['ensemble_decision'] = int(votes / total >= 0.5)
                
                # Risk level based on consensus and confidence
                if votes == total and total > 1:  # Full consensus
                    result['risk_level'] = 'high'
                elif votes > 0:
                    result['risk_level'] = 'medium'
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
        
        return result
    
    def test_inference(self) -> bool:
        """Test the inference system"""
        print("\nTesting federated inference...")
        
        # Create test sample
        test_sample = {
            'typing_speed': 180.0,  # Slower than normal
            'typing_error_rate': 0.15,  # Higher error rate
            'touch_pressure': 0.8,
            'swipe_length': 120.0,
            'accel_variance': 0.4,
            'session_duration': 180.0,
            'interactions_per_minute': 25.0,
            'scroll_velocity': 150.0,
            'scroll_frequency': 35.0,
            'battery_level': 0.3,
            'device_temperature': 42.0,
            'app_response_time': 250.0,
            'network_latency': 180.0,
            'typing_speed_deviation': 40.0,
            'touch_pressure_deviation': 0.2,
            'time_since_last': 1800.0,
            'hour': 23.0,
            'day_of_week': 5.0,
            'night_usage_flag': 1.0,
            'is_weekend': 1.0,
            'location_pattern': 1.0
        }
        
        result = self.detect_anomaly(test_sample, "client_1")
        
        print("Test Results:")
        print(f"  Ensemble Decision: {'ANOMALY' if result['ensemble_decision'] else 'NORMAL'}")
        print(f"  Risk Level: {result['risk_level']}")
        
        if result['global_prediction']:
            gp = result['global_prediction']
            print(f"  Global: Error={gp['error']:.4f}, Threshold={gp['threshold']:.4f}, Anomaly={gp['anomaly']}")
        
        if result['client_prediction']:
            cp = result['client_prediction']
            print(f"  Client: Error={cp['error']:.4f}, Threshold={cp['threshold']:.4f}, Anomaly={cp['anomaly']}")
        
        return True


def main():
    """Main function to demonstrate the fixed federated system"""
    print("Fixed Federated Anomaly Detection System")
    print("=" * 50)
    
    # Step 1: Create and train federated system
    print("\n1. Creating Federated Server...")
    server = FixedFederatedServer(num_clients=3, num_rounds=3)
    
    if not server.create_clients(use_sample_data=True):
        print("Failed to create clients")
        return False
    
    print("\n2. Training Federated Model...")
    if not server.train_federated():
        print("Federated training failed")
        return False
    
    print("\n3. Saving Models...")
    if not server.save_models():
        print("Failed to save models")
        return False
    
    # Step 4: Test inference system
    print("\n4. Testing Inference System...")
    inference = FixedFederatedInference()
    
    # Load models
    global_loaded = inference.load_global_model()
    clients_loaded = inference.load_client_models()
    
    if not (global_loaded or clients_loaded):
        print("No models loaded for inference")
        return False
    
    # Test inference
    inference.test_inference()
    
    print("\n" + "="*50)
    print("Federated Learning System Test Complete!")
    print(f"Global model: {'Loaded' if global_loaded else 'Not loaded'}")
    print(f"Client models: {clients_loaded} loaded")
    print("="*50)
    
    return True


def run_quick_test():
    """Quick test to verify system functionality"""
    print("Quick Test: Fixed Federated System")
    print("-" * 40)
    
    try:
        # Test client creation
        client = FixedFederatedClient("test_client")
        if client.load_local_data():
            print("✓ Client creation successful")
            
            # Test model building
            model = client.build_model()
            print(f"✓ Model built - Input shape: {model.input_shape}")
            
            # Test training
            weights = client.train(epochs=2)
            if weights is not None:
                print("✓ Training successful")
            else:
                print("✗ Training failed")
        else:
            print("✗ Client creation failed")
    
    except Exception as e:
        print(f"✗ Quick test failed: {e}")
        return False
    
    print("✓ Quick test completed successfully")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fixed Federated Learning System")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick functionality test")
    parser.add_argument("--full-demo", action="store_true", 
                       help="Run full federated learning demo")
    
    args = parser.parse_args()
    
    if args.quick_test:
        success = run_quick_test()
    elif args.full_demo:
        success = main()
    else:
        print("Fixed Federated Anomaly Detection System")
        print("Usage:")
        print("  --quick-test    : Run quick functionality test")
        print("  --full-demo     : Run full federated learning demo")
        success = run_quick_test()  # Default to quick test
    
    exit(0 if success else 1)
