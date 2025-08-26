import sys
sys.path.append('../anomaly-detection')
from enhanced_training_model import EnhancedAnomalyDetector
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

class FederatedAnomalyInference:
    def __init__(self):
        self.federated_model = None
        self.client_models = {}
        self.scalers = {}
        self.thresholds = {}
        self.inference_history = []
        
    def load_federated_model(self, model_path: str = "federated_models/federated_global_model.h5"):
        """Load the federated global model"""
        try:
            if os.path.exists(model_path):
                # Try loading with custom objects first
                try:
                    self.federated_model = tf.keras.models.load_model(
                        model_path, 
                        custom_objects={
                            'mae': tf.keras.metrics.MeanAbsoluteError(),
                            'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError()
                        }
                    )
                except:
                    # Fallback: load without custom objects
                    self.federated_model = tf.keras.models.load_model(model_path, compile=False)
                    # Recompile the model
                    self.federated_model.compile(
                        optimizer='adam', 
                        loss='mse', 
                        metrics=[tf.keras.metrics.MeanAbsoluteError()]
                    )
                
                print(f"✅ Loaded federated global model: {model_path}")
                return True
            else:
                print(f"⚠️  Federated model not found: {model_path}")
                return False
        except Exception as e:
            print(f"❌ Error loading federated model: {e}")
            return False
    
    def load_client_models(self, models_dir: str = "local_models"):
        """Load all client models"""
        if not os.path.exists(models_dir):
            print(f"⚠️  Client models directory not found: {models_dir}")
            return False
        
        print("📂 Loading client models...")
        loaded_count = 0
        
        for file in os.listdir(models_dir):
            if file.endswith("_model.h5"):
                client_id = file.replace("_model.h5", "")
                model_path = os.path.join(models_dir, file)
                scaler_path = os.path.join(models_dir, f"{client_id}_scaler.pkl")
                
                try:
                    # Try loading with custom objects first
                    try:
                        model = tf.keras.models.load_model(
                            model_path,
                            custom_objects={
                                'mae': tf.keras.metrics.MeanAbsoluteError(),
                                'MeanAbsoluteError': tf.keras.metrics.MeanAbsoluteError()
                            }
                        )
                    except:
                        # Fallback: load without custom objects
                        model = tf.keras.models.load_model(model_path, compile=False)
                        # Recompile the model
                        model.compile(
                            optimizer='adam', 
                            loss='mse', 
                            metrics=[tf.keras.metrics.MeanAbsoluteError()]
                        )
                    
                    self.client_models[client_id] = model
                    
                    # Load scaler
                    if os.path.exists(scaler_path):
                        scaler = joblib.load(scaler_path)
                        self.scalers[client_id] = scaler
                    
                    loaded_count += 1
                    print(f"  ✅ Loaded {client_id}")
                    
                except Exception as e:
                    print(f"  ❌ Error loading {client_id}: {e}")
        
        print(f"📊 Loaded {loaded_count} client models")
        return loaded_count > 0
    
    def prepare_features(self, sample_data: Dict) -> np.ndarray:
        """Prepare features from raw sample data - matches training exactly"""
        # Base features (21 total) - must match training exactly
        feature_cols = [
            'typing_speed', 'typing_error_rate', 'touch_pressure',
            'swipe_length', 'accel_variance', 'session_duration',
            'interactions_per_minute', 'scroll_velocity', 'scroll_frequency',
            'battery_level', 'device_temperature', 'app_response_time',
            'network_latency', 'typing_speed_deviation', 'touch_pressure_deviation',
            'time_since_last', 'hour', 'day_of_week', 'night_usage_flag',
            'is_weekend', 'location_pattern'
        ]
        
        # Extract features with fallback defaults
        features = []
        for feature in feature_cols:
            if feature in sample_data:
                features.append(sample_data[feature])
            else:
                # Use reasonable defaults for missing features
                defaults = {
                    'typing_speed': 250, 'typing_error_rate': 0.05,
                    'touch_pressure': 0.5, 'swipe_length': 150,
                    'accel_variance': 0.2, 'session_duration': 300,
                    'interactions_per_minute': 15, 'scroll_velocity': 200,
                    'scroll_frequency': 25, 'battery_level': 0.7,
                    'device_temperature': 35, 'app_response_time': 150,
                    'network_latency': 100, 'typing_speed_deviation': 0.1,
                    'touch_pressure_deviation': 0.05, 'time_since_last': 3600,
                    'hour': 12, 'day_of_week': 2, 'night_usage_flag': 0,
                    'is_weekend': 0, 'location_pattern': 0
                }
                features.append(defaults.get(feature, 0))
        
        # Add interaction features (3 total) - must match training exactly
        # typing_efficiency
        if 'typing_speed' in sample_data and 'typing_error_rate' in sample_data:
            typing_efficiency = sample_data['typing_speed'] / (1 + sample_data['typing_error_rate'])
        else:
            typing_efficiency = 250 / 1.05  # Default efficiency
        features.append(typing_efficiency)
        
        # usage_intensity
        if 'session_duration' in sample_data and 'interactions_per_minute' in sample_data:
            usage_intensity = sample_data['session_duration'] * sample_data['interactions_per_minute']
        else:
            usage_intensity = 4500  # Default intensity
        features.append(usage_intensity)
        
        # device_stress
        if 'device_temperature' in sample_data and 'battery_level' in sample_data:
            device_stress = sample_data['device_temperature'] * sample_data['battery_level']
        else:
            device_stress = 24.5  # Default stress
        features.append(device_stress)
        
        # Total: 21 base + 3 interaction = 24 features
        return np.array(features).reshape(1, -1)
    
    def detect_anomaly_federated(self, sample_data: Dict, client_id: Optional[str] = None) -> Dict:
        """Detect anomalies using federated models"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'federated_scores': {},
            'client_scores': {},
            'ensemble_decision': 0,
            'risk_level': 'low',
            'explanations': []
        }
        
        # Prepare features
        X = self.prepare_features(sample_data)
        
        # Federated global model prediction
        if self.federated_model is not None:
            try:
                # Scale features for federated model
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Get reconstruction
                reconstruction = self.federated_model.predict(X_scaled, verbose=0)
                federated_error = np.mean((X_scaled - reconstruction) ** 2)
                
                # Calculate threshold
                federated_threshold = np.mean(federated_error) + 2 * np.std(federated_error)
                federated_anomaly = int(federated_error > federated_threshold)
                
                results['federated_scores'] = {
                    'error': float(federated_error),
                    'threshold': float(federated_threshold),
                    'anomaly': federated_anomaly,
                    'confidence': min(1.0, federated_error / federated_threshold) if federated_threshold > 0 else 0.0
                }
                
                if federated_anomaly:
                    results['explanations'].append("Federated global model detected anomaly")
                    
            except Exception as e:
                print(f"Federated model prediction error: {e}")
        
        # Client-specific model predictions
        if client_id and client_id in self.client_models:
            try:
                client_model = self.client_models[client_id]
                client_scaler = self.scalers.get(client_id)
                
                if client_scaler is not None:
                    X_scaled = client_scaler.transform(X)
                else:
                    # Use default scaler if client scaler not available
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                
                # Get reconstruction
                reconstruction = client_model.predict(X_scaled, verbose=0)
                client_error = np.mean((X_scaled - reconstruction) ** 2)
                
                # Calculate threshold
                client_threshold = np.mean(client_error) + 2 * np.std(client_error)
                client_anomaly = int(client_error > client_threshold)
                
                results['client_scores'] = {
                    'error': float(client_error),
                    'threshold': float(client_threshold),
                    'anomaly': client_anomaly,
                    'confidence': min(1.0, client_error / client_threshold) if client_threshold > 0 else 0.0
                }
                
                if client_anomaly:
                    results['explanations'].append(f"Client {client_id} model detected anomaly")
                    
            except Exception as e:
                print(f"Client model prediction error: {e}")
        
        # Ensemble decision
        anomaly_votes = 0
        total_models = 0
        
        if results['federated_scores']:
            anomaly_votes += results['federated_scores']['anomaly']
            total_models += 1
        
        if results['client_scores']:
            anomaly_votes += results['client_scores']['anomaly']
            total_models += 1
        
        if total_models > 0:
            vote_ratio = anomaly_votes / total_models
            results['ensemble_decision'] = int(vote_ratio > 0.5)
            
            # Determine risk level
            if vote_ratio > 0.7:
                results['risk_level'] = 'high'
            elif vote_ratio > 0.4:
                results['risk_level'] = 'medium'
            else:
                results['risk_level'] = 'low'
        
        # Store inference history
        self.inference_history.append(results)
        
        return results
    
    def batch_federated_inference(self, df: pd.DataFrame, client_id: str = None) -> pd.DataFrame:
        """Process batch of data using federated models"""
        results_list = []
        
        print(f"🔄 Processing {len(df)} samples with federated models...")
        
        for idx, row in df.iterrows():
            sample_data = row.to_dict()
            
            result = self.detect_anomaly_federated(sample_data, client_id)
            
            # Flatten results for DataFrame
            flat_result = {
                'sample_id': idx,
                'client_id': client_id,
                'ensemble_decision': result['ensemble_decision'],
                'risk_level': result['risk_level'],
                'explanations': '; '.join(result['explanations'])
            }
            
            # Add federated scores
            if result['federated_scores']:
                flat_result['federated_error'] = result['federated_scores']['error']
                flat_result['federated_anomaly'] = result['federated_scores']['anomaly']
                flat_result['federated_confidence'] = result['federated_scores']['confidence']
            
            # Add client scores
            if result['client_scores']:
                flat_result['client_error'] = result['client_scores']['error']
                flat_result['client_anomaly'] = result['client_scores']['anomaly']
                flat_result['client_confidence'] = result['client_scores']['confidence']
            
            results_list.append(flat_result)
        
        results_df = pd.DataFrame(results_list)
        
        # Summary statistics
        total_anomalies = results_df['ensemble_decision'].sum()
        risk_distribution = results_df['risk_level'].value_counts()
        
        print(f"\n=== Federated Inference Results ===")
        print(f"Total samples processed: {len(results_df)}")
        print(f"Anomalies detected: {total_anomalies}")
        print(f"Anomaly rate: {total_anomalies/len(results_df):.2%}")
        print(f"Risk level distribution:")
        for risk, count in risk_distribution.items():
            print(f"  {risk}: {count} ({count/len(results_df):.1%})")
        
        return results_df
    
    def compare_models(self, test_data_path: str = "../anomaly-detection/enhanced_behavior_dataset.csv"):
        """Compare federated vs centralized model performance"""
        print("\n🔍 Comparing Federated vs Centralized Models...")
        
        # Load test data
        test_df = pd.read_csv(test_data_path)
        test_sample = test_df.head(20)  # Use first 20 samples for comparison
        
        # Federated inference
        federated_results = self.batch_federated_inference(test_sample)
        
        # Load centralized model for comparison
        centralized_results = None
        try:
            from anomaly_detection.real_time_inference import AdaptiveAnomalyDetector
            centralized_detector = AdaptiveAnomalyDetector()
            centralized_detector.load_models()
            centralized_results = centralized_detector.batch_inference(test_sample)
            
            print(f"\n📊 Model Comparison:")
            print(f"  Federated Anomalies: {federated_results['ensemble_decision'].sum()}")
            print(f"  Centralized Anomalies: {centralized_results['final_decision'].sum()}")
            
            # Calculate agreement
            agreement = np.mean(federated_results['ensemble_decision'] == centralized_results['final_decision'])
            print(f"  Model Agreement: {agreement:.2%}")
            
        except Exception as e:
            print(f"⚠️  Could not load centralized model for comparison: {e}")
        
        return federated_results, centralized_results
    
    def save_inference_results(self, results_df: pd.DataFrame, filename: str = "federated_inference_results.csv"):
        """Save inference results"""
        results_df.to_csv(filename, index=False)
        print(f"✅ Federated inference results saved to {filename}")
    
    def plot_federated_results(self, results_df: pd.DataFrame):
        """Plot federated inference results"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. Anomaly distribution
            anomaly_counts = results_df['ensemble_decision'].value_counts()
            axes[0, 0].pie(anomaly_counts.values, labels=anomaly_counts.index, autopct='%1.1f%%')
            axes[0, 0].set_title('Anomaly Distribution')
            
            # 2. Risk level distribution
            risk_counts = results_df['risk_level'].value_counts()
            colors = {'low': 'green', 'medium': 'orange', 'high': 'red'}
            risk_colors = [colors.get(risk, 'gray') for risk in risk_counts.index]
            axes[0, 1].bar(risk_counts.index, risk_counts.values, color=risk_colors)
            axes[0, 1].set_title('Risk Level Distribution')
            axes[0, 1].set_ylabel('Count')
            
            # 3. Federated vs Client errors (if available)
            if 'federated_error' in results_df.columns and 'client_error' in results_df.columns:
                axes[1, 0].scatter(results_df['federated_error'], results_df['client_error'], alpha=0.6)
                axes[1, 0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
                axes[1, 0].set_xlabel('Federated Error')
                axes[1, 0].set_ylabel('Client Error')
                axes[1, 0].set_title('Federated vs Client Errors')
            
            # 4. Confidence distribution
            if 'federated_confidence' in results_df.columns:
                axes[1, 1].hist(results_df['federated_confidence'], bins=20, alpha=0.7)
                axes[1, 1].set_xlabel('Confidence')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Federated Model Confidence')
            
            plt.tight_layout()
            plt.savefig('federated_inference_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Federated inference analysis plot saved")
            
        except ImportError:
            print("⚠️  matplotlib not available for plotting")

def main():
    """Main function to run federated inference"""
    print("🌐 Federated Anomaly Detection Inference")
    print("=" * 50)
    
    # Initialize federated inference
    federated_inference = FederatedAnomalyInference()
    
    # Load models
    federated_loaded = federated_inference.load_federated_model()
    client_loaded = federated_inference.load_client_models()
    
    if not federated_loaded and not client_loaded:
        print("❌ No models loaded. Please run federated training first.")
        return
    
    # Test single sample
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
    
    print("\n🔍 Single Sample Test:")
    result = federated_inference.detect_anomaly_federated(test_sample, "client_1")
    
    print(f"  Ensemble Decision: {'ANOMALY' if result['ensemble_decision'] else 'NORMAL'}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Explanations: {result['explanations']}")
    
    if result['federated_scores']:
        print(f"  Federated Error: {result['federated_scores']['error']:.4f}")
        print(f"  Federated Confidence: {result['federated_scores']['confidence']:.4f}")
    
    if result['client_scores']:
        print(f"  Client Error: {result['client_scores']['error']:.4f}")
        print(f"  Client Confidence: {result['client_scores']['confidence']:.4f}")
    
    # Batch inference
    print("\n📊 Running Batch Inference...")
    try:
        test_data = pd.read_csv("../anomaly-detection/enhanced_behavior_dataset.csv")
        test_sample = test_data.head(50)  # Use first 50 samples
        
        batch_results = federated_inference.batch_federated_inference(test_sample, "client_1")
        
        # Save results
        federated_inference.save_inference_results(batch_results)
        
        # Plot results
        federated_inference.plot_federated_results(batch_results)
        
        # Compare with centralized model
        federated_inference.compare_models()
        
    except FileNotFoundError:
        print("⚠️  Test dataset not found. Skipping batch inference.")
    
    print(f"\n🎉 Federated inference completed!")

if __name__ == "__main__":
    main()
