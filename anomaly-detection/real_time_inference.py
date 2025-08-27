import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import json
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import deque
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class Sampling(tf.keras.layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        self.add_loss(0.1 * kl_loss)
        return z

class AdaptiveAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        self.user_profiles = {}
        self.recent_scores = deque(maxlen=100)  # Store recent anomaly scores
        self.adaptation_rate = 0.1  # How quickly thresholds adapt
        
    def load_models(self):
        """Load all trained models and components"""
        try:
            # Load TensorFlow models
            if os.path.exists("enhanced_vae_model.h5"):
                self.models['vae'] = tf.keras.models.load_model(
                    "enhanced_vae_model.h5", compile=False, custom_objects={"Sampling": Sampling}
                )
                print("✅ VAE model loaded")
            
            if os.path.exists("enhanced_lstm_model.h5"):
                self.models['lstm_ae'] = tf.keras.models.load_model("enhanced_lstm_model.h5", compile=False)
                print("✅ LSTM model loaded")
            
            # Load sklearn models and scalers
            if os.path.exists("enhanced_scalers.pkl"):
                self.scalers = joblib.load("enhanced_scalers.pkl")
                print("✅ Scalers loaded")
            
            if os.path.exists("isolation_forest_model.pkl"):
                self.models['isolation_forest'] = joblib.load("isolation_forest_model.pkl")
                print("✅ Isolation Forest loaded")
            
            # Load thresholds
            if os.path.exists("enhanced_thresholds.json"):
                with open("enhanced_thresholds.json", "r") as f:
                    self.thresholds = json.load(f)
                print("✅ Thresholds loaded")
            
            # Fallback to original models if enhanced ones not available
            if not self.models:
                if os.path.exists("rich_behavior_model.h5"):
                    self.models['basic_ae'] = tf.keras.models.load_model("rich_behavior_model.h5", compile=False)
                    print("⚠️  Using basic autoencoder model")
                
                if os.path.exists("rich_scaler.pkl"):
                    self.scalers['standard'] = joblib.load("rich_scaler.pkl")
                    print("⚠️  Using basic scaler")
                
                if os.path.exists("rich_threshold.json"):
                    with open("rich_threshold.json", "r") as f:
                        self.thresholds['basic_ae'] = json.load(f)["threshold"]
                        print("⚠️  Using basic threshold")
        
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
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
    
    def update_user_profile(self, user_id: str, features: np.ndarray, is_anomaly: bool):
        """Update user's behavioral profile for personalization"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'feature_history': deque(maxlen=50),
                'anomaly_count': 0,
                'total_sessions': 0,
                'personal_threshold_multiplier': 1.0
            }
        
        profile = self.user_profiles[user_id]
        profile['feature_history'].append(features[0])
        profile['total_sessions'] += 1
        
        if is_anomaly:
            profile['anomaly_count'] += 1
        
        # Adjust personal threshold based on user's anomaly rate
        personal_anomaly_rate = profile['anomaly_count'] / profile['total_sessions']
        
        # If user has very low anomaly rate, they might be very consistent
        # If very high, they might be naturally variable
        if personal_anomaly_rate < 0.01:  # Very consistent user
            profile['personal_threshold_multiplier'] = 0.8
        elif personal_anomaly_rate > 0.1:  # Naturally variable user
            profile['personal_threshold_multiplier'] = 1.3
        else:
            profile['personal_threshold_multiplier'] = 1.0
    
    def adaptive_threshold(self, base_threshold: float, user_id: str = None) -> float:
        """Calculate adaptive threshold based on recent patterns"""
        adapted_threshold = base_threshold
        
        # Global adaptation based on recent anomaly rates
        if len(self.recent_scores) > 10:
            recent_mean = np.mean(self.recent_scores)
            recent_std = np.std(self.recent_scores)
            
            # If recent scores are generally higher, increase threshold
            if recent_mean > base_threshold:
                adapted_threshold = base_threshold + 0.5 * recent_std
            else:
                adapted_threshold = max(base_threshold * 0.7, base_threshold - 0.3 * recent_std)
        
        # User-specific adaptation
        if user_id and user_id in self.user_profiles:
            multiplier = self.user_profiles[user_id]['personal_threshold_multiplier']
            adapted_threshold *= multiplier
        
        return adapted_threshold
    
    def detect_anomaly(self, sample_data: Dict, user_id: str = None) -> Dict:
        """Comprehensive anomaly detection with multiple models"""
        
        # Prepare features
        X = self.prepare_features(sample_data)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'anomaly_scores': {},
            'predictions': {},
            'confidence_scores': {},
            'final_decision': 0,
            'risk_level': 'low',
            'explanation': []
        }
        
        anomaly_votes = 0
        total_models = 0
        weighted_score = 0
        
        # VAE Prediction
        if 'vae' in self.models and 'robust' in self.scalers:
            try:
                X_scaled = self.scalers['robust'].transform(X)
                recon = self.models['vae'].predict(X_scaled, verbose=0)
                vae_error = np.mean((X_scaled - recon) ** 2)
                
                base_threshold = self.thresholds.get('vae', 0.1)
                adaptive_thresh = self.adaptive_threshold(base_threshold, user_id)
                
                vae_anomaly = int(vae_error > adaptive_thresh)
                
                results['anomaly_scores']['vae'] = float(vae_error)
                results['predictions']['vae'] = vae_anomaly
                results['confidence_scores']['vae'] = min(1.0, vae_error / adaptive_thresh)
                
                anomaly_votes += vae_anomaly * 2  # VAE gets higher weight
                total_models += 2
                weighted_score += vae_error * 0.4
                
                if vae_anomaly:
                    results['explanation'].append("VAE detected unusual behavioral reconstruction pattern")
                    
            except Exception as e:
                print(f"VAE prediction error: {e}")
        
        # Isolation Forest Prediction
        if 'isolation_forest' in self.models and 'standard' in self.scalers:
            try:
                X_scaled = self.scalers['standard'].transform(X)
                iso_score = self.models['isolation_forest'].decision_function(X_scaled)[0]
                iso_prediction = self.models['isolation_forest'].predict(X_scaled)[0]
                iso_anomaly = int(iso_prediction == -1)
                
                results['anomaly_scores']['isolation_forest'] = float(-iso_score)
                results['predictions']['isolation_forest'] = iso_anomaly
                results['confidence_scores']['isolation_forest'] = min(1.0, abs(iso_score))
                
                anomaly_votes += iso_anomaly
                total_models += 1
                weighted_score += abs(iso_score) * 0.3
                
                if iso_anomaly:
                    results['explanation'].append("Isolation Forest detected anomalous feature combination")
                    
            except Exception as e:
                print(f"Isolation Forest prediction error: {e}")
        
        # LSTM Prediction (if available and we have sequence data)
        if 'lstm_ae' in self.models and user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if len(profile['feature_history']) >= 10:
                try:
                    # Create sequence from user's recent history
                    sequence = np.array(list(profile['feature_history'])[-10:])
                    sequence = sequence.reshape(1, 10, -1)
                    
                    # Scale sequence
                    if 'standard' in self.scalers:
                        sequence_scaled = self.scalers['standard'].transform(
                            sequence.reshape(-1, sequence.shape[-1])
                        ).reshape(sequence.shape)
                        
                        lstm_recon = self.models['lstm_ae'].predict(sequence_scaled, verbose=0)
                        lstm_error = np.mean((sequence_scaled - lstm_recon) ** 2)
                        
                        base_threshold = self.thresholds.get('lstm_ae', 0.1)
                        adaptive_thresh = self.adaptive_threshold(base_threshold, user_id)
                        
                        lstm_anomaly = int(lstm_error > adaptive_thresh)
                        
                        results['anomaly_scores']['lstm_ae'] = float(lstm_error)
                        results['predictions']['lstm_ae'] = lstm_anomaly
                        results['confidence_scores']['lstm_ae'] = min(1.0, lstm_error / adaptive_thresh)
                        
                        anomaly_votes += lstm_anomaly * 1.5  # LSTM gets medium weight
                        total_models += 1.5
                        weighted_score += lstm_error * 0.3
                        
                        if lstm_anomaly:
                            results['explanation'].append("LSTM detected unusual temporal behavior pattern")
                            
                except Exception as e:
                    print(f"LSTM prediction error: {e}")
        
        # Basic Autoencoder (fallback)
        if 'basic_ae' in self.models and 'standard' in self.scalers:
            try:
                X_scaled = self.scalers['standard'].transform(X)
                recon = self.models['basic_ae'].predict(X_scaled, verbose=0)
                ae_error = np.mean((X_scaled - recon) ** 2)
                
                base_threshold = self.thresholds.get('basic_ae', 0.1)
                adaptive_thresh = self.adaptive_threshold(base_threshold, user_id)
                
                ae_anomaly = int(ae_error > adaptive_thresh)
                
                results['anomaly_scores']['basic_ae'] = float(ae_error)
                results['predictions']['basic_ae'] = ae_anomaly
                results['confidence_scores']['basic_ae'] = min(1.0, ae_error / adaptive_thresh)
                
                anomaly_votes += ae_anomaly
                total_models += 1
                weighted_score += ae_error * 0.2
                
                if ae_anomaly:
                    results['explanation'].append("Basic autoencoder detected behavioral deviation")
                    
            except Exception as e:
                print(f"Basic AE prediction error: {e}")
        
        # Make final decision based on ensemble voting
        if total_models > 0:
            vote_ratio = anomaly_votes / total_models
            results['final_decision'] = int(vote_ratio > 0.4)  # 40% threshold for ensemble
            
            # Determine risk level
            if vote_ratio > 0.7:
                results['risk_level'] = 'high'
            elif vote_ratio > 0.4:
                results['risk_level'] = 'medium'
            else:
                results['risk_level'] = 'low'
            
            # Store score for adaptation
            self.recent_scores.append(weighted_score)
            
            # Update user profile
            if user_id:
                self.update_user_profile(user_id, X, results['final_decision'])
        
        return results
    
    def batch_inference(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process batch of data for inference"""
        results_list = []
        
        print(f"Processing {len(df)} samples for batch inference...")
        
        for idx, row in df.iterrows():
            sample_data = row.to_dict()
            user_id = sample_data.get('user_id', f'user_{idx}')
            
            result = self.detect_anomaly(sample_data, user_id)
            
            # Flatten results for DataFrame
            flat_result = {
                'sample_id': idx,
                'user_id': user_id,
                'final_decision': result['final_decision'],
                'risk_level': result['risk_level'],
                'explanation': '; '.join(result['explanation'])
            }
            
            # Add individual model scores
            for model, score in result['anomaly_scores'].items():
                flat_result[f'{model}_score'] = score
                flat_result[f'{model}_prediction'] = result['predictions'][model]
            
            results_list.append(flat_result)
        
        results_df = pd.DataFrame(results_list)
        
        # Summary statistics
        total_anomalies = results_df['final_decision'].sum()
        risk_distribution = results_df['risk_level'].value_counts()
        
        print(f"\n=== Batch Inference Results ===")
        print(f"Total samples processed: {len(results_df)}")
        print(f"Anomalies detected: {total_anomalies}")
        print(f"Anomaly rate: {total_anomalies/len(results_df):.2%}")
        print(f"Risk level distribution:")
        for risk, count in risk_distribution.items():
            print(f"  {risk}: {count} ({count/len(results_df):.1%})")
        
        return results_df
    
    def real_time_simulation(self, df: pd.DataFrame, delay: float = 0.5):
        """Simulate real-time anomaly detection"""
        print("\n=== Real-time Anomaly Detection Simulation ===")
        print("Processing samples one by one...\n")
        
        for idx, row in df.iterrows():
            sample_data = row.to_dict()
            user_id = sample_data.get('user_id', f'user_{idx}')
            
            start_time = time.time()
            result = self.detect_anomaly(sample_data, user_id)
            processing_time = time.time() - start_time
            
            # Display result
            status = "🚨 ANOMALY" if result['final_decision'] else "✅ NORMAL"
            risk = result['risk_level'].upper()
            
            print(f"[Sample {idx+1}] {status} | Risk: {risk} | User: {user_id}")
            print(f"  Processing time: {processing_time*1000:.1f}ms")
            
            if result['explanation']:
                print(f"  Reasons: {'; '.join(result['explanation'])}")
            
            # Show top model scores
            if result['anomaly_scores']:
                scores_str = ', '.join([
                    f"{model}: {score:.4f}" 
                    for model, score in result['anomaly_scores'].items()
                ])
                print(f"  Scores: {scores_str}")
            
            print()
            
            # Simulate processing delay
            time.sleep(delay)
        
        print("✅ Real-time simulation completed!")

# Initialize detector and load models
detector = AdaptiveAnomalyDetector()
detector.load_models()

# Load test data
try:
    print("Loading test data...")
    test_data = pd.read_csv("enhanced_behavior_dataset.csv")
    
    # Take a random subset for demonstration
    test_sample = test_data.sample(n=20, random_state=42).copy()  # Fixed seed for reproducibility
    
    # Batch inference
    print("\n" + "="*60)
    print("RUNNING BATCH INFERENCE")
    print("="*60)
    batch_results = detector.batch_inference(test_sample)
    
    # Save batch results
    batch_results.to_csv("advanced_batch_results.csv", index=False)
    print(f"\n✅ Batch results saved to 'advanced_batch_results.csv'")
    
    # Real-time simulation (first 10 samples)
    print("\n" + "="*60)
    print("RUNNING REAL-TIME SIMULATION")
    print("="*60)
    detector.real_time_simulation(test_sample.head(10), delay=0.3)
    
except FileNotFoundError:
    print("⚠️  Test dataset not found. Please run the enhanced data generator first.")
    print("Creating sample data for demonstration...")
    
    # Create minimal sample data for testing
    sample_data = {
        'user_id': 'demo_user',
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
    
    print("\nRunning single sample detection...")
    result = detector.detect_anomaly(sample_data, 'demo_user')
    
    print(f"Sample Result:")
    print(f"  Decision: {'ANOMALY' if result['final_decision'] else 'NORMAL'}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Explanations: {result['explanation']}")
    print(f"  Scores: {result['anomaly_scores']}")

print(f"\n🎯 Advanced inference system ready!")
print(f"📊 User profiles maintained: {len(detector.user_profiles)}")
print(f"🔄 Adaptive thresholds active")
print(f"🤖 Models loaded: {list(detector.models.keys())}")

