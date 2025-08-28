import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'anomaly-detection')))
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from tensorflow.keras.callbacks import EarlyStopping

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(0.1 * kl_loss)
        return z

class EnhancedAnomalyDetector:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.thresholds = {}
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Enhanced feature engineering"""
        # Select and engineer features
        feature_cols = [
            'typing_speed', 'typing_error_rate', 'touch_pressure',
            'swipe_length', 'accel_variance', 'session_duration',
            'interactions_per_minute', 'scroll_velocity', 'scroll_frequency',
            'battery_level', 'device_temperature', 'app_response_time',
            'network_latency', 'typing_speed_deviation', 'touch_pressure_deviation',
            'time_since_last', 'hour', 'day_of_week', 'night_usage_flag',
            'is_weekend', 'location_pattern'
        ]
        
        # Handle missing values
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
        
        # Add interaction features
        df['typing_efficiency'] = df['typing_speed'] / (1 + df['typing_error_rate'])
        df['usage_intensity'] = df['session_duration'] * df['interactions_per_minute']
        df['device_stress'] = df['device_temperature'] * df['battery_level']
        
        interaction_features = ['typing_efficiency', 'usage_intensity', 'device_stress']
        all_features = feature_cols + interaction_features
        
        return df[all_features].values
    
    def build_variational_autoencoder(self, input_dim: int, latent_dim: int = 8) -> Model:
        """Build Variational Autoencoder for better anomaly detection"""
        
        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        h = layers.Dense(64, activation='relu')(encoder_input)
        h = layers.Dropout(0.2)(h)
        h = layers.Dense(32, activation='relu')(h)
        h = layers.Dropout(0.2)(h)
        
        z_mean = layers.Dense(latent_dim)(h)
        z_log_var = layers.Dense(latent_dim)(h)
        
        # Sampling layer (defined at module level) that also adds KL divergence via add_loss
        z = Sampling()([z_mean, z_log_var])
        
        # Decoder
        decoder_h = layers.Dense(32, activation='relu')(z)
        decoder_h = layers.Dropout(0.2)(decoder_h)
        decoder_h = layers.Dense(64, activation='relu')(decoder_h)
        decoder_output = layers.Dense(input_dim, activation='linear')(decoder_h)
        
        # VAE model
        vae = Model(encoder_input, decoder_output)
        
        # Compile with standard reconstruction loss; KL is added via Sampling layer
        vae.compile(optimizer='adam', loss='mse')
        return vae
    
    def build_lstm_autoencoder(self, input_dim: int, sequence_length: int = 10) -> Model:
        """Build LSTM Autoencoder for sequential patterns"""
        
        # Encoder
        encoder_input = layers.Input(shape=(sequence_length, input_dim))
        encoded = layers.LSTM(64, return_state=True)
        encoder_output, state_h, state_c = encoded(encoder_input)
        
        # Decoder
        decoder_lstm = layers.LSTM(64, return_sequences=True, return_state=True)
        decoder_dense = layers.Dense(input_dim)
        
        # Repeat encoder output for decoder
        decoder_input = layers.RepeatVector(sequence_length)(encoder_output)
        decoder_output, _, _ = decoder_lstm(decoder_input, initial_state=[state_h, state_c])
        decoder_output = decoder_dense(decoder_output)
        
        lstm_ae = Model(encoder_input, decoder_output)
        lstm_ae.compile(optimizer='adam', loss='mse')
        return lstm_ae
    
    def create_sequences(self, X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
        """Create sequences for LSTM"""
        sequences = []
        for i in range(len(X) - sequence_length + 1):
            sequences.append(X[i:i + sequence_length])
        return np.array(sequences)
    
    def train_models(self, df: pd.DataFrame) -> Dict:
        """Train multiple anomaly detection models"""
        
        # Prepare features
        X = self.prepare_features(df)
        y = df['label'].values if 'label' in df.columns else None
        
        # Split data
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
            y_train = y_test = None
        
        # Scale features
        robust_scaler = RobustScaler()  # Better for outliers
        standard_scaler = StandardScaler()
        
        X_train_robust = robust_scaler.fit_transform(X_train)
        X_test_robust = robust_scaler.transform(X_test)
        
        X_train_standard = standard_scaler.fit_transform(X_train)
        X_test_standard = standard_scaler.transform(X_test)
        
        self.scalers['robust'] = robust_scaler
        self.scalers['standard'] = standard_scaler
        
        results = {}
        
        # 1. Variational Autoencoder
        print("Training Variational Autoencoder...")
        vae = self.build_variational_autoencoder(X_train_robust.shape[1])
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        vae.fit(X_train_robust, X_train_robust, 
                epochs=100, batch_size=32, verbose=0,
                validation_split=0.2, callbacks=[early_stop])
        
        self.models['vae'] = vae
        
        # Calculate VAE threshold
        vae_recon = vae.predict(X_train_robust, verbose=0)
        vae_errors = np.mean((X_train_robust - vae_recon) ** 2, axis=1)
        self.thresholds['vae'] = np.mean(vae_errors) + 2.5 * np.std(vae_errors)
        
        # Evaluate VAE
        if y_test is not None:
            vae_test_recon = vae.predict(X_test_robust, verbose=0)
            vae_test_errors = np.mean((X_test_robust - vae_test_recon) ** 2, axis=1)
            vae_predictions = (vae_test_errors > self.thresholds['vae']).astype(int)
            results['vae'] = {
                'auc': roc_auc_score(y_test, vae_test_errors),
                'accuracy': np.mean(vae_predictions == y_test),
                'classification_report': classification_report(y_test, vae_predictions)
            }
        
        # 2. LSTM Autoencoder (for temporal patterns)
        print("Training LSTM Autoencoder...")
        sequence_length = min(10, len(X_train) // 100)  # Adjust based on data size
        
        # Prepare sequences
        X_train_seq = self.create_sequences(X_train_standard, sequence_length)
        X_test_seq = self.create_sequences(X_test_standard, sequence_length)
        
        if len(X_train_seq) > 0:  # Only if we have enough data for sequences
            lstm_ae = self.build_lstm_autoencoder(X_train_standard.shape[1], sequence_length)
            lstm_ae.fit(X_train_seq, X_train_seq,
                       epochs=50, batch_size=16, verbose=0,
                       validation_split=0.2)
            
            self.models['lstm_ae'] = lstm_ae
            
            # Calculate LSTM threshold
            lstm_recon = lstm_ae.predict(X_train_seq, verbose=0)
            lstm_errors = np.mean((X_train_seq - lstm_recon) ** 2, axis=(1, 2))
            self.thresholds['lstm_ae'] = np.mean(lstm_errors) + 2.5 * np.std(lstm_errors)
            
            # Evaluate LSTM
            if y_test is not None and len(X_test_seq) > 0:
                lstm_test_recon = lstm_ae.predict(X_test_seq, verbose=0)
                lstm_test_errors = np.mean((X_test_seq - lstm_test_recon) ** 2, axis=(1, 2))
                lstm_predictions = (lstm_test_errors > self.thresholds['lstm_ae']).astype(int)
                
                # Align labels with sequence predictions
                y_test_seq = y_test[sequence_length-1:][:len(lstm_test_errors)]
                
                results['lstm_ae'] = {
                    'auc': roc_auc_score(y_test_seq, lstm_test_errors),
                    'accuracy': np.mean(lstm_predictions == y_test_seq),
                    'classification_report': classification_report(y_test_seq, lstm_predictions)
                }
        
        # 3. Isolation Forest (ensemble method)
        print("Training Isolation Forest...")
        iso_forest = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        iso_forest.fit(X_train_standard)
        self.models['isolation_forest'] = iso_forest
        
        if y_test is not None:
            iso_scores = iso_forest.decision_function(X_test_standard)
            iso_predictions = iso_forest.predict(X_test_standard)
            iso_predictions = (iso_predictions == -1).astype(int)  # Convert to 0/1
            
            results['isolation_forest'] = {
                'auc': roc_auc_score(y_test, -iso_scores),  # Negative because lower scores = anomaly
                'accuracy': np.mean(iso_predictions == y_test),
                'classification_report': classification_report(y_test, iso_predictions)
            }
        
        # 4. Ensemble approach
        if y_test is not None:
            print("Creating ensemble predictions...")
            ensemble_scores = []
            
            # Normalize all scores to [0, 1]
            vae_scores_norm = (vae_test_errors - np.min(vae_test_errors)) / (np.max(vae_test_errors) - np.min(vae_test_errors))
            iso_scores_norm = (-iso_scores - np.min(-iso_scores)) / (np.max(-iso_scores) - np.min(-iso_scores))
            
            ensemble_score = 0.6 * vae_scores_norm + 0.4 * iso_scores_norm
            ensemble_predictions = (ensemble_score > 0.5).astype(int)
            
            results['ensemble'] = {
                'auc': roc_auc_score(y_test, ensemble_score),
                'accuracy': np.mean(ensemble_predictions == y_test),
                'classification_report': classification_report(y_test, ensemble_predictions)
            }
        
        # Save all models and scalers
        self.save_models()
        
        return results
    
    def save_models(self):
        """Save all trained models and components"""
        # Save TensorFlow models
        if 'vae' in self.models:
            self.models['vae'].save("enhanced_vae_model.h5")
        if 'lstm_ae' in self.models:
            self.models['lstm_ae'].save("enhanced_lstm_model.h5")
        
        # Save sklearn models and scalers
        joblib.dump(self.scalers, "enhanced_scalers.pkl")
        joblib.dump(self.models.get('isolation_forest'), "isolation_forest_model.pkl")
        
        # Save thresholds
        with open("enhanced_thresholds.json", "w") as f:
            json.dump(self.thresholds, f)
        
        print("✅ All models and components saved!")
    
    def plot_results(self, results: Dict):
        """Plot model comparison results"""
        if not results:
            return
            
        models = list(results.keys())
        aucs = [results[model]['auc'] for model in models]
        accuracies = [results[model]['accuracy'] for model in models]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # AUC comparison
        ax1.bar(models, aucs)
        ax1.set_title('Model AUC Comparison')
        ax1.set_ylabel('AUC Score')
        ax1.set_ylim(0, 1)
        
        # Accuracy comparison
        ax2.bar(models, accuracies)
        ax2.set_title('Model Accuracy Comparison')
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

# Load data and train models
print("Loading enhanced dataset...")
try:
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "enhanced_behavior_dataset.csv"))
except FileNotFoundError:
    print("Enhanced dataset not found. Using original dataset...")
    df = pd.read_csv("rich_user_behavior.csv")

# Initialize and train detector
detector = EnhancedAnomalyDetector()
results = detector.train_models(df)

# Display results
print("\n" + "="*50)
print("ENHANCED MODEL EVALUATION RESULTS")
print("="*50)

for model_name, metrics in results.items():
    print(f"\n{model_name.upper()} Results:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print("  Classification Report:")
    print("  " + metrics['classification_report'].replace('\n', '\n  '))

# Plot comparison if results available
if results:
    detector.plot_results(results)

print(f"\n✅ Enhanced training completed!")
print(f"Models trained: {list(results.keys())}")