import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class FederatedAnomalyServer:
    def __init__(self, num_clients: int = 3, rounds: int = 10):
        self.num_clients = num_clients
        self.rounds = rounds
        self.global_model = None
        self.client_models = []
        self.training_history = []
        
    def create_federated_dataset(self, base_dataset_path: str = "../anomaly-detection/enhanced_behavior_dataset.csv"):
        """Split the main dataset into client-specific datasets for federated learning"""
        print("🔄 Creating federated datasets...")
        
        # Load the main dataset
        df = pd.read_csv(base_dataset_path)
        
        # Split data among clients (simulating different organizations/users)
        client_datasets = []
        samples_per_client = len(df) // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < self.num_clients - 1 else len(df)
            
            client_df = df.iloc[start_idx:end_idx].copy()
            client_df['client_id'] = f'client_{i+1}'
            
            # Add some client-specific variations to simulate real-world differences
            if i == 1:  # Client 2 has slightly different typing patterns
                client_df['typing_speed'] *= np.random.normal(1.1, 0.05, len(client_df))
            elif i == 2:  # Client 3 has different session durations
                client_df['session_duration'] *= np.random.normal(0.9, 0.1, len(client_df))
            
            client_datasets.append(client_df)
            
            # Save client dataset
            os.makedirs("client_data", exist_ok=True)
            client_df.to_csv(f"client_data/client_{i+1}_data.csv", index=False)
            print(f"✅ Created client_{i+1}_data.csv with {len(client_df)} samples")
        
        return client_datasets
    
    def build_global_model(self, input_dim: int = 24):
        """Build the global federated model architecture"""
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
    
    def federated_averaging(self, client_weights: List[List]) -> List:
        """Perform federated averaging of client model weights"""
        if not client_weights:
            return None
        
        # Average the weights across all clients
        averaged_weights = []
        for weights_list_tuple in zip(*client_weights):
            averaged_weights.append(
                np.array([np.array(w).mean(axis=0) for w in zip(*weights_list_tuple)])
            )
        
        return averaged_weights
    
    def train_federated_model(self, client_datasets: List[pd.DataFrame]):
        """Train the federated model across multiple rounds"""
        print(f"\n🚀 Starting Federated Learning Training ({self.rounds} rounds)")
        print("=" * 60)
        
        # Initialize global model - calculate actual input dimension after filtering
        # Use the same filtering logic as in training
        exclude_cols = ['label', 'client_id']
        # Also exclude any timestamp or date columns
        for col in client_datasets[0].columns:
            if 'time' in col.lower() or 'date' in col.lower() or client_datasets[0][col].dtype == 'object':
                exclude_cols.append(col)
        
        # Get the actual features that will be used
        feature_cols = [col for col in client_datasets[0].columns if col not in exclude_cols]
        input_dim = len(feature_cols)
        print(f"📊 Model input dimension: {input_dim} features")
        print(f"📋 Features: {feature_cols}")
        self.global_model = self.build_global_model(input_dim)
        
        # Training rounds
        for round_num in range(1, self.rounds + 1):
            print(f"\n📊 Round {round_num}/{self.rounds}")
            print("-" * 40)
            
            client_weights = []
            round_losses = []
            
            # Train on each client's data
            for client_idx, client_data in enumerate(client_datasets):
                print(f"  Training on Client {client_idx + 1}...")
                
                # Prepare client data - use the same features as global model
                exclude_cols = ['label', 'client_id']
                # Also exclude any timestamp or date columns
                for col in client_data.columns:
                    if 'time' in col.lower() or 'date' in col.lower() or client_data[col].dtype == 'object':
                        exclude_cols.append(col)
                
                # Get the same features as used in global model
                feature_cols = [col for col in client_data.columns if col not in exclude_cols]
                X = client_data[feature_cols].values
                y = client_data['label'].values
                
                # Debug: Print feature info
                if client_idx == 0:  # Only print for first client
                    print(f"    📋 Using features: {feature_cols}")
                    print(f"    📊 Data shape: {X.shape}")
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Create client model (copy of global model)
                client_model = tf.keras.models.clone_model(self.global_model)
                client_model.set_weights(self.global_model.get_weights())
                client_model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.MeanAbsoluteError()], run_eagerly=False)
                
                # Train client model
                history = client_model.fit(
                    X_scaled, X_scaled,  # Autoencoder: input = output
                    epochs=5,
                    batch_size=32,
                    verbose=0,
                    validation_split=0.2
                )
                
                # Store client weights and loss
                client_weights.append(client_model.get_weights())
                round_losses.append(history.history['loss'][-1])
                
                # Save client model after training
                if round_num == self.rounds:  # Save on final round
                    os.makedirs("local_models", exist_ok=True)
                    client_model.save(f"local_models/client_{client_idx+1}_model.h5")
                    
                    # Save client scaler
                    import joblib
                    joblib.dump(scaler, f"local_models/client_{client_idx+1}_scaler.pkl")
                    print(f"    💾 Saved client_{client_idx+1} model")
                
                print(f"    Loss: {history.history['loss'][-1]:.4f}")
            
            # Federated averaging
            averaged_weights = self.federated_averaging(client_weights)
            self.global_model.set_weights(averaged_weights)
            
            # Calculate round metrics
            avg_loss = np.mean(round_losses)
            self.training_history.append({
                'round': round_num,
                'avg_loss': avg_loss,
                'client_losses': round_losses
            })
            
            print(f"  📈 Average Loss: {avg_loss:.4f}")
            print(f"  📊 Client Losses: {[f'{loss:.4f}' for loss in round_losses]}")
        
        print(f"\n✅ Federated training completed!")
        return self.global_model
    
    def evaluate_federated_model(self, test_dataset_path: str = "../anomaly-detection/enhanced_behavior_dataset.csv"):
        """Evaluate the federated model on test data"""
        print("\n🔍 Evaluating Federated Model...")
        
        # Load test data
        test_df = pd.read_csv(test_dataset_path)
        
        # Exclude non-numeric columns
        exclude_cols = ['label']
        # Also exclude any timestamp or date columns
        for col in test_df.columns:
            if 'time' in col.lower() or 'date' in col.lower() or test_df[col].dtype == 'object':
                exclude_cols.append(col)
        
        X_test = test_df.drop(exclude_cols, axis=1).values
        y_test = test_df['label'].values
        
        # Scale features
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        
        # Evaluate global model
        test_loss, test_mae = self.global_model.evaluate(X_test_scaled, X_test_scaled, verbose=0)
        
        # Calculate reconstruction errors for anomaly detection
        reconstructions = self.global_model.predict(X_test_scaled, verbose=0)
        reconstruction_errors = np.mean((X_test_scaled - reconstructions) ** 2, axis=1)
        
        # Calculate threshold for anomaly detection
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        
        # Predict anomalies
        predictions = (reconstruction_errors > threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        accuracy = np.mean(predictions == y_test)
        auc = roc_auc_score(y_test, reconstruction_errors)
        
        print(f"📊 Evaluation Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Anomaly Threshold: {threshold:.4f}")
        
        # Save evaluation results
        evaluation_results = {
            'test_loss': float(test_loss),
            'test_mae': float(test_mae),
            'accuracy': float(accuracy),
            'auc': float(auc),
            'threshold': float(threshold),
            'training_history': self.training_history
        }
        
        with open("federated_evaluation_results.json", "w") as f:
            json.dump(evaluation_results, f, indent=2)
        
        return evaluation_results
    
    def save_federated_model(self):
        """Save the federated model and components"""
        print("\n💾 Saving Federated Model...")
        
        # Create models directory
        os.makedirs("federated_models", exist_ok=True)
        
        # Save global model
        self.global_model.save("federated_models/federated_global_model.h5")
        
        # Save training history
        with open("federated_models/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        print("✅ Federated model saved successfully!")
        print("  - federated_global_model.h5")
        print("  - training_history.json")
        print("  - federated_evaluation_results.json")
    
    def plot_training_history(self):
        """Plot federated training history"""
        try:
            import matplotlib.pyplot as plt
            
            rounds = [h['round'] for h in self.training_history]
            avg_losses = [h['avg_loss'] for h in self.training_history]
            
            plt.figure(figsize=(10, 6))
            plt.plot(rounds, avg_losses, 'b-o', linewidth=2, markersize=6)
            plt.title('Federated Learning Training History', fontsize=14, fontweight='bold')
            plt.xlabel('Training Round')
            plt.ylabel('Average Loss')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig('federated_training_history.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ Training history plot saved as 'federated_training_history.png'")
            
        except ImportError:
            print("⚠️  matplotlib not available for plotting")

def main():
    """Main function to run federated learning"""
    print("🌐 Federated Anomaly Detection System")
    print("=" * 50)
    
    # Initialize federated server
    server = FederatedAnomalyServer(num_clients=3, rounds=10)
    
    # Create federated datasets
    client_datasets = server.create_federated_dataset()
    
    # Train federated model
    global_model = server.train_federated_model(client_datasets)
    
    # Evaluate model
    evaluation_results = server.evaluate_federated_model()
    
    # Save model
    server.save_federated_model()
    
    # Plot training history
    server.plot_training_history()
    
    print(f"\n🎉 Federated Learning Complete!")
    print(f"📊 Final AUC: {evaluation_results['auc']:.4f}")
    print(f"📊 Final Accuracy: {evaluation_results['accuracy']:.4f}")

if __name__ == "__main__":
    main()
