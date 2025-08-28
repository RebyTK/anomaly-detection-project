# 🌐 Federated Learning for Anomaly Detection

This directory contains a comprehensive federated learning system for anomaly detection that integrates seamlessly with your existing centralized anomaly detection system.

## 📁 File Structure

```
federated-learning/
├── federated_anomaly_server.py    # 🌐 Main federated learning server
├── federated_anomaly_client.py    # 🏠 Enhanced client implementation  
├── federated_inference.py         # 🔍 Integrated inference system
└── README.md                      # 📚 This documentation
```

## 🚀 Quick Start

### 1. Run Federated Training

```bash
cd federated-learning
python federated_anomaly_server.py
```

This will:
- ✅ Split your main dataset into client-specific datasets
- ✅ Train federated models across multiple rounds
- ✅ Save the global federated model
- ✅ Generate training history and evaluation results
- ✅ Create visualizations of training progress

### 2. Test Individual Clients

```bash
python federated_anomaly_client.py
```

This will:
- ✅ Load client-specific data
- ✅ Train local models
- ✅ Evaluate performance
- ✅ Test anomaly detection
- ✅ Save local models and scalers

### 3. Run Federated Inference

```bash
python federated_inference.py
```

This will:
- ✅ Load federated and client models
- ✅ Perform anomaly detection
- ✅ Compare with centralized models
- ✅ Generate comprehensive visualizations
- ✅ Save inference results

## 🔧 System Architecture

### Federated Learning Process

1. **📊 Data Distribution**: Main dataset is split among multiple clients
2. **🏋️ Local Training**: Each client trains on their local data
3. **🔄 Model Aggregation**: Server averages model weights from all clients
4. **📤 Global Update**: Updated global model is distributed back to clients
5. **🔄 Iteration**: Process repeats for multiple rounds

### Key Components

#### 🌐 FederatedAnomalyServer
- **Purpose**: Orchestrates the entire federated learning process
- **Features**:
  - 🎯 Dataset splitting and client simulation
  - ⚖️ Federated averaging algorithm
  - 🔄 Multi-round training coordination
  - 📊 Model evaluation and comparison
  - 📈 Training history tracking
  - 💾 Automatic model saving

#### 🏠 FederatedAnomalyClient
- **Purpose**: Individual client implementation with privacy protection
- **Features**:
  - 📂 Local data loading and preprocessing
  - 🏋️ Local model training
  - 📊 Model evaluation and saving
  - 🔍 Real-time anomaly detection
  - 🔒 Privacy-preserving operations

#### 🔍 FederatedAnomalyInference
- **Purpose**: Integrated inference system with ensemble decisions
- **Features**:
  - 📥 Load both federated and client models
  - 🎯 Ensemble decision making
  - 🔄 Comparison with centralized models
  - 📊 Comprehensive visualization
  - 💾 Results saving and analysis

## 📊 Features

### 🔒 Privacy-Preserving
- **🚫 No Data Sharing**: Raw data never leaves client devices
- **📦 Model-Only Communication**: Only model weights are exchanged
- **🏠 Local Processing**: All sensitive data stays local
- **🔐 Compliance Ready**: Meets data privacy regulations (GDPR, CCPA)

### 🎯 Enhanced Performance
- **🌍 Diverse Data**: Learns from multiple data sources
- **🛡️ Robust Models**: Better generalization across different patterns
- **🎲 Ensemble Decisions**: Combines multiple model predictions
- **📈 Adaptive Learning**: Continuously improves with new data

### 📈 Comprehensive Analytics
- **📊 Training History**: Track performance across rounds
- **🔄 Model Comparison**: Compare federated vs centralized approaches
- **📈 Visualization**: Rich plots and analysis tools
- **📋 Detailed Reports**: Comprehensive evaluation metrics

## 🛠️ Configuration

### Server Configuration

```python
# In federated_anomaly_server.py
server = FederatedAnomalyServer(
    num_clients=3,    # Number of federated clients
    rounds=10         # Number of training rounds
)
```

### Client Configuration

```python
# In federated_anomaly_client.py
client = FederatedAnomalyClient(
    client_id="client_1",
    data_path="client_data/client_1_data.csv"
)
```

### Model Architecture

The federated system uses the same autoencoder architecture as your centralized system:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(24,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(24, activation='linear')
])
```

## 📈 Performance Metrics

### Training Metrics
- **📉 Loss**: Mean squared error across training rounds
- **📊 MAE**: Mean absolute error for reconstruction quality
- **🔄 Convergence**: Training stability across rounds
- **⚡ Speed**: Training time per round

### Inference Metrics
- **🎯 Accuracy**: Overall anomaly detection accuracy
- **📈 AUC**: Area under ROC curve
- **📋 Precision/Recall**: Detailed classification metrics
- **🤝 Model Agreement**: Consistency between federated and centralized models

## 🔄 Workflow

### Training Workflow
1. **📂 Data Preparation**: Split main dataset into client datasets
2. **🏗️ Model Initialization**: Create global model architecture
3. **🔄 Federated Rounds**:
   - 📤 Distribute global model to clients
   - 🏋️ Train on local data
   - 📥 Collect updated weights
   - ⚖️ Average weights to create new global model
4. **📊 Evaluation**: Test global model performance
5. **💾 Saving**: Store models and training history

### Inference Workflow
1. **📥 Model Loading**: Load federated and client models
2. **🔧 Feature Preparation**: Process input data
3. **🎯 Multi-Model Prediction**: Get predictions from all models
4. **🤝 Ensemble Decision**: Combine predictions for final result
5. **📊 Analysis**: Generate reports and visualizations

## 📊 Output Files

### Training Outputs
- `federated_models/federated_global_model.h5` - Global federated model
- `federated_models/training_history.json` - Training metrics
- `federated_evaluation_results.json` - Model evaluation results
- `federated_training_history.png` - Training visualization

### Client Outputs
- `local_models/client_X_model.h5` - Individual client models
- `local_models/client_X_scaler.pkl` - Client-specific scalers
- `local_models/client_X_history.json` - Client training history

### Inference Outputs
- `federated_inference_results.csv` - Batch inference results
- `federated_inference_analysis.png` - Inference visualizations

## 🔍 Usage Examples

### Single Sample Detection

```python
from federated_inference import FederatedAnomalyInference

# Initialize inference system
inference = FederatedAnomalyInference()
inference.load_federated_model()
inference.load_client_models()

# Detect anomaly
sample_data = {
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

result = inference.detect_anomaly_federated(sample_data, "client_1")
print(f"Anomaly: {result['ensemble_decision']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Confidence: {result['federated_scores']['confidence']:.4f}")
```

### Batch Processing

```python
import pandas as pd

# Load test data
test_data = pd.read_csv("../anomaly-detection/enhanced_behavior_dataset.csv")

# Process batch
results = inference.batch_federated_inference(test_data, "client_1")

# Save results
inference.save_inference_results(results)
inference.plot_federated_results(results)
```

## 🆚 Comparison with Centralized System

### Advantages of Federated Learning
- **🔒 Privacy**: No raw data sharing required
- **🌍 Scalability**: Can handle distributed data sources
- **🎯 Diversity**: Learns from varied data patterns
- **📋 Compliance**: Meets data privacy regulations
- **🛡️ Security**: Reduced attack surface

### Advantages of Centralized System
- **⚡ Simplicity**: Easier to implement and maintain
- **📊 Performance**: Potentially better with centralized data
- **🔄 Consistency**: Single model for all predictions
- **🚀 Speed**: Faster training and inference

## 🚨 Troubleshooting

### Common Issues

1. **📁 Model Loading Errors**
   - ✅ Ensure federated training has been completed
   - ✅ Check file paths and permissions
   - ✅ Verify model file integrity

2. **📂 Data Path Issues**
   - ✅ Confirm dataset exists in expected location
   - ✅ Check relative paths are correct
   - ✅ Ensure data format matches expectations

3. **💾 Memory Issues**
   - ✅ Reduce batch size or number of clients
   - ✅ Use smaller model architecture
   - ✅ Process data in smaller chunks

### Performance Optimization

1. **⚡ Training Speed**
   - ✅ Reduce number of training rounds
   - ✅ Use smaller local epochs
   - ✅ Optimize batch sizes

2. **🚀 Inference Speed**
   - ✅ Load models once and reuse
   - ✅ Use batch processing
   - ✅ Optimize feature preparation

## 📚 Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn joblib matplotlib seaborn
```

## 🤝 Integration with Main System

The federated learning system is designed to work alongside your existing centralized anomaly detection system:

- **🏗️ Shared Architecture**: Uses same model architecture and feature engineering
- **📊 Compatible Data**: Works with existing datasets and data formats
- **📈 Comparable Metrics**: Provides same evaluation metrics for comparison
- **🔌 Unified Interface**: Similar API for easy integration

## 🔮 Future Enhancements

- **🔐 Secure Aggregation**: Add cryptographic protocols for secure weight aggregation
- **🌍 Heterogeneous Data**: Support for different data distributions across clients
- **🔄 Dynamic Client Management**: Add/remove clients during training
- **⚖️ Advanced Aggregation**: Implement weighted averaging based on data quality
- **⏱️ Real-time Federated Learning**: Continuous model updates from streaming data
- **🔒 Differential Privacy**: Add privacy guarantees to federated learning
- **🌐 Cross-Device Learning**: Support for mobile and IoT devices

## 📄 License

This federated learning system is part of the anomaly detection project and follows the same licensing terms.

---

## 🎯 **Quick Commands Summary**

```bash
# Train federated models
cd federated-learning
python federated_anomaly_server.py

# Test individual clients
python federated_anomaly_client.py

# Run federated inference
python federated_inference.py
```

**🌐 Federated Learning + 🔍 Anomaly Detection = Privacy-Preserving Security**

*Your data stays local, your security gets global!* 🛡️
