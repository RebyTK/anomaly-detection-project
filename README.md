# Samsung EnnovateX 2025 AI Challenge Submission

Problem Statement - On-Device Multi-Agent System for Behavior-Based Anomaly & Fraud Detection
Team name - Rishabh and Raghav
Team members (Names) - Raghav Bagai, Rishabh Garg
Demo Video Link - 

# Anomaly Detection Project - Technical Documentation

## Overview

This project implements a robust anomaly detection system using both centralized and federated learning approaches. It is designed to detect abnormal user behavior in time-series and tabular data, supporting privacy-preserving distributed training across multiple clients.


#### Recommended Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
```


## File Structure

```
anomaly-detection-project/
│
├── anomaly-detection/         # Centralized anomaly detection code and models
│   ├── data_generator.py      # Synthetic data generator
│   ├── enhanced_training_model.py  # Centralized model training and evaluation
│   ├── enhanced_behavior_dataset.csv # Generated dataset
│   ├── ...                    # Model files, results, and utilities
│
├── federated-learning/        # Federated learning implementation
│   ├── federated_anomaly_client.py   # Client-side training logic
│   ├── federated_anomaly_server.py   # Server-side aggregation logic
│   ├── federated_inference.py        # Federated inference and evaluation
│   ├── client_data/           # Simulated client datasets
│   ├── federated_models/      # Global federated models
│   ├── local_models/          # Individual client models
│   ├── ...                    # Results, plots, and configs
│
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview and setup
└── .gitignore                 # Git ignore rules
```

## Key Components

1. Data Generation (data_generator.py)
Generates synthetic user behavior data with realistic and anomalous patterns.
Supports feature engineering and temporal feature addition.
2. Centralized Model Training (enhanced_training_model.py)
Trains anomaly detection models (Isolation Forest, VAE, LSTM, etc.) on the full dataset.
Evaluates models using metrics like accuracy, ROC-AUC, and confusion matrix.
Saves trained models and scalers for inference.
3. Federated Learning (federated-learning)
Client (federated_anomaly_client.py): Loads local data, trains local models, and shares model weights.
Server (federated_anomaly_server.py): Aggregates client weights using federated averaging to update the global model.
Inference (federated_inference.py): Loads global and client models for distributed anomaly detection and evaluation.

### Setup Instructions
```
pip install -r requirements.txt

python anomaly-detection/data_generator.py

python anomaly-detection/enhanced_training_model.py

python federated-learning/federated_anomaly_server.py

python federated-learning/federated_inference.py
```

### Model Parameters
Configure model behavior in `config.py`:

```python
# Model hyperparameters
LEARNING_RATE = 0.001
EPOCHS = 100
HIDDEN_LAYERS = [128, 64, 32]
DROPOUT_RATE = 0.3

# Anomaly detection thresholds
ANOMALY_THRESHOLD = 0.8
CONFIDENCE_THRESHOLD = 0.9
```

## Development Guidelines

### Code Structure
- Follow PEP 8 style guidelines
- Use type hints for function parameters
- Include comprehensive docstrings
- Implement unit tests for critical functions

### Testing
Run the test suite:
```bash
python -m pytest tests/
```

### Adding New Features
1. Create feature branch: `git checkout -b feature/new-feature`
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## Performance Optimization

### On-device Optimization
- **Memory Management**: Efficient data structures and garbage collection
- **CPU Optimization**: Multi-threading for parallel processing
- **Model Compression**: Quantization and pruning for reduced model size
- **Caching**: Intelligent caching strategies for frequently accessed data

### Real-time Performance
- **Streaming Processing**: Handle continuous data streams efficiently
- **Buffer Management**: Optimize buffer sizes for low latency
- **Parallel Processing**: Utilize multiple cores for concurrent analysis

## Troubleshooting

### Common Issues

#### Installation Problems
**Issue**: Dependencies not installing correctly
**Solution**: 
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

#### Memory Issues
**Issue**: Out of memory during batch processing
**Solution**: Reduce batch size in configuration or use data chunking

#### Model Performance
**Issue**: Poor anomaly detection accuracy
**Solution**: 
- Increase training data size
- Tune hyperparameters
- Validate feature selection

### Debug Mode
Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python inference_rich_behavior.py
```

## Contributing

We welcome contributions to improve the anomaly detection system!

### How to Contribute

1. **Fork the repository**
2. **Create feature branch**:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes and commit**:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to branch**:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open Pull Request**

### Contribution Guidelines
- Ensure code follows project standards
- Add tests for new features
- Update documentation as needed
- Include clear commit messages

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

## Support and Contact

- **Project Repository**: [https://github.com/RebyTK/anomaly-detection-project](https://github.com/RebyTK/anomaly-detection-project)
- **Issue Tracker**: [GitHub Issues](https://github.com/RebyTK/anomaly-detection-project/issues)
- **Documentation**: This file serves as the primary technical documentation


---

*This documentation is maintained alongside the project. For the most up-to-date information, please refer to the repository and commit history.*

