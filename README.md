# Samsung EnnovateX 2025 AI Challenge Submission

Problem Statement - On-Device Multi-Agent System for Behavior-Based Anomaly & Fraud Detection
Team name - Rishabh and Raghav
Team members (Names) - Raghav Bagai, Rishabh Garg
Demo Video Link - 

# Anomaly Detection Project - Technical Documentation

## Overview

The Anomaly Detection Project is an on-device multi-agent system designed for behavior-based anomaly and fraud detection. This project implements advanced machine learning techniques to identify unusual patterns and potential security threats through behavioral analysis.

### Team Information
- **Team Name**: Rishabh and Raghav
- **Team Members**: 
  - Raghav Bagai
  - Rishabh Garg

## Table of Contents

1. [Project Architecture](#project-architecture)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
5. [API Reference](#api-reference)
6. [File Structure](#file-structure)
7. [Configuration](#configuration)
8. [Contributing](#contributing)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

## Project Architecture

The system follows a modular architecture with the following key components:

- **Dataset Generation Module**: Creates rich behavioral datasets for training
- **Model Training Module**: Implements both basic and advanced behavior models
- **Real-time Inference Engine**: Provides live anomaly detection capabilities
- **Batch Processing System**: Handles large-scale data analysis
- **On-device Analysis**: Optimized for edge computing scenarios

## Features

### Core Capabilities
- **Behavior Model Training**: Train models with rich datasets for accurate behavior pattern recognition
- **Real-time Anomaly Inference**: Detect anomalies as they occur with minimal latency
- **Batch Processing**: Analyze large volumes of historical data efficiently
- **On-device Behavior Analysis**: Perform analysis locally without cloud dependency
- **Multi-agent System**: Coordinate multiple detection agents for comprehensive coverage

### Key Benefits
- Low latency detection
- Privacy-preserving on-device processing
- Scalable architecture
- Rich dataset support
- Fraud detection capabilities

## Installation

### Prerequisites
- Python 3.7+
- Virtual environment support
- Git

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RebyTK/anomaly-detection-project.git
   cd anomaly-detection-project
   ```

2. **Create and activate virtual environment**:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate (Linux/Mac)
   source venv/bin/activate
   
   # Activate (Windows)
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### System Requirements

#### Recommended Dependencies
```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
torch>=1.9.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## Usage

### Quick Start

The project provides several entry points for different use cases:

#### 1. Dataset Generation
Generate rich behavioral datasets for model training:
```bash
python generate_rich_behavior_dataset.py
```

#### 2. Model Training

**Basic Behavior Model**:
```bash
python train_behavior_model.py
```

**Advanced Rich Behavior Model**:
```bash
python train_behavior_rich_model.py
```

#### 3. Inference

**Real-time Inference**:
```bash
python inference_rich_behavior.py
```

**Batch Processing**:
```bash
python infer_rich_behavior_batch.py
```

### Workflow Example

```bash
# 1. Generate training data
python generate_rich_behavior_dataset.py

# 2. Train the behavior model
python train_behavior_rich_model.py

# 3. Run real-time anomaly detection
python inference_rich_behavior.py
```

## API Reference

### Core Modules

#### Dataset Generation (`generate_rich_behavior_dataset.py`)
Generates comprehensive behavioral datasets with various patterns and anomalies.

**Key Functions**:
- `generate_normal_behavior()`: Creates baseline behavior patterns
- `inject_anomalies()`: Introduces various types of anomalies
- `save_dataset()`: Exports generated data for training

#### Model Training (`train_behavior_rich_model.py`)
Implements advanced machine learning models for behavior analysis.

**Key Classes**:
- `BehaviorModel`: Main model class for behavior pattern recognition
- `FeatureExtractor`: Extracts relevant features from raw data
- `ModelTrainer`: Handles training process and optimization

#### Real-time Inference (`inference_rich_behavior.py`)
Provides live anomaly detection capabilities.

**Key Functions**:
- `load_model()`: Loads trained behavior model
- `process_real_time_data()`: Analyzes incoming data streams
- `detect_anomalies()`: Identifies suspicious patterns

#### Batch Processing (`infer_rich_behavior_batch.py`)
Handles large-scale data analysis for historical data review.

**Key Features**:
- Efficient batch processing algorithms
- Parallel processing support
- Result aggregation and reporting

## File Structure

```
anomaly-detection-project/
├── generate_rich_behavior_dataset.py    # Dataset generation module
├── train_behavior_model.py             # Basic behavior model training
├── train_behavior_rich_model.py        # Advanced behavior model training
├── inference_rich_behavior.py          # Real-time inference engine
├── infer_rich_behavior_batch.py        # Batch processing system
├── ondevice-behaviour/                 # On-device analysis module
│   ├── __init__.py
│   ├── core/                          # Core analysis algorithms
│   ├── models/                        # Model definitions
│   └── utils/                         # Utility functions
├── requirements.txt                    # Project dependencies
├── README.md                          # Project overview
├── LICENSE                            # License information
└── .gitignore                         # Git ignore rules
```

## Configuration

### Environment Variables
Create a `.env` file for configuration:

```env
# Model Configuration
MODEL_PATH=./models/
DATASET_PATH=./data/
LOG_LEVEL=INFO

# Processing Configuration
BATCH_SIZE=1000
INFERENCE_THRESHOLD=0.85
REAL_TIME_BUFFER=100

# On-device Configuration
MAX_MEMORY_USAGE=512MB
PROCESSING_THREADS=4
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

## Future Enhancements

### Planned Features
- **Deep Learning Integration**: Advanced neural network architectures
- **Federated Learning**: Distributed training across multiple devices
- **Explainable AI**: Interpretability features for anomaly explanations
- **Auto-scaling**: Dynamic resource allocation based on workload
- **Advanced Visualization**: Interactive dashboards for monitoring

### Research Areas
- Adversarial anomaly detection
- Zero-shot learning for unknown anomaly types
- Transfer learning across different domains
- Real-time model updating and adaptation

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.

## Support and Contact

- **Project Repository**: [https://github.com/RebyTK/anomaly-detection-project](https://github.com/RebyTK/anomaly-detection-project)
- **Issue Tracker**: [GitHub Issues](https://github.com/RebyTK/anomaly-detection-project/issues)
- **Documentation**: This file serves as the primary technical documentation

## Acknowledgments

- Research community for anomaly detection methodologies
- Open-source libraries and frameworks used in this project
- Contributors and collaborators

---

*This documentation is maintained alongside the project. For the most up-to-date information, please refer to the repository and commit history.*

