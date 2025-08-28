# Anomaly Detection Android App

A comprehensive Android application for real-time anomaly detection using device sensors and advanced statistical algorithms.

## Features

### 🚀 Core Functionality
- **Real-time Sensor Monitoring**: Continuously monitors device accelerometer data
- **Advanced Anomaly Detection**: Uses Z-score based statistical analysis to detect outliers
- **Configurable Thresholds**: Adjustable sensitivity levels for anomaly detection
- **Live Data Visualization**: Real-time display of sensor readings and detection results

### 📱 User Interface
- **Modern Material Design**: Clean, intuitive interface with card-based layout
- **Status Monitoring**: Real-time status updates and anomaly alerts
- **Threshold Control**: Interactive seekbar for adjusting detection sensitivity
- **Results Display**: Comprehensive list of detected anomalies with timestamps

### 🔧 Technical Features
- **MVVM Architecture**: Clean separation of concerns using ViewModel and LiveData
- **Sensor Integration**: Native Android sensor API integration with fallback simulation
- **Coroutine Support**: Asynchronous data processing for smooth performance
- **Permission Handling**: Proper Android permission management for sensor access

## Screenshots

The app features a clean, card-based interface with:
- Status monitoring card
- Control buttons for starting/stopping detection
- Real-time sensor data display
- Configurable threshold controls
- Comprehensive results display

## Architecture

### Components
1. **MainActivity**: Main UI controller and sensor service coordinator
2. **AnomalyDetectionViewModel**: Business logic and data management
3. **SensorService**: Sensor data collection and processing
4. **AnomalyAdapter**: RecyclerView adapter for anomaly display

### Data Flow
```
Sensor Data → SensorService → ViewModel → UI Updates
                ↓
         Anomaly Detection Algorithm
                ↓
         Results → RecyclerView Display
```

## Installation

### Prerequisites
- Android Studio Arctic Fox or later
- Android SDK 21+ (Android 5.0 Lollipop)
- Kotlin support enabled

### Build Steps
1. Clone the repository
2. Open the project in Android Studio
3. Sync Gradle files
4. Build and run on device/emulator

### Permissions
The app requires the following permissions:
- `ACTIVITY_RECOGNITION`: For sensor access
- `WAKE_LOCK`: For continuous sensor monitoring

## Usage

### Starting Detection
1. Launch the app
2. Adjust threshold sensitivity using the seekbar
3. Tap "Start Detection" to begin monitoring
4. The app will continuously monitor sensor data

### Monitoring Results
- **Status Card**: Shows current detection status
- **Sensor Data**: Displays real-time accelerometer readings
- **Threshold Control**: Adjust detection sensitivity (0-100%)
- **Results**: Lists all detected anomalies with details

### Stopping Detection
- Tap "Stop Detection" to halt monitoring
- Sensor data collection will stop immediately

## Anomaly Detection Algorithm

### Statistical Method
The app uses a **Z-score based approach** for anomaly detection:

1. **Data Collection**: Maintains a rolling buffer of recent sensor readings
2. **Statistical Analysis**: Calculates mean and standard deviation
3. **Threshold Comparison**: Compares current readings against statistical thresholds
4. **Anomaly Classification**: Categorizes anomalies by severity level

### Threshold Levels
- **User Configurable**: 0-100% sensitivity adjustment
- **Statistical Baseline**: 3 standard deviations as base threshold
- **Combined Logic**: User preference × statistical threshold

### Anomaly Categories
- **Slight Anomaly**: Minor statistical deviation
- **Statistical Outlier**: Significant deviation (>3σ)
- **Major Deviation**: Large deviation (>4σ)
- **Extreme Outlier**: Very large deviation (>5σ)

## Technical Implementation

### Dependencies
```gradle
implementation 'androidx.core:core-ktx:1.12.0'
implementation 'androidx.lifecycle:lifecycle-viewmodel-ktx:2.7.0'
implementation 'androidx.recyclerview:recyclerview:1.3.2'
implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
```

### Key Technologies
- **Kotlin**: Modern Android development language
- **MVVM**: Architecture pattern for clean code structure
- **LiveData**: Reactive data handling
- **Coroutines**: Asynchronous programming
- **RecyclerView**: Efficient list display

## Customization

### Adding New Sensors
1. Extend `SensorService` class
2. Add new sensor types to the service
3. Update data models as needed
4. Modify UI to display new sensor data

### Algorithm Modifications
1. Update `detectAnomaly()` method in ViewModel
2. Implement new statistical methods
3. Add custom threshold logic
4. Test with various data patterns

### UI Enhancements
1. Modify layout files in `res/layout/`
2. Update color schemes in `res/values/colors.xml`
3. Add new string resources
4. Implement custom animations

## Performance Considerations

### Memory Management
- Rolling data buffer prevents memory leaks
- Efficient RecyclerView implementation
- Proper lifecycle management

### Battery Optimization
- Sensor sampling rate optimization
- Efficient coroutine usage
- Minimal UI updates

### Data Processing
- Statistical calculations optimized for mobile
- Efficient anomaly detection algorithms
- Minimal computational overhead

## Testing

### Unit Tests
- ViewModel logic testing
- Algorithm validation
- Data processing verification

### Integration Tests
- Sensor service integration
- UI component testing
- End-to-end workflow testing

### Performance Testing
- Memory usage monitoring
- Battery consumption analysis
- Sensor data processing speed

## Troubleshooting

### Common Issues
1. **Sensor Not Working**: Check device sensor availability
2. **Permission Denied**: Grant required permissions in settings
3. **App Crashes**: Verify device compatibility (API 21+)
4. **No Anomalies**: Adjust threshold sensitivity

### Debug Mode
Enable logging for detailed debugging:
- Sensor data logging
- Algorithm execution tracking
- Performance metrics

## Future Enhancements

### Planned Features
- **Machine Learning**: Integration with TensorFlow Lite
- **Cloud Sync**: Remote anomaly storage and analysis
- **Advanced Visualization**: Charts and graphs for data analysis
- **Multi-Sensor Support**: Gyroscope, magnetometer integration
- **Export Functionality**: Data export to CSV/JSON formats

### Performance Improvements
- **GPU Acceleration**: OpenGL-based data visualization
- **Native Code**: C++ implementation for critical algorithms
- **Optimized Algorithms**: More efficient statistical methods

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add appropriate tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the code examples

---

**Built with ❤️ for the Android community**