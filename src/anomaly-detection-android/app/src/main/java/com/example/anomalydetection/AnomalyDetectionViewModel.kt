package com.example.anomalydetection

import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs
import kotlin.math.sqrt

class AnomalyDetectionViewModel : ViewModel() {

    private val _anomalyData = MutableLiveData<List<Anomaly>>()
    val anomalyData: LiveData<List<Anomaly>> get() = _anomalyData

    private val _currentSensorData = MutableLiveData<DataPoint>()
    val currentSensorData: LiveData<DataPoint> get() = _currentSensorData

    private val _isDetectionRunning = MutableLiveData<Boolean>()
    val isDetectionRunning: LiveData<Boolean> get() = _isDetectionRunning

    private val _lastUpdateTime = MutableLiveData<String>()
    val lastUpdateTime: LiveData<String> get() = _lastUpdateTime

    private val _status = MutableLiveData<String>()
    val status: LiveData<String> get() = _status

    private var dataBuffer = mutableListOf<DataPoint>()
    private var threshold = 0.7 // Default threshold
    private val bufferSize = 50 // Keep last 50 data points for analysis

    init {
        _anomalyData.value = emptyList()
        _isDetectionRunning.value = false
        _status.value = "Ready"
        _lastUpdateTime.value = "Never"
    }

    fun setThreshold(newThreshold: Float) {
        threshold = newThreshold.toDouble() / 100.0
    }

    fun startDetection() {
        _isDetectionRunning.value = true
        _status.value = "Detection Active"
        updateLastUpdateTime()
        
        viewModelScope.launch {
            while (_isDetectionRunning.value == true) {
                delay(1000) // Update every second
                updateLastUpdateTime()
            }
        }
    }

    fun stopDetection() {
        _isDetectionRunning.value = false
        _status.value = "Detection Stopped"
        updateLastUpdateTime()
    }

    fun processSensorData(dataPoint: DataPoint) {
        _currentSensorData.value = dataPoint
        updateLastUpdateTime()
        
        // Add to buffer
        dataBuffer.add(dataPoint)
        if (dataBuffer.size > bufferSize) {
            dataBuffer.removeAt(0)
        }
        
        // Check for anomalies
        if (dataBuffer.size >= 10) { // Need minimum data for analysis
            val anomaly = detectAnomaly(dataPoint)
            if (anomaly != null) {
                val currentList = _anomalyData.value?.toMutableList() ?: mutableListOf()
                currentList.add(anomaly)
                _anomalyData.value = currentList
                _status.value = "Anomaly Detected!"
            } else {
                _status.value = "No Anomaly"
            }
        }
    }

    private fun detectAnomaly(dataPoint: DataPoint): Anomaly? {
        if (dataBuffer.size < 10) return null
        
        val values = dataBuffer.map { it.value }
        val mean = values.average()
        val variance = values.map { (it - mean) * (it - mean) }.average()
        val stdDev = sqrt(variance)
        
        // Z-score based anomaly detection
        val zScore = abs(dataPoint.value - mean) / stdDev
        
        // Statistical threshold (3 standard deviations)
        val statisticalThreshold = 3.0
        
        // Combined threshold with user preference
        val combinedThreshold = statisticalThreshold * (1 + threshold)
        
        return if (zScore > combinedThreshold) {
            val reason = when {
                zScore > 5.0 -> "Extreme outlier"
                zScore > 4.0 -> "Major deviation"
                zScore > 3.0 -> "Statistical outlier"
                else -> "Slight anomaly"
            }
            Anomaly(dataPoint, reason)
        } else {
            null
        }
    }

    private fun updateLastUpdateTime() {
        val currentTime = java.text.SimpleDateFormat("HH:mm:ss", java.util.Locale.getDefault())
            .format(java.util.Date())
        _lastUpdateTime.value = currentTime
    }

    fun clearAnomalies() {
        _anomalyData.value = emptyList()
        _status.value = "Anomalies Cleared"
        updateLastUpdateTime()
    }

    fun getDataBufferSize(): Int = dataBuffer.size

    fun getCurrentThreshold(): Double = threshold

    override fun onCleared() {
        super.onCleared()
        // Clean up resources if needed
    }
}

data class DataPoint(val value: Double, val timestamp: Long)
data class Anomaly(val dataPoint: DataPoint, val reason: String)