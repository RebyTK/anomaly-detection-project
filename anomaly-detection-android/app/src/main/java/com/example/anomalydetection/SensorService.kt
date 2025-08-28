package com.example.anomalydetection

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import kotlinx.coroutines.*
import kotlin.random.Random

class SensorService(private val context: Context) : SensorEventListener {
    
    private val sensorManager: SensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val accelerometer: Sensor? = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    
    private var isRunning = false
    private var dataCallback: ((DataPoint) -> Unit)? = null
    private var job: Job? = null
    
    fun startDataCollection(callback: (DataPoint) -> Unit) {
        dataCallback = callback
        isRunning = true
        
        // Start real sensor if available
        accelerometer?.let {
            sensorManager.registerListener(this, it, SensorManager.SENSOR_DELAY_NORMAL)
        }
        
        // Start simulated data generation as fallback
        job = CoroutineScope(Dispatchers.Default).launch {
            while (isRunning) {
                val simulatedData = generateSimulatedData()
                dataCallback?.invoke(simulatedData)
                delay(1000) // Generate data every second
            }
        }
    }
    
    fun stopDataCollection() {
        isRunning = false
        sensorManager.unregisterListener(this)
        job?.cancel()
        dataCallback = null
    }
    
    private fun generateSimulatedData(): DataPoint {
        // Generate realistic sensor data with occasional anomalies
        val baseValue = 9.8 // Normal gravity
        val noise = Random.nextDouble(-0.5, 0.5)
        
        // 5% chance of anomaly
        val anomaly = if (Random.nextDouble() < 0.05) {
            Random.nextDouble(-5.0, 5.0)
        } else {
            0.0
        }
        
        val value = baseValue + noise + anomaly
        return DataPoint(value, System.currentTimeMillis())
    }
    
    override fun onSensorChanged(event: SensorEvent?) {
        event?.let {
            if (it.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                val magnitude = Math.sqrt(
                    (it.values[0] * it.values[0] + 
                     it.values[1] * it.values[1] + 
                     it.values[2] * it.values[2]).toDouble()
                )
                val dataPoint = DataPoint(magnitude, System.currentTimeMillis())
                dataCallback?.invoke(dataPoint)
            }
        }
    }
    
    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {
        // Not needed for this implementation
    }
}
