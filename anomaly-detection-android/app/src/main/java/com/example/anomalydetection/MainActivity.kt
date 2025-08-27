package com.example.anomalydetection

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView

class MainActivity : AppCompatActivity() {

    private lateinit var viewModel: AnomalyDetectionViewModel
    private lateinit var sensorService: SensorService
    private lateinit var anomalyAdapter: AnomalyAdapter

    // UI Components
    private lateinit var startDetectionButton: Button
    private lateinit var stopDetectionButton: Button
    private lateinit var statusTextView: TextView
    private lateinit var lastUpdateTextView: TextView
    private lateinit var sensorDataTextView: TextView
    private lateinit var thresholdSeekBar: SeekBar
    private lateinit var thresholdValueTextView: TextView
    private lateinit var resultTextView: TextView
    private lateinit var anomaliesRecyclerView: RecyclerView

    companion object {
        private const val SENSOR_PERMISSION_REQUEST = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        initializeViews()
        setupViewModel()
        setupSensorService()
        setupRecyclerView()
        setupListeners()
        checkPermissions()
    }

    private fun initializeViews() {
        startDetectionButton = findViewById(R.id.startDetectionButton)
        stopDetectionButton = findViewById(R.id.stopDetectionButton)
        statusTextView = findViewById(R.id.statusTextView)
        lastUpdateTextView = findViewById(R.id.lastUpdateTextView)
        sensorDataTextView = findViewById(R.id.sensorDataTextView)
        thresholdSeekBar = findViewById(R.id.thresholdSeekBar)
        thresholdValueTextView = findViewById(R.id.thresholdValueTextView)
        resultTextView = findViewById(R.id.resultTextView)
        anomaliesRecyclerView = findViewById(R.id.anomaliesRecyclerView)
    }

    private fun setupViewModel() {
        viewModel = ViewModelProvider(this).get(AnomalyDetectionViewModel::class.java)
        
        // Observe ViewModel data
        viewModel.status.observe(this) { status ->
            statusTextView.text = status
            updateStatusColor(status)
        }
        
        viewModel.lastUpdateTime.observe(this) { time ->
            lastUpdateTextView.text = "Last Update: $time"
        }
        
        viewModel.currentSensorData.observe(this) { dataPoint ->
            sensorDataTextView.text = "Current: ${String.format("%.2f", dataPoint.value)} m/s²"
        }
        
        viewModel.anomalyData.observe(this) { anomalies ->
            anomalyAdapter.updateAnomalies(anomalies)
            updateResultText(anomalies.size)
        }
        
        viewModel.isDetectionRunning.observe(this) { isRunning ->
            startDetectionButton.isEnabled = !isRunning
            stopDetectionButton.isEnabled = isRunning
        }
    }

    private fun setupSensorService() {
        sensorService = SensorService(this)
    }

    private fun setupRecyclerView() {
        anomalyAdapter = AnomalyAdapter()
        anomaliesRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = anomalyAdapter
        }
    }

    private fun setupListeners() {
        startDetectionButton.setOnClickListener {
            startAnomalyDetection()
        }
        
        stopDetectionButton.setOnClickListener {
            stopAnomalyDetection()
        }
        
        thresholdSeekBar.setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                thresholdValueTextView.text = "Threshold: ${progress}%"
                viewModel.setThreshold(progress.toFloat())
            }
            
            override fun onStartTrackingTouch(seekBar: SeekBar?) {}
            override fun onStopTrackingTouch(seekBar: SeekBar?) {}
        })
    }

    private fun startAnomalyDetection() {
        viewModel.startDetection()
        sensorService.startDataCollection { dataPoint ->
            runOnUiThread {
                viewModel.processSensorData(dataPoint)
            }
        }
    }

    private fun stopAnomalyDetection() {
        viewModel.stopDetection()
        sensorService.stopDataCollection()
    }

    private fun updateStatusColor(status: String) {
        val colorRes = when {
            status.contains("Anomaly") -> R.color.error_color
            status.contains("Active") -> R.color.success_color
            status.contains("Stopped") -> R.color.warning_color
            else -> R.color.text_secondary
        }
        statusTextView.setTextColor(ContextCompat.getColor(this, colorRes))
    }

    private fun updateResultText(anomalyCount: Int) {
        resultTextView.text = when {
            anomalyCount == 0 -> "No anomalies detected yet"
            anomalyCount == 1 -> "1 anomaly detected"
            else -> "$anomalyCount anomalies detected"
        }
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.ACTIVITY_RECOGNITION
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.ACTIVITY_RECOGNITION),
                SENSOR_PERMISSION_REQUEST
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == SENSOR_PERMISSION_REQUEST) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // Permission granted, can start sensor service
            } else {
                // Permission denied, show message or handle gracefully
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        sensorService.stopDataCollection()
    }
}