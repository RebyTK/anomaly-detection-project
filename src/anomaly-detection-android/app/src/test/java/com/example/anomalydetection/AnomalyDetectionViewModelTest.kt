package com.example.anomalydetection

import androidx.arch.core.executor.testing.InstantTaskExecutorRule
import androidx.lifecycle.Observer
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.MockitoJUnitRunner

@RunWith(MockitoJUnitRunner::class)
class AnomalyDetectionViewModelTest {

    @get:Rule
    val instantTaskExecutorRule = InstantTaskExecutorRule()

    private lateinit var viewModel: AnomalyDetectionViewModel

    @Mock
    private lateinit var statusObserver: Observer<String>

    @Mock
    private lateinit var anomalyDataObserver: Observer<List<Anomaly>>

    @Before
    fun setup() {
        MockitoAnnotations.openMocks(this)
        viewModel = AnomalyDetectionViewModel()
    }

    @Test
    fun `test initial state`() {
        // Verify initial values
        assert(viewModel.status.value == "Ready")
        assert(viewModel.anomalyData.value?.isEmpty() == true)
        assert(viewModel.isDetectionRunning.value == false)
    }

    @Test
    fun `test start detection`() {
        viewModel.startDetection()
        assert(viewModel.isDetectionRunning.value == true)
        assert(viewModel.status.value == "Detection Active"
    }

    @Test
    fun `test stop detection`() {
        viewModel.startDetection()
        viewModel.stopDetection()
        assert(viewModel.isDetectionRunning.value == false)
        assert(viewModel.status.value == "Detection Stopped"
    }

    @Test
    fun `test threshold setting`() {
        viewModel.setThreshold(50f)
        assert(viewModel.getCurrentThreshold() == 0.5)
    }

    @Test
    fun `test clear anomalies`() {
        viewModel.clearAnomalies()
        assert(viewModel.anomalyData.value?.isEmpty() == true)
        assert(viewModel.status.value == "Anomalies Cleared"
    }
}
