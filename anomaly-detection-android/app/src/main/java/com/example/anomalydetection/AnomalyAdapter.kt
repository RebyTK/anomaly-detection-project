package com.example.anomalydetection

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.SimpleDateFormat
import java.util.*

class AnomalyAdapter : RecyclerView.Adapter<AnomalyAdapter.AnomalyViewHolder>() {

    private var anomalies: List<Anomaly> = emptyList()
    private val dateFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())

    fun updateAnomalies(newAnomalies: List<Anomaly>) {
        anomalies = newAnomalies
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): AnomalyViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_anomaly, parent, false)
        return AnomalyViewHolder(view)
    }

    override fun onBindViewHolder(holder: AnomalyViewHolder, position: Int) {
        holder.bind(anomalies[position])
    }

    override fun getItemCount(): Int = anomalies.size

    inner class AnomalyViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val valueTextView: TextView = itemView.findViewById(R.id.anomalyValueTextView)
        private val reasonTextView: TextView = itemView.findViewById(R.id.anomalyReasonTextView)
        private val timestampTextView: TextView = itemView.findViewById(R.id.anomalyTimestampTextView)

        fun bind(anomaly: Anomaly) {
            valueTextView.text = "Value: ${String.format("%.2f", anomaly.dataPoint.value)}"
            reasonTextView.text = "Reason: ${anomaly.reason}"
            timestampTextView.text = "Time: ${dateFormat.format(Date(anomaly.dataPoint.timestamp))}"
        }
    }
}
