package org.tensorflow.lite.examples.poseestimation

import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class MainActivity : AppCompatActivity() {
    private lateinit var btnRealTime: Button
    private lateinit var btnCapture: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnRealTime = findViewById(R.id.btnRealTime)
        btnCapture = findViewById(R.id.btnCapture)

        btnRealTime.setOnClickListener {
            startActivity(Intent(this, RealTimeEstimateActivity::class.java))
        }
        btnCapture.setOnClickListener {
            startActivity(Intent(this, CaptureActivity::class.java))
        }
    }
}