package org.tensorflow.lite.examples.poseestimation

import android.app.Activity
import android.content.Intent
import android.graphics.BitmapFactory
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.content.FileProvider
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.ml.MoveNet
import org.tensorflow.lite.examples.poseestimation.ml.PoseClassifier
import org.tensorflow.lite.examples.poseestimation.ml.PoseDetector
import org.w3c.dom.Text
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class CaptureActivity : AppCompatActivity() {
    companion object {
        private const val REQUEST_IMAGE_CAPTURE = 1
    }

    private lateinit var currentPhotoPath: String
    private lateinit var btnTakePhoto: Button
    private lateinit var imgImageResult: ImageView
    private lateinit var detector: PoseDetector
    private lateinit var classifier: PoseClassifier
    private lateinit var tvClassificationValue1: TextView
    private lateinit var tvClassificationValue2: TextView
    private lateinit var tvClassificationValue3: TextView
    private lateinit var tvClassificationValue4: TextView
    private lateinit var tvClassificationValue5: TextView
    private lateinit var llClassification: LinearLayout
    private lateinit var tvNoPersonDetected: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_capture)
        btnTakePhoto = findViewById(R.id.btnTakePhoto)
        imgImageResult = findViewById(R.id.imgImageResult)
        tvClassificationValue1 = findViewById(R.id.tvClassificationValue1)
        tvClassificationValue2 = findViewById(R.id.tvClassificationValue2)
        tvClassificationValue3 = findViewById(R.id.tvClassificationValue3)
        tvClassificationValue4 = findViewById(R.id.tvClassificationValue4)
        tvClassificationValue5 = findViewById(R.id.tvClassificationValue5)
        llClassification = findViewById(R.id.llClassification)
        tvNoPersonDetected = findViewById(R.id.tvNoPersonDetected)
        btnTakePhoto.setOnClickListener {
            dispatchTakePictureIntent()
        }
        detector = MoveNet.create(this, Device.CPU)
        classifier = PoseClassifier.create(this)
    }


    @Throws(IOException::class)
    private fun createImageFile(): File {
        // Create an image file name
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File? = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_", /* prefix */
            ".jpg", /* suffix */
            storageDir /* directory */
        ).apply {
            // Save a file: path for use with ACTION_VIEW intents
            currentPhotoPath = absolutePath
        }
    }

    private fun dispatchTakePictureIntent() {
        Intent(MediaStore.ACTION_IMAGE_CAPTURE).also { takePictureIntent ->
            // Ensure that there's a camera activity to handle the intent
            takePictureIntent.resolveActivity(packageManager)?.also {
                // Create the File where the photo should go
                val photoFile: File? = try {
                    createImageFile()
                } catch (ex: IOException) {
                    // Error occurred while creating the File
                    null
                }
                // Continue only if the File was successfully created
                photoFile?.also {
                    val photoURI: Uri = FileProvider.getUriForFile(
                        this,
                        "com.example.android.fileprovider",
                        it
                    )
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
                }
            }
        }
    }

    private fun setPic() {
        // Get the dimensions of the View
        val targetW: Int = imgImageResult.width
        val targetH: Int = imgImageResult.height

        val bmOptions = BitmapFactory.Options().apply {
            // Get the dimensions of the bitmap
            inJustDecodeBounds = true

            BitmapFactory.decodeFile(currentPhotoPath, this)

            val photoW: Int = outWidth
            val photoH: Int = outHeight

            // Determine how much to scale down the image
            val scaleFactor: Int = Math.max(1, Math.min(photoW / targetW, photoH / targetH))

            // Decode the image file into a Bitmap sized to fill the View
            inJustDecodeBounds = false
            inSampleSize = scaleFactor
            inPurgeable = true
        }
        BitmapFactory.decodeFile(currentPhotoPath, bmOptions)?.also { bitmap ->
            detector.estimatePoses(bitmap).let {
                var output = bitmap
                if (it.isNotEmpty()) {
                    val classificationResult = classifier.classify(it[0])
                    if (it[0].score > 0.2) {
                        llClassification.visibility = View.VISIBLE
                        classificationResult.sortedByDescending { score ->
                            score.second
                        }.let { result ->
                            if (result.isEmpty()) return
                            tvClassificationValue1.text =
                                "${result[0].first} : (${String.format("%.2f", result[0].second)})"
                            tvClassificationValue2.text =
                                "${result[1].first} : (${String.format("%.2f", result[1].second)})"
                            tvClassificationValue3.text =
                                "${result[2].first} : (${String.format("%.2f", result[2].second)})"
                            tvClassificationValue4.text =
                                "${result[3].first} : (${String.format("%.2f", result[3].second)})"
                            tvClassificationValue5.text =
                                "${result[4].first} : (${String.format("%.2f", result[4].second)})"
                        }
                        tvNoPersonDetected.visibility = View.GONE
                        output = VisualizationUtils.drawBodyKeypoints(bitmap, it, false)
                    } else {
                        tvNoPersonDetected.visibility = View.VISIBLE
                        llClassification.visibility = View.INVISIBLE
                    }
                }

                imgImageResult.setImageBitmap(output)
            }
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE &&
            resultCode == Activity.RESULT_OK
        ) {
            setPic()
        }
    }
}