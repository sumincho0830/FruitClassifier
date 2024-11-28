package com.sum.cancerclassification

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import com.google.firebase.FirebaseApp
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import com.sum.cancerclassification.databinding.ActivityMainBinding
import org.pytorch.*
import java.io.File
import java.io.FileOutputStream
import java.util.*

class MainActivity : AppCompatActivity() {
    private val TAG = "MainActivity"

    private lateinit var binding: ActivityMainBinding
    private val REQUEST_CAMERA_PERMISSION = 100
    val REQUEST_IMAGE_CAPTURE = 1
    lateinit var photoFile: File

    val imageList = mutableListOf<ImageInfo>()
    val adapter = ViewPagerAdapter(imageList) {
        Toast.makeText(this, "Image deleted successfully.", Toast.LENGTH_SHORT).show()
    }

    private lateinit var modelProcessor: PyTorchModelProcessor

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        FirebaseApp.initializeApp(this)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val viewPager = binding.viewPager
        viewPager.adapter = adapter

        // Initialize PyTorchModelProcessor and Load model
        modelProcessor = PyTorchModelProcessor(this)
        try{
            modelProcessor.loadModel("kickboard_model.ptl") // Model file in assets
            Toast.makeText(this, "Model loaded successfully!", Toast.LENGTH_SHORT).show()
        }catch (e: Exception){
            Log.d(TAG, "Error loading model: ${e.message}")
            Toast.makeText(this, "Failed to load model: ${e.message}", Toast.LENGTH_SHORT).show()
        }

        binding.floatingActionButton.setOnClickListener {
            checkCameraPermissionAndOpenCamera()
        }

        loadImageFromFirebase()
    }


    private fun checkCameraPermissionAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(android.Manifest.permission.CAMERA),
                REQUEST_CAMERA_PERMISSION
            )
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CAMERA_PERMISSION && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        } else {
            Toast.makeText(this, "Camera permission is required to use the camera", Toast.LENGTH_SHORT).show()
        }
    }

    private fun openCamera() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = getPhotoFile(createFileName())

        val fileProvider = FileProvider.getUriForFile(this, "com.sum.cancerclassification.fileprovider", photoFile)
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)

        if (takePictureIntent.resolveActivity(packageManager) != null) {
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
        }
    }

    private fun createFileName(): String {
        val date = Utils.fileFormat.format(Date())
        return "pred_$date"
    }

    private fun getPhotoFile(fileName: String): File {
        val storageDirectory = getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File(storageDirectory, "$fileName.jpg")
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK) {
            saveImageWithInfo(photoFile)
        }
    }

    private fun saveImageWithInfo(file: File) {
        val bitmap = BitmapFactory.decodeFile(file.absolutePath)

        val classificationResult: String
        try {
            // Classification result
            classificationResult = modelProcessor.classifyImage(bitmap)
        } catch (e: IllegalStateException) {
            Toast.makeText(this, "Model not initialized. Cannot classify image.", Toast.LENGTH_SHORT).show()
            return
        }
        val date: String = Utils.dateFormat.format(Date())
        val imageInfo = ImageInfo(file.path, classificationResult, date)

        // Save result to list
        imageList.add(imageInfo)

        // Notify RecyclerView to update
        adapter.notifyDataSetChanged()

        // Save to Firebase Database
        val database = FirebaseDatabase.getInstance()
        val myRef = database.getReference("images").push()
        myRef.setValue(imageInfo)
    }


    private fun loadImageFromFirebase() {
        val database = FirebaseDatabase.getInstance()
        val imageRef = database.getReference("images")

        imageRef.addValueEventListener(object : ValueEventListener {
            override fun onDataChange(snapshot: DataSnapshot) {
                imageList.clear()
                for (imageSnapshot in snapshot.children) {
                    val imageInfo = imageSnapshot.getValue(ImageInfo::class.java)
                    imageInfo?.let { imageList.add(it) }
                }
                adapter.notifyDataSetChanged()
            }

            override fun onCancelled(error: DatabaseError) {
                Toast.makeText(this@MainActivity, "Failed to load data.", Toast.LENGTH_SHORT).show()
            }
        })
    }

    data class ImageInfo(val imagePath: String = "", val classification: String = "", val date: String = "")
}
