package com.sum.cancerclassification

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.recyclerview.widget.GridLayoutManager
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.firebase.FirebaseApp
import com.google.firebase.database.DataSnapshot
import com.google.firebase.database.DatabaseError
import com.google.firebase.database.FirebaseDatabase
import com.google.firebase.database.ValueEventListener
import com.google.firebase.ktx.Firebase
import com.sum.cancerclassification.databinding.ActivityMainBinding

import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import android.content.res.AssetFileDescriptor
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.nio.channels.FileChannel
import java.io.File
import java.util.Date

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private val REQUEST_CAMERA_PERMISSION = 100

    val REQUEST_IMAGE_CAPTURE = 1
    val FILE_NAME = "photo.jpg"
    lateinit var photoFile: File

    /** imageList to be changed later */
    val imageList = mutableListOf<ImageInfo>()
    var adapter = ImageAdapter(imageList)


    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        enableEdgeToEdge()
        setContentView(R.layout.activity_main)
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        FirebaseApp.initializeApp(this)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Set up RecyclerView with the adapter and Layout Manager
        val recyclerView = binding.recyclerView // Assuming recyclerView is in activity_main
        recyclerView.layoutManager = LinearLayoutManager(this)
        recyclerView.adapter = adapter

        val floatingActionButton = binding.floatingActionButton
        floatingActionButton.setOnClickListener{
            checkCameraPermissionAndOpenCamera()
        }

//        val floatingActionButton.setOnClickListener(object: View.OnClickListener{
//            override fun onClick(v: View?) {
//                TODO("Not yet implemented")
//            }
//        })

        loadImageFromFirebase()

    }

    private fun checkCameraPermissionAndOpenCamera(){
        if(ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED){
            openCamera()
        }else{
            // Request the camera permission
            ActivityCompat.requestPermissions(this,
                arrayOf(android.Manifest.permission.CAMERA), REQUEST_CAMERA_PERMISSION)
        }
    }


    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
        deviceId: Int
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults, deviceId)
        if(requestCode == REQUEST_CAMERA_PERMISSION){
            if(grantResults.isNotEmpty() && (grantResults[0] == PackageManager.PERMISSION_GRANTED)){
                // Permission was granted, open the camera
                openCamera()
            }else{
                // Permission denied
                Toast.makeText(this, "Camera permission is required to use the camera", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun openCamera(){
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = getPhotoFile(createFileName()) //file path (saved in a File object)

        val fileProvider = FileProvider.getUriForFile(this,"com.sum.fruitclassifier.fileprovider", photoFile)
        takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, fileProvider)

        if(takePictureIntent.resolveActivity(packageManager)!= null){
            startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE)
        }
    }

    private fun createFileName(): String{
        val date:String = Utils.fileFormat.format(Date())
        return "pred_"+date
    }

    private fun getPhotoFile(fileName:String):File{
        val storageDirectory = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES)
        return File(storageDirectory, "$fileName.jpg")
    }

    // Add Camera Preview?

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode == REQUEST_IMAGE_CAPTURE && resultCode == Activity.RESULT_OK){
            saveImageWithInfo(photoFile) //if Image captured successfully, save it to the preset path
        }
    }

    data class ImageInfo(val imagePath: String = "", val probability: String = "", val date:String = "")



    fun saveImageWithInfo(file:File){
        /** Incorporating Prediction Model:
         * 1. Load image as bitmap from file.path
         * 2. Preprocess image for model input (call preprocessImage())
         * 3. Set output buffer
         * 4. Run model Interpreter interface
         * */

        // Initialize the interpreter
        val interpreter : Interpreter
        try{
            interpreter = Interpreter(loadModelFile())
        }catch (e:Exception){
            e.printStackTrace()
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show()
            return
        }

        /** 1. */
        val imageBitmap: Bitmap = BitmapFactory.decodeFile(file.absolutePath)

        /** 2. */
        val inputBuffer = preprocessImage(imageBitmap)

        /** 3. */
        // Array(1) -> batch size
        // FloatArray(2) -> inner array size 1 holds the confidence score or probability for the "orange" class
        val output = Array(1) {FloatArray(1)} // adjust numClasses for your model's output
        // classification model outputs a single array with scores for each class.
        // After running the model, output[0][0] will contain a single float value.
        // This vlaue typically represents the confidence that the input image is an "orange"

        /** 4. */
        interpreter.run(inputBuffer, output)

        val confidence = output[0][0]
        val date:String = Utils.dateFormat.format(Date())
        /** Data Saved Here */
        val imageInfo = ImageInfo(file.path, confidence.toString(), date)
        imageList.add(imageInfo)

        // Notify RecyclerView to update
        adapter.notifyDataSetChanged()


        // stores the index of the class with the highest confidence score
        // val predictedLabelIndex = output[0].indexOf(output[0].maxOrNull()!!)


        // Save to Firebase Database
        val database = FirebaseDatabase.getInstance()
        val myRef = database.getReference("images").push() //Each image gets a unique ID
        myRef.setValue(imageInfo) // Save image data with path, probability, and date

        /** Save imge to gallery as MediaStore*/
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, file.name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES) // Makes it appear in the gallery
        }

        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        uri?.let {
            contentResolver.openOutputStream(it)?.use { outputStream ->
                BitmapFactory.decodeFile(file.absolutePath).compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            }
            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(uri, values, null, null)
        }

    }

    private fun loadImageFromFirebase(){
        val database = FirebaseDatabase.getInstance()
        val imageRef = database.getReference("images") // get reference to the images node

        imageRef.addValueEventListener(object: ValueEventListener{
            override fun onDataChange(snapshot: DataSnapshot) {
                // clear current list to avoid duplication
                imageList.clear()

                // Loop through each child in the "images" node
                for(imageSnapshot in snapshot.children){
                    // Convert each child to an ImageInfo object
                    val imageInfo = imageSnapshot.getValue(ImageInfo::class.java) // get value as an ImageInfo object
                    imageInfo?.let {imageList.add(it)} // add imageList to info
                }

                // Notify RecyclerView to update
                adapter.notifyDataSetChanged()
            }

            override fun onCancelled(error: DatabaseError) {
                // Handle possible errors
                Toast.makeText(this@MainActivity, "Failed to load data.", Toast.LENGTH_SHORT).show()
            }
        })
    }

    // funtion to load the model file from assets
    fun loadModelFile(): ByteBuffer {
        val assetFileDescriptor: AssetFileDescriptor = assets.openFd("fruit_classifier_model.tflite")
        val fileinputStream = assetFileDescriptor.createInputStream()
        val fileChannel = fileinputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }



    // Prepare input data (assuming you have an image in Bitmap format)
    fun preprocessImage(image:Bitmap): ByteBuffer{
        val inputImageBuffer = ByteBuffer.allocateDirect(4 * 320 * 258 * 3) // 4 bytes per float
        inputImageBuffer.order(ByteOrder.nativeOrder())

        // resize the taken photo to 320x258 (keep size compatible from the beginning)
        val resizedImage = Bitmap.createScaledBitmap(image, 320, 258, true)
        for(y in 0 until 258){
            for(x in 0 until 320){
                val pixel = resizedImage.getPixel(x, y)
                inputImageBuffer.putFloat((pixel shr 16 and 0xFF)/255.0f) // Red
                inputImageBuffer.putFloat((pixel shr 8 and 0xFF)/255.0f) // Green
                inputImageBuffer.putFloat((pixel and 0xFF)/255.0f) // Blue
            }
        }
        return inputImageBuffer
    }



}