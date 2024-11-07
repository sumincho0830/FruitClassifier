package com.sum.cancerclassification

import android.app.Activity
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
                // Permissionw as granted, open the camera
            }else{
                // Permission denied
                Toast.makeText(this, "Camera permission is required to use the camera", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun openCamera(){
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile = getPhotoFile(createFileName()) //file path (saved in a File object)

        val fileProvider = FileProvider.getUriForFile(this,"com.sum.cancerclassification.fileprovider", photoFile)
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

        /**
         * Incorporate Probability Results Here
         */

        val probability = "Probability: 0.00%"
        val date:String = Utils.dateFormat.format(Date())

        /** Data Saved Here */
        val imageInfo = ImageInfo(file.path, probability, date)
        imageList.add(imageInfo)

        // Notify RecyclerView to update
        adapter.notifyDataSetChanged()

        // Save to Firebase Database
        val database = FirebaseDatabase.getInstance()
        val myRef = database.getReference("images").push() //Each image gets a unique ID
        myRef.setValue(imageInfo) // Save image data with path, probability, and date

        // Notify Media Scanner to show the image in the gallery
        val intent = Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE)
        intent.data = Uri.fromFile(file)
        sendBroadcast(intent) // send Broadcast for the OS to notice


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
                    imageInfo?.let {imageList.add(it)}
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
}