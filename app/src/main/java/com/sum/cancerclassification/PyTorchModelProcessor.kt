package com.sum.cancerclassification

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream

class PyTorchModelProcessor(private val context:Context) {
    private val TAG = "PyTorchModelProcessor"
    private lateinit var model:Module

    // load the model
    fun loadModel(assetName: String){
        try {
            val modelPath = assetFilePath(context, assetName)
            model = Module.load(modelPath)
        }catch (e:Exception){
            Log.d(TAG, "Failed to load model.")
            throw RuntimeException("Failed to load model: ${e.message}")
        }
    }

    // Preprocess the image and run interface
    fun classifyImage(bitmap: Bitmap): String{

        if (!::model.isInitialized) {
            throw IllegalStateException("Model has not been initialized. Make sure to call loadModel() first.")
        }

        // Preprocess the image to Tensor
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            floatArrayOf(0.485f, 0.456f, 0.406f), // Normalize mean
            floatArrayOf(0.229f, 0.224f, 0.225f)  // Normalize std
        )

        // Run interface
        val outputTensor = model.forward(IValue.from(inputTensor)).toTensor()

        // Process the output tensor
        val scores = outputTensor.dataAsFloatArray
        return if(probability >= 0.5) "정석" else "불법"
    }

    // Helper to copy model from assets to internal storage
    private fun assetFilePath(context: Context, assetName: String): String{
        val file = File(context.filesDir, assetName)
        if(!file.exists()){

            try{
                Log.d(TAG, "Copying $assetName to internal storage")
                context.assets.open(assetName).use { inputStream ->
                    FileOutputStream(file).use { outputStream ->
                        inputStream.copyTo(outputStream)
                    }
                }
                Log.d(TAG, "File copied to ${file.absolutePath}")
            }catch (e: Exception) {
                Log.d(TAG, "Failed to copy asset file")
                throw RuntimeException("Failed to copy asset file: $assetName", e)
            }
        }else{
            Log.d(TAG, "File already exists at $        ")
        }
        return file.absolutePath
    }
}
