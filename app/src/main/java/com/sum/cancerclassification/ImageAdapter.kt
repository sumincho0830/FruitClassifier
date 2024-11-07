package com.sum.cancerclassification

import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.io.File

class ImageAdapter(private val images: List<MainActivity.ImageInfo>): //생성 시 리스트를 받아 온다
    RecyclerView.Adapter<ImageAdapter.ImageViewHolder>() {


    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_image, parent, false)
        return ImageViewHolder(view)
    }

    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
        val imageInfo = images[position]
        holder.probabilityText.text = imageInfo.probability
        holder.dateText.text = imageInfo.date
        holder.imageView.setImageURI(Uri.fromFile(File(imageInfo.imagePath)))

    }

    inner class ImageViewHolder(view: View) : RecyclerView.ViewHolder(view) { //코틀린은 생성자가 통합되어있다
        val imageView: ImageView = view.findViewById(R.id.itemImageView)
        val probabilityText: TextView = view.findViewById(R.id.probabilityText)
        val dateText: TextView = view.findViewById(R.id.dateText)
    }

    override fun getItemCount() = images.size
}