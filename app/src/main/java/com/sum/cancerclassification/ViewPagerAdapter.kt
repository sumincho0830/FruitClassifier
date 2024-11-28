package com.sum.cancerclassification

import android.app.AlertDialog
import android.content.Context
import android.net.Uri
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.recyclerview.widget.RecyclerView
import com.google.firebase.database.FirebaseDatabase
import java.io.File

class ViewPagerAdapter(
    private val images: MutableList<MainActivity.ImageInfo>, // Changed to MutableList
    private val onItemDeleted: () -> Unit // Callback to notify MainActivity when an item is deleted
) : RecyclerView.Adapter<ViewPagerAdapter.ImageViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ImageViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_image, parent, false)
        return ImageViewHolder(view)
    }

    override fun onBindViewHolder(holder: ImageViewHolder, position: Int) {
        val imageInfo = images[position]
        holder.probabilityText.text = imageInfo.classification
        holder.dateText.text = imageInfo.date
        holder.imageView.setImageURI(Uri.fromFile(File(imageInfo.imagePath)))

        // Handle long-press for delete
        holder.itemView.setOnLongClickListener {
            showDeleteDialog(holder.itemView.context, imageInfo, position)
            true
        }
    }

    inner class ImageViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val imageView: ImageView = view.findViewById(R.id.itemImageView)
        val probabilityText: TextView = view.findViewById(R.id.probabilityText)
        val dateText: TextView = view.findViewById(R.id.dateText)
    }

    override fun getItemCount() = images.size

    private fun showDeleteDialog(context: Context, imageInfo: MainActivity.ImageInfo, position: Int) {
        AlertDialog.Builder(context).apply {
            setTitle("Delete Image")
            setMessage("Are you sure you want to delete this image?")
            setPositiveButton("Yes") { _, _ ->
                deleteImage(context, imageInfo, position)
            }
            setNegativeButton("No", null)
        }.show()
    }

    private fun deleteImage(context: Context, imageInfo: MainActivity.ImageInfo, position: Int) {
        // Delete the file from local storage
        val file = File(imageInfo.imagePath)
        if (file.exists() && file.delete()) {
            // Remove from Firebase
            val database = FirebaseDatabase.getInstance()
            val imageRef = database.getReference("images")
            imageRef.orderByChild("imagePath").equalTo(imageInfo.imagePath)
                .addListenerForSingleValueEvent(object : com.google.firebase.database.ValueEventListener {
                    override fun onDataChange(snapshot: com.google.firebase.database.DataSnapshot) {
                        for (childSnapshot in snapshot.children) {
                            childSnapshot.ref.removeValue()
                        }
                    }

                    override fun onCancelled(error: com.google.firebase.database.DatabaseError) {
                        Toast.makeText(
                            context,
                            "Failed to delete from database.",
                            Toast.LENGTH_SHORT
                        ).show()
                    }
                })

            // Remove from local list
            images.removeAt(position)
            notifyItemRemoved(position)
            notifyItemRangeChanged(position, images.size)

            // Notify MainActivity
            onItemDeleted()
        } else {
            Toast.makeText(
                context,
                "Failed to delete the file.",
                Toast.LENGTH_SHORT
            ).show()
        }
    }
}
