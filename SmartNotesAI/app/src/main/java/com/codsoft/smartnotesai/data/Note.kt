package com.codsoft.smartnotesai.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for notes.
 * priorityColor keeps UI color hex for easy beginner-level handling.
 */
@Entity(tableName = "notes")
data class Note(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val title: String,
    val content: String,
    val priorityColor: String = "#F9A825"
)
