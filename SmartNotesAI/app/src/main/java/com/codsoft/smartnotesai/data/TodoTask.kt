package com.codsoft.smartnotesai.data

import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * Room entity for to-do tasks.
 */
@Entity(tableName = "tasks")
data class TodoTask(
    @PrimaryKey(autoGenerate = true) val id: Int = 0,
    val title: String,
    val isCompleted: Boolean = false,
    val priority: Int = 2 // 1=high(red), 2=medium(yellow), 3=low(green)
)
