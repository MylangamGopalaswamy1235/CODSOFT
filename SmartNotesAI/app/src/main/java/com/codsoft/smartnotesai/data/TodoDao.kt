package com.codsoft.smartnotesai.data

import androidx.lifecycle.LiveData
import androidx.room.Dao
import androidx.room.Insert
import androidx.room.Query
import androidx.room.Update

@Dao
interface TodoDao {
    @Query("SELECT * FROM tasks ORDER BY priority ASC, id DESC")
    fun getAllTasks(): LiveData<List<TodoTask>>

    @Insert
    suspend fun insert(task: TodoTask)

    @Update
    suspend fun update(task: TodoTask)
}
