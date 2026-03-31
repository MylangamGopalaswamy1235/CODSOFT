package com.codsoft.smartnotesai.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LiveData
import androidx.lifecycle.viewModelScope
import com.codsoft.smartnotesai.data.AppDatabase
import com.codsoft.smartnotesai.data.TodoRepository
import com.codsoft.smartnotesai.data.TodoTask
import kotlinx.coroutines.launch

class TodoViewModel(application: Application) : AndroidViewModel(application) {
    private val repository: TodoRepository
    val tasks: LiveData<List<TodoTask>>

    init {
        val todoDao = AppDatabase.getDatabase(application).todoDao()
        repository = TodoRepository(todoDao)
        tasks = repository.tasks
    }

    fun addTask(title: String, priority: Int) {
        viewModelScope.launch {
            repository.addTask(TodoTask(title = title, priority = priority))
        }
    }

    fun toggleTask(task: TodoTask) {
        viewModelScope.launch {
            repository.updateTask(task.copy(isCompleted = !task.isCompleted))
        }
    }
}
