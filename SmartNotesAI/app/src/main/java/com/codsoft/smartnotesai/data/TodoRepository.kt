package com.codsoft.smartnotesai.data

class TodoRepository(private val todoDao: TodoDao) {
    val tasks = todoDao.getAllTasks()

    suspend fun addTask(task: TodoTask) = todoDao.insert(task)
    suspend fun updateTask(task: TodoTask) = todoDao.update(task)
}
