package com.codsoft.smartnotesai.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.codsoft.smartnotesai.data.AppDatabase
import com.codsoft.smartnotesai.data.Note
import com.codsoft.smartnotesai.data.NotesRepository
import kotlinx.coroutines.launch

class NotesViewModel(application: Application) : AndroidViewModel(application) {
    private val repository: NotesRepository
    val notes

    init {
        val noteDao = AppDatabase.getDatabase(application).noteDao()
        repository = NotesRepository(noteDao)
        notes = repository.notes
    }

    fun addNote(title: String, content: String, color: String) {
        viewModelScope.launch {
            repository.addNote(Note(title = title, content = content, priorityColor = color))
        }
    }
}
