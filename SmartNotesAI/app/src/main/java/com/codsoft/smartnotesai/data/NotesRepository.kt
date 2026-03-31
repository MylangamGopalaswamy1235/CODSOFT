package com.codsoft.smartnotesai.data

class NotesRepository(private val noteDao: NoteDao) {
    val notes = noteDao.getAllNotes()

    suspend fun addNote(note: Note) = noteDao.insert(note)
}
