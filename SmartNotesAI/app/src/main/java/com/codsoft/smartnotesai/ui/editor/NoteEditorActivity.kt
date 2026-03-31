package com.codsoft.smartnotesai.ui.editor

import android.os.Bundle
import android.widget.Toast
import androidx.activity.viewModels
import androidx.appcompat.app.AppCompatActivity
import com.codsoft.smartnotesai.databinding.ActivityNoteEditorBinding
import com.codsoft.smartnotesai.viewmodel.NotesViewModel

class NoteEditorActivity : AppCompatActivity() {
    private lateinit var binding: ActivityNoteEditorBinding
    private val notesViewModel: NotesViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityNoteEditorBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnSave.setOnClickListener {
            val title = binding.etTitle.text.toString().ifBlank { "Untitled Note" }
            val content = binding.etContent.text.toString()
            notesViewModel.addNote(title, content, "#F9A825")
            finish()
        }

        // UI-only assistant actions for now.
        val aiButtons = listOf(binding.btnSummarize, binding.btnPolish, binding.btnShorten, binding.btnExtend, binding.btnGrammar)
        aiButtons.forEach { button ->
            button.setOnClickListener {
                Toast.makeText(this, "${button.text} clicked (UI only)", Toast.LENGTH_SHORT).show()
            }
        }

        val attachmentButtons = listOf(binding.btnImage, binding.btnAudio, binding.btnScan)
        attachmentButtons.forEach { button ->
            button.setOnClickListener {
                Toast.makeText(this, "${button.text} option selected (UI only)", Toast.LENGTH_SHORT).show()
            }
        }
    }
}
