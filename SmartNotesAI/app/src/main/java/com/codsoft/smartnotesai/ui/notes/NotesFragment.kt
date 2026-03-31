package com.codsoft.smartnotesai.ui.notes

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.codsoft.smartnotesai.adapter.NotesAdapter
import com.codsoft.smartnotesai.databinding.FragmentNotesBinding
import com.codsoft.smartnotesai.ui.editor.NoteEditorActivity
import com.codsoft.smartnotesai.viewmodel.NotesViewModel

class NotesFragment : Fragment() {
    private var _binding: FragmentNotesBinding? = null
    private val binding get() = _binding!!
    private val notesViewModel: NotesViewModel by viewModels()
    private val notesAdapter = NotesAdapter()

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentNotesBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.rvNotes.apply {
            layoutManager = LinearLayoutManager(requireContext())
            adapter = notesAdapter
        }

        notesViewModel.notes.observe(viewLifecycleOwner) {
            notesAdapter.submitList(it)
            binding.rvNotes.scheduleLayoutAnimation()
        }

        binding.fabAddNote.setOnClickListener {
            startActivity(Intent(requireContext(), NoteEditorActivity::class.java))
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
