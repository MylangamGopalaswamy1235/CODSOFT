package com.codsoft.smartnotesai.ui.map

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import androidx.fragment.app.Fragment
import com.codsoft.smartnotesai.databinding.FragmentKnowledgeMapBinding

class KnowledgeMapFragment : Fragment() {
    private var _binding: FragmentKnowledgeMapBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentKnowledgeMapBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        val topics = listOf(
            "Computer Science",
            "  ├── DBMS",
            "  ├── AI",
            "  │     ├── ML",
            "  │     └── NLP",
            "  └── Networks"
        )
        binding.lvTree.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_list_item_1, topics)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
