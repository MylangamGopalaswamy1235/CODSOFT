package com.codsoft.smartnotesai.ui.assistant

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.fragment.app.Fragment
import com.codsoft.smartnotesai.databinding.FragmentAiAssistantBinding

class AiAssistantFragment : Fragment() {
    private var _binding: FragmentAiAssistantBinding? = null
    private val binding get() = _binding!!

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentAiAssistantBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        binding.btnAsk.setOnClickListener {
            val query = binding.etQuery.text.toString().trim()
            val message = if (query.isEmpty()) "Type something to ask AI." else "AI feature UI ready for: $query"
            Toast.makeText(requireContext(), message, Toast.LENGTH_SHORT).show()
        }

        setupSuggestionCard(binding.cardDbms, "Revise DBMS")
        setupSuggestionCard(binding.cardReport, "Complete AI report")
    }

    private fun setupSuggestionCard(card: View, text: String) {
        card.setOnClickListener {
            card.animate().scaleX(0.98f).scaleY(0.98f).setDuration(80).withEndAction {
                card.animate().scaleX(1f).scaleY(1f).setDuration(80).start()
            }.start()
            binding.etQuery.setText(text)
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
