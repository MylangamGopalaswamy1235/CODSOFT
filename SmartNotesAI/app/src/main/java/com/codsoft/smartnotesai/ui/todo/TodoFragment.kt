package com.codsoft.smartnotesai.ui.todo

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ArrayAdapter
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.recyclerview.widget.LinearLayoutManager
import com.codsoft.smartnotesai.adapter.TodoAdapter
import com.codsoft.smartnotesai.databinding.FragmentTodoBinding
import com.codsoft.smartnotesai.viewmodel.TodoViewModel

class TodoFragment : Fragment() {
    private var _binding: FragmentTodoBinding? = null
    private val binding get() = _binding!!
    private val viewModel: TodoViewModel by viewModels()
    private lateinit var adapter: TodoAdapter

    override fun onCreateView(inflater: LayoutInflater, container: ViewGroup?, savedInstanceState: Bundle?): View {
        _binding = FragmentTodoBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        adapter = TodoAdapter { task -> viewModel.toggleTask(task) }
        binding.rvTasks.layoutManager = LinearLayoutManager(requireContext())
        binding.rvTasks.adapter = adapter

        val priorities = listOf("High", "Medium", "Low")
        binding.spinnerPriority.adapter = ArrayAdapter(requireContext(), android.R.layout.simple_spinner_dropdown_item, priorities)

        viewModel.tasks.observe(viewLifecycleOwner) {
            adapter.submitList(it)
            binding.rvTasks.scheduleLayoutAnimation()
        }

        binding.btnAddTask.setOnClickListener {
            val taskTitle = binding.etTaskTitle.text.toString().trim()
            if (taskTitle.isNotEmpty()) {
                val priority = binding.spinnerPriority.selectedItemPosition + 1
                viewModel.addTask(taskTitle, priority)
                binding.etTaskTitle.setText("")
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
