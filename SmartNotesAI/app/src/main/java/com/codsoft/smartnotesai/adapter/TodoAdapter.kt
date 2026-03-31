package com.codsoft.smartnotesai.adapter

import android.graphics.Color
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.codsoft.smartnotesai.data.TodoTask
import com.codsoft.smartnotesai.databinding.ItemTaskBinding

class TodoAdapter(private val onTaskChecked: (TodoTask) -> Unit) : RecyclerView.Adapter<TodoAdapter.TaskViewHolder>() {
    private var items: List<TodoTask> = emptyList()

    fun submitList(tasks: List<TodoTask>) {
        items = tasks
        notifyDataSetChanged()
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): TaskViewHolder {
        val binding = ItemTaskBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return TaskViewHolder(binding)
    }

    override fun onBindViewHolder(holder: TaskViewHolder, position: Int) {
        holder.bind(items[position], onTaskChecked)
    }

    override fun getItemCount(): Int = items.size

    class TaskViewHolder(private val binding: ItemTaskBinding) : RecyclerView.ViewHolder(binding.root) {
        fun bind(task: TodoTask, onTaskChecked: (TodoTask) -> Unit) {
            binding.cbTask.text = task.title
            binding.cbTask.isChecked = task.isCompleted
            binding.cbTask.setOnClickListener { onTaskChecked(task) }
            val color = when (task.priority) {
                1 -> "#E53935"
                2 -> "#F9A825"
                else -> "#43A047"
            }
            binding.viewPriority.setBackgroundColor(Color.parseColor(color))
        }
    }
}
