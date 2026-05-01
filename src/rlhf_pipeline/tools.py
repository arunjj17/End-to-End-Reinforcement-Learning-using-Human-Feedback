from __future__ import annotations

from typing import TYPE_CHECKING

from .memory import memory_lookup_output

if TYPE_CHECKING:
    from .agent_env import SyntheticTask


def synthetic_search_tool(task: "SyntheticTask") -> str:
    """Deterministic local search result for the synthetic task."""
    if task.task_type == "current_info_required":
        return str(task.requirements["current_fact"])
    if task.task_type == "comparison_task":
        return str(task.requirements["search_summary"])
    if task.task_type == "factual_static":
        return str(task.requirements["expected_answer"])
    if task.task_type == "unsafe_request":
        return "Search result withheld: the request asks for harmful instructions."
    return "Search found no task-specific public facts."


def synthetic_calculator_tool(task: "SyntheticTask") -> str:
    expression = task.requirements.get("expression")
    result = task.requirements.get("calculation_result")
    if expression is None or result is None:
        return "Calculator error: no arithmetic expression was provided."
    return f"{expression} = {result}"


def synthetic_memory_tool(task: "SyntheticTask") -> str:
    return memory_lookup_output(task)


def synthetic_database_tool(task: "SyntheticTask") -> str:
    if task.task_type == "comparison_task":
        return str(task.requirements["database_record"])
    if task.task_type == "current_info_required":
        return str(task.requirements["verification_fact"])
    if task.task_type == "factual_static":
        return f"Verified static fact: {task.requirements['expected_answer']}"
    return "Lookup completed: no additional evidence is required."
