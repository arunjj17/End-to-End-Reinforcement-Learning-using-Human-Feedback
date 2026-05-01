from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .tools import (
    synthetic_calculator_tool,
    synthetic_database_tool,
    synthetic_memory_tool,
    synthetic_search_tool,
)

if TYPE_CHECKING:
    from .agent_env import SyntheticTask


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    action: str
    description: str
    input_schema: str
    output_schema: str
    cost: float
    latency: float
    risk: str
    recommended_task_types: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredTool:
    metadata: ToolMetadata
    executor: Callable[["SyntheticTask"], str]


LOCAL_TOOL_REGISTRY: dict[str, RegisteredTool] = {
    "search_tool": RegisteredTool(
        metadata=ToolMetadata(
            name="search_tool",
            action="search_tool",
            description="Use for synthetic current information and evidence gathering.",
            input_schema="SyntheticTask with current or comparison requirements.",
            output_schema="Deterministic text snippet from a local synthetic search index.",
            cost=0.12,
            latency=0.35,
            risk="medium",
            recommended_task_types=("current_info_required", "comparison_task"),
        ),
        executor=synthetic_search_tool,
    ),
    "calculator_tool": RegisteredTool(
        metadata=ToolMetadata(
            name="calculator_tool",
            action="calculator_tool",
            description="Use for arithmetic, numerical comparisons, and simple calculations.",
            input_schema="SyntheticTask containing an arithmetic expression.",
            output_schema="Expression and deterministic numeric result.",
            cost=0.05,
            latency=0.10,
            risk="low",
            recommended_task_types=("math_calculation", "comparison_task"),
        ),
        executor=synthetic_calculator_tool,
    ),
    "memory_lookup": RegisteredTool(
        metadata=ToolMetadata(
            name="memory_lookup",
            action="memory_lookup",
            description="Use for relevant synthetic user preferences and personalization memory.",
            input_schema="SyntheticTask with memory keys in hidden requirements.",
            output_schema="Relevant memory entries or an empty-memory message.",
            cost=0.03,
            latency=0.05,
            risk="low",
            recommended_task_types=("personalized_answer", "comparison_task"),
        ),
        executor=synthetic_memory_tool,
    ),
    "verify_answer": RegisteredTool(
        metadata=ToolMetadata(
            name="verify_answer",
            action="verify_answer",
            description="Use to verify answers against a local synthetic database lookup.",
            input_schema="SyntheticTask with verification or comparison requirements.",
            output_schema="Deterministic verification or lookup record.",
            cost=0.08,
            latency=0.20,
            risk="low",
            recommended_task_types=(
                "current_info_required",
                "math_calculation",
                "comparison_task",
            ),
        ),
        executor=synthetic_database_tool,
    ),
}


def registered_tool_actions() -> tuple[str, ...]:
    return tuple(LOCAL_TOOL_REGISTRY)


def get_tool(action: str) -> RegisteredTool | None:
    return LOCAL_TOOL_REGISTRY.get(action)


def get_tool_metadata(action: str) -> ToolMetadata | None:
    tool = get_tool(action)
    return None if tool is None else tool.metadata


def execute_registered_tool(task: "SyntheticTask", action: str) -> tuple[ToolMetadata, str] | None:
    tool = get_tool(action)
    if tool is None:
        return None
    return tool.metadata, tool.executor(task)


def recommended_tool_actions(task_type: str) -> tuple[str, ...]:
    return tuple(
        action
        for action, tool in LOCAL_TOOL_REGISTRY.items()
        if task_type in tool.metadata.recommended_task_types
    )


def total_tool_cost(actions: tuple[str, ...]) -> float:
    return float(
        sum(
            tool.metadata.cost
            for action in actions
            if (tool := get_tool(action)) is not None
        )
    )


def total_tool_latency(actions: tuple[str, ...]) -> float:
    return float(
        sum(
            tool.metadata.latency
            for action in actions
            if (tool := get_tool(action)) is not None
        )
    )

