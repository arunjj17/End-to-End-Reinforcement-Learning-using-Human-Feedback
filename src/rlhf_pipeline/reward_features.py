from __future__ import annotations

import numpy as np

from .agent_env import (
    ACTIONS,
    TASK_TYPES,
    TOOL_ACTIONS,
    AgentTrajectory,
    SyntheticTask,
    trajectory_diagnostics,
)
from .agent_policy import PROMPT_FLAG_NAMES, prompt_flags


def _one_hot(value: str, choices: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(choices), dtype=np.float64)
    if value in choices:
        out[choices.index(value)] = 1.0
    return out


def trajectory_features(
    task: SyntheticTask,
    trajectory: AgentTrajectory,
    *,
    max_steps: int,
) -> np.ndarray:
    actions = trajectory.actions
    action_counts = np.array([actions.count(action) for action in ACTIONS], dtype=np.float64)
    normalized_counts = action_counts / max(1.0, float(max_steps))
    used_flags = (action_counts > 0).astype(np.float64)
    first_action = actions[0] if actions else "<none>"
    last_action = actions[-1] if actions else "<none>"
    action_choices = ("<none>",) + ACTIONS
    diagnostics = trajectory_diagnostics(task, actions)
    tool_count = sum(1 for action in actions if action in TOOL_ACTIONS)

    scalar_features = np.array(
        [
            len(actions) / max(1.0, float(max_steps)),
            tool_count / max(1.0, float(max_steps)),
            diagnostics.unnecessary_tool_calls / max(1.0, float(max_steps)),
            diagnostics.repeated_tool_calls / max(1.0, float(max_steps)),
            1.0 if diagnostics.required_tools_used else 0.0,
            1.0 if diagnostics.success else 0.0,
            diagnostics.memory_usage_accuracy,
            diagnostics.clarification_accuracy,
            diagnostics.style_match_score or 0.0,
            trajectory.answer_quality,
            trajectory.cost_penalty,
            trajectory.total_tool_cost,
            trajectory.total_tool_latency,
        ],
        dtype=np.float64,
    )

    return np.concatenate(
        (
            np.ones(1, dtype=np.float64),
            _one_hot(task.task_type, TASK_TYPES),
            prompt_flags(task),
            normalized_counts,
            used_flags,
            _one_hot(first_action, action_choices),
            _one_hot(last_action, action_choices),
            scalar_features,
        )
    )


def trajectory_feature_dim(max_steps: int) -> int:
    # 1 bias + task type + prompt flags + action counts + used flags
    # + first action + last action + scalar diagnostics.
    return (
        1
        + len(TASK_TYPES)
        + len(PROMPT_FLAG_NAMES)
        + len(ACTIONS)
        + len(ACTIONS)
        + (1 + len(ACTIONS))
        + (1 + len(ACTIONS))
        + 13
    )
