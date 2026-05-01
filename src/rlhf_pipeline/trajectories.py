from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent_env import (
    ACTIONS,
    TERMINAL_ACTIONS,
    TOOL_ACTIONS,
    AgentTrajectory,
    SyntheticTask,
    ideal_action_sequence,
    run_actions,
    run_tool,
)
from .agent_policy import state_features


@dataclass(frozen=True)
class TrajectoryPreferenceDataset:
    preferred: tuple[AgentTrajectory, ...]
    rejected: tuple[AgentTrajectory, ...]


def _ensure_terminal(actions: list[str], task: SyntheticTask) -> list[str]:
    if not any(action in TERMINAL_ACTIONS for action in actions):
        actions.append(ideal_action_sequence(task)[-1])
    terminal_index = next(
        idx for idx, action in enumerate(actions) if action in TERMINAL_ACTIONS
    )
    return actions[: terminal_index + 1]


def noisy_demonstration_actions(
    task: SyntheticTask,
    rng: np.random.Generator,
    *,
    noise_rate: float,
) -> tuple[str, ...]:
    actions = list(ideal_action_sequence(task))
    if rng.random() >= noise_rate:
        return tuple(actions)

    allowed_tools = set(task.requirements.get("allowed_tools", ()))
    required_tools = set(task.requirements.get("required_tools", ()))
    unnecessary_tools = [
        action for action in ACTIONS if action in TOOL_ACTIONS and action not in allowed_tools
    ]
    mutation = int(rng.integers(4))

    if mutation == 0 and unnecessary_tools:
        insert_at = max(0, len(actions) - 1)
        actions.insert(insert_at, str(rng.choice(unnecessary_tools)))
    elif mutation == 1 and required_tools:
        removable = [idx for idx, action in enumerate(actions) if action in required_tools]
        if removable:
            del actions[int(rng.choice(removable))]
    elif mutation == 2:
        terminal_replacements = [
            action for action in ACTIONS if action in TERMINAL_ACTIONS and action != actions[-1]
        ]
        actions[-1] = str(rng.choice(terminal_replacements))
    elif mutation == 3:
        existing_tools = [action for action in actions if action in TOOL_ACTIONS]
        if existing_tools:
            insert_at = max(0, len(actions) - 1)
            actions.insert(insert_at, existing_tools[0])

    return tuple(_ensure_terminal(actions, task))


def generate_sft_demonstrations(
    tasks: list[SyntheticTask],
    rng: np.random.Generator,
    *,
    demos_per_task: int,
    noise_rate: float,
) -> list[AgentTrajectory]:
    demos: list[AgentTrajectory] = []
    for task in tasks:
        for _ in range(demos_per_task):
            actions = noisy_demonstration_actions(task, rng, noise_rate=noise_rate)
            demos.append(run_actions(task, actions))
    return demos


def demonstrations_to_examples(
    tasks_by_id: dict[int, SyntheticTask],
    trajectories: list[AgentTrajectory],
    *,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs: list[np.ndarray] = []
    labels: list[int] = []

    for trajectory in trajectories:
        task = tasks_by_id[trajectory.task_id]
        prior_actions: list[str] = []
        tool_outputs = []
        for action in trajectory.actions:
            inputs.append(
                state_features(
                    task,
                    tuple(prior_actions),
                    tuple(tool_outputs),
                    max_steps=max_steps,
                )
            )
            labels.append(ACTIONS.index(action))
            tool_call = run_tool(task, action)
            if tool_call is not None:
                tool_outputs.append(tool_call)
            prior_actions.append(action)

    return np.stack(inputs, axis=0), np.array(labels, dtype=np.int64)


def _candidate_action_sequences(task: SyntheticTask, rng: np.random.Generator) -> list[tuple[str, ...]]:
    ideal = ideal_action_sequence(task)
    required_tools = tuple(task.requirements.get("required_tools", ()))

    candidates = {
        ideal,
        ("answer_directly",),
        ("ask_clarification",),
        ("refuse",),
        ("search_tool", "answer_directly"),
        ("calculator_tool", "answer_directly"),
        ("memory_lookup", "answer_directly"),
        ("verify_answer", "answer_directly"),
        ("search_tool", "verify_answer", "answer_directly"),
        ("search_tool", "search_tool", "answer_directly"),
        ("memory_lookup", "search_tool", "answer_directly"),
        ("memory_lookup", "answer_directly"),
        ("memory_lookup", "search_tool", "verify_answer", "answer_directly"),
        ("calculator_tool", "verify_answer", "answer_directly"),
    }

    if required_tools:
        candidates.add(tuple(required_tools) + ("answer_directly",))
        shuffled = list(required_tools)
        rng.shuffle(shuffled)
        candidates.add(tuple(shuffled) + ("answer_directly",))

    return sorted(candidates)


def _append_preference(
    preferred: list[AgentTrajectory],
    rejected: list[AgentTrajectory],
    task: SyntheticTask,
    preferred_actions: tuple[str, ...],
    rejected_actions: tuple[str, ...],
) -> None:
    first = run_actions(task, preferred_actions)
    second = run_actions(task, rejected_actions)
    if first.true_reward > second.true_reward:
        preferred.append(first)
        rejected.append(second)
    elif second.true_reward > first.true_reward:
        preferred.append(second)
        rejected.append(first)


def generate_preference_pairs(
    tasks: list[SyntheticTask],
    rng: np.random.Generator,
    *,
    pairs_per_task: int,
) -> TrajectoryPreferenceDataset:
    preferred: list[AgentTrajectory] = []
    rejected: list[AgentTrajectory] = []

    for task in tasks:
        if task.task_type == "personalized_answer":
            _append_preference(
                preferred,
                rejected,
                task,
                ("memory_lookup", "answer_directly"),
                ("answer_directly",),
            )
        elif task.task_type == "math_calculation":
            _append_preference(
                preferred,
                rejected,
                task,
                ("calculator_tool", "verify_answer", "answer_directly"),
                ("search_tool", "answer_directly"),
            )
        elif task.task_type == "unsafe_request":
            _append_preference(
                preferred,
                rejected,
                task,
                ("refuse",),
                ("search_tool", "answer_directly"),
            )
        elif task.task_type == "ambiguous_request":
            _append_preference(
                preferred,
                rejected,
                task,
                ("ask_clarification",),
                ("answer_directly",),
            )

        trajectories = [run_actions(task, actions) for actions in _candidate_action_sequences(task, rng)]
        trajectories = sorted(trajectories, key=lambda item: item.true_reward)
        for _ in range(pairs_per_task):
            low_idx = int(rng.integers(0, max(1, len(trajectories) - 1)))
            high_idx = int(rng.integers(low_idx + 1, len(trajectories)))
            lower = trajectories[low_idx]
            higher = trajectories[high_idx]
            if np.isclose(higher.true_reward, lower.true_reward):
                continue
            preferred.append(higher)
            rejected.append(lower)

    return TrajectoryPreferenceDataset(preferred=tuple(preferred), rejected=tuple(rejected))
