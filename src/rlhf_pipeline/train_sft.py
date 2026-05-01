from __future__ import annotations

import numpy as np

from .agent_env import SyntheticTask
from .agent_policy import AgentPolicy, state_feature_dim
from .trajectories import demonstrations_to_examples, generate_sft_demonstrations


def train_agent_sft_policy(
    tasks: list[SyntheticTask],
    rng: np.random.Generator,
    *,
    max_steps: int,
    demos_per_task: int,
    noise_rate: float,
    epochs: int,
    lr: float,
    batch_size: int,
    l2_penalty: float = 1e-4,
) -> tuple[AgentPolicy, list[dict[str, float]]]:
    demos = generate_sft_demonstrations(
        tasks,
        rng,
        demos_per_task=demos_per_task,
        noise_rate=noise_rate,
    )
    tasks_by_id = {task.task_id: task for task in tasks}
    inputs, labels = demonstrations_to_examples(tasks_by_id, demos, max_steps=max_steps)

    policy = AgentPolicy.initialize(state_feature_dim(max_steps), rng)
    history = policy.supervised_finetune(
        inputs,
        labels,
        epochs=epochs,
        lr=lr,
        batch_size=max(1, batch_size),
        l2_penalty=l2_penalty,
        rng=rng,
    )
    return policy, history

