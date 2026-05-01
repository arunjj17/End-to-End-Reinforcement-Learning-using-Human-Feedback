from __future__ import annotations

import numpy as np

from .agent_env import SyntheticTask
from .reward_features import trajectory_feature_dim, trajectory_features
from .reward_model import AgentRewardModel
from .trajectories import TrajectoryPreferenceDataset, generate_preference_pairs


def preference_feature_arrays(
    tasks_by_id: dict[int, SyntheticTask],
    preferences: TrajectoryPreferenceDataset,
    *,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    preferred = np.stack(
        [
            trajectory_features(tasks_by_id[trajectory.task_id], trajectory, max_steps=max_steps)
            for trajectory in preferences.preferred
        ],
        axis=0,
    )
    rejected = np.stack(
        [
            trajectory_features(tasks_by_id[trajectory.task_id], trajectory, max_steps=max_steps)
            for trajectory in preferences.rejected
        ],
        axis=0,
    )
    return preferred, rejected


def train_agent_reward_model(
    tasks: list[SyntheticTask],
    rng: np.random.Generator,
    *,
    max_steps: int,
    pairs_per_task: int,
    epochs: int,
    lr: float,
    batch_size: int,
    l2_penalty: float = 1e-2,
) -> tuple[AgentRewardModel, list[dict[str, float]], TrajectoryPreferenceDataset]:
    preferences = generate_preference_pairs(tasks, rng, pairs_per_task=pairs_per_task)
    tasks_by_id = {task.task_id: task for task in tasks}
    preferred_features, rejected_features = preference_feature_arrays(
        tasks_by_id,
        preferences,
        max_steps=max_steps,
    )
    reward_model = AgentRewardModel.initialize(trajectory_feature_dim(max_steps))
    history = reward_model.train(
        preferred_features,
        rejected_features,
        epochs=epochs,
        lr=lr,
        batch_size=max(1, batch_size),
        l2_penalty=l2_penalty,
        rng=rng,
    )
    return reward_model, history, preferences

