from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent_env import (
    ACTIONS,
    TERMINAL_ACTIONS,
    AgentTrajectory,
    SyntheticTask,
    run_actions,
    run_tool,
    trajectory_diagnostics,
)
from .agent_policy import AgentPolicy, state_features
from .reward_features import trajectory_features
from .reward_model import AgentRewardModel


@dataclass(frozen=True)
class AgentEvaluation:
    true_reward: float
    predicted_reward: float
    tool_accuracy: float
    unnecessary_tool_calls: float
    safety_accuracy: float
    memory_usage_accuracy: float
    style_match_score: float
    avg_trajectory_length: float
    avg_tool_cost: float
    avg_tool_latency: float


@dataclass(frozen=True)
class RolloutTrace:
    trajectory: AgentTrajectory
    state_feature_rows: tuple[np.ndarray, ...]
    action_indices: tuple[int, ...]
    log_ratio: float


def rollout_policy(
    policy: AgentPolicy,
    task: SyntheticTask,
    rng: np.random.Generator,
    *,
    max_steps: int,
    greedy: bool,
    reference_policy: AgentPolicy | None = None,
) -> RolloutTrace:
    actions: list[str] = []
    tool_outputs = []
    feature_rows: list[np.ndarray] = []
    action_indices: list[int] = []
    log_ratio = 0.0

    for _ in range(max_steps):
        features = state_features(task, tuple(actions), tuple(tool_outputs), max_steps=max_steps)
        probs = policy.probabilities_from_features(features)
        if greedy:
            action_idx = int(np.argmax(probs))
        else:
            action_idx = int(rng.choice(len(ACTIONS), p=probs))
        action = ACTIONS[action_idx]

        feature_rows.append(features)
        action_indices.append(action_idx)
        if reference_policy is not None:
            logp_current = policy.log_prob_from_features(features, action_idx)
            logp_reference = reference_policy.log_prob_from_features(features, action_idx)
            log_ratio += logp_current - logp_reference

        actions.append(action)
        tool_call = run_tool(task, action)
        if tool_call is not None:
            tool_outputs.append(tool_call)
        if action in TERMINAL_ACTIONS:
            break

    trajectory = run_actions(task, actions)
    return RolloutTrace(
        trajectory=trajectory,
        state_feature_rows=tuple(feature_rows),
        action_indices=tuple(action_indices),
        log_ratio=log_ratio / max(1, len(action_indices)),
    )


def evaluate_agent_policy(
    policy: AgentPolicy,
    reward_model: AgentRewardModel,
    tasks: list[SyntheticTask],
    rng: np.random.Generator,
    *,
    max_steps: int,
    samples_per_task: int,
    greedy: bool,
) -> AgentEvaluation:
    true_rewards: list[float] = []
    predicted_rewards: list[float] = []
    tool_accuracy: list[float] = []
    unnecessary: list[int] = []
    safety: list[float] = []
    memory: list[float] = []
    style: list[float] = []
    lengths: list[int] = []
    tool_costs: list[float] = []
    tool_latencies: list[float] = []

    for task in tasks:
        for _ in range(samples_per_task):
            trace = rollout_policy(policy, task, rng, max_steps=max_steps, greedy=greedy)
            trajectory = trace.trajectory
            diagnostics = trajectory_diagnostics(task, trajectory.actions)
            features = trajectory_features(task, trajectory, max_steps=max_steps)
            true_rewards.append(trajectory.true_reward)
            predicted_rewards.append(reward_model.score_features(features))
            tool_accuracy.append(diagnostics.tool_accuracy)
            unnecessary.append(diagnostics.unnecessary_tool_calls)
            if diagnostics.safety_accuracy is not None:
                safety.append(diagnostics.safety_accuracy)
            memory.append(diagnostics.memory_usage_accuracy)
            if diagnostics.style_match_score is not None:
                style.append(diagnostics.style_match_score)
            lengths.append(len(trajectory.actions))
            tool_costs.append(trajectory.total_tool_cost)
            tool_latencies.append(trajectory.total_tool_latency)

    return AgentEvaluation(
        true_reward=float(np.mean(true_rewards)),
        predicted_reward=float(np.mean(predicted_rewards)),
        tool_accuracy=float(np.mean(tool_accuracy)),
        unnecessary_tool_calls=float(np.mean(unnecessary)),
        safety_accuracy=float(np.mean(safety)) if safety else 0.0,
        memory_usage_accuracy=float(np.mean(memory)) if memory else 0.0,
        style_match_score=float(np.mean(style)) if style else 0.0,
        avg_trajectory_length=float(np.mean(lengths)),
        avg_tool_cost=float(np.mean(tool_costs)),
        avg_tool_latency=float(np.mean(tool_latencies)),
    )


def run_agent_policy_gradient(
    policy: AgentPolicy,
    reward_model: AgentRewardModel,
    tasks: list[SyntheticTask],
    rng: np.random.Generator,
    *,
    max_steps: int,
    episodes: int,
    batch_size: int,
    lr: float,
    kl_coef: float,
    reference_policy: AgentPolicy,
    baseline_momentum: float,
    grad_clip_norm: float = 5.0,
) -> list[dict[str, float]]:
    baseline = 0.0
    history: list[dict[str, float]] = []

    for episode in range(1, episodes + 1):
        grad_accum = np.zeros_like(policy.weights)
        predicted_rewards: list[float] = []
        true_rewards: list[float] = []
        kl_terms: list[float] = []
        lengths: list[int] = []

        for _ in range(batch_size):
            task = tasks[int(rng.integers(0, len(tasks)))]
            trace = rollout_policy(
                policy,
                task,
                rng,
                max_steps=max_steps,
                greedy=False,
                reference_policy=reference_policy,
            )
            features = trajectory_features(task, trace.trajectory, max_steps=max_steps)
            predicted_reward = reward_model.score_features(features)
            advantage = predicted_reward - baseline - kl_coef * trace.log_ratio

            trajectory_grad = np.zeros_like(policy.weights)
            for state_feature_row, action_idx in zip(
                trace.state_feature_rows,
                trace.action_indices,
                strict=True,
            ):
                trajectory_grad += policy.grad_log_prob(state_feature_row, action_idx)
            trajectory_grad /= max(1, len(trace.action_indices))
            grad_accum += advantage * trajectory_grad

            predicted_rewards.append(predicted_reward)
            true_rewards.append(trace.trajectory.true_reward)
            kl_terms.append(trace.log_ratio)
            lengths.append(len(trace.trajectory.actions))

        grad_accum /= max(1, batch_size)
        grad_norm = float(np.linalg.norm(grad_accum))
        if grad_norm > grad_clip_norm:
            grad_accum *= grad_clip_norm / (grad_norm + 1e-12)
        policy.weights += lr * grad_accum

        batch_predicted = float(np.mean(predicted_rewards))
        baseline = baseline * baseline_momentum + batch_predicted * (1.0 - baseline_momentum)
        history.append(
            {
                "episode": float(episode),
                "predicted_reward": batch_predicted,
                "true_reward": float(np.mean(true_rewards)),
                "kl_term": float(np.mean(kl_terms)),
                "avg_length": float(np.mean(lengths)),
            }
        )

    return history
