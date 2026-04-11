from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SyntheticEnvironment
from .policy import Policy
from .reward_model import RewardModel


@dataclass
class EvaluationResult:
    mean_true_reward: float
    mean_predicted_reward: float


def evaluate_policy(
    policy: Policy,
    prompts: np.ndarray,
    env: SyntheticEnvironment,
    reward_model: RewardModel,
    rng: np.random.Generator,
    *,
    samples_per_prompt: int,
) -> EvaluationResult:
    true_rewards: list[float] = []
    predicted_rewards: list[float] = []
    for prompt in prompts:
        for _ in range(samples_per_prompt):
            response = policy.sample(prompt, rng)
            true_rewards.append(float(env.reward(prompt, response)))
            predicted_rewards.append(reward_model.score(prompt, response))
    return EvaluationResult(
        mean_true_reward=float(np.mean(true_rewards)),
        mean_predicted_reward=float(np.mean(predicted_rewards)),
    )


def estimate_true_reward(
    policy: Policy,
    prompts: np.ndarray,
    env: SyntheticEnvironment,
    rng: np.random.Generator,
    *,
    samples_per_prompt: int,
) -> float:
    rewards: list[float] = []
    for prompt in prompts:
        for _ in range(samples_per_prompt):
            response = policy.sample(prompt, rng)
            rewards.append(float(env.reward(prompt, response)))
    return float(np.mean(rewards))


def run_policy_gradient(
    policy: Policy,
    reward_model: RewardModel,
    prompts: np.ndarray,
    env: SyntheticEnvironment,
    rng: np.random.Generator,
    *,
    episodes: int,
    batch_size: int,
    lr: float,
    kl_coef: float,
    reference_policy: Policy | None,
    baseline_momentum: float = 0.9,
) -> list[dict[str, float]]:
    baseline = 0.0
    history: list[dict[str, float]] = []
    num_prompts = prompts.shape[0]

    for episode in range(1, episodes + 1):
        grad_w_accum = np.zeros_like(policy.weights)
        grad_b_accum = np.zeros_like(policy.bias)
        predicted_rewards: list[float] = []
        true_rewards: list[float] = []
        kl_terms: list[float] = []

        for _ in range(batch_size):
            prompt_idx = rng.integers(num_prompts)
            prompt = prompts[prompt_idx]
            response = policy.sample(prompt, rng)

            predicted_reward = reward_model.score(prompt, response)
            true_reward = float(env.reward(prompt, response))

            current_log_prob = policy.log_prob(prompt, response)
            if reference_policy is not None:
                reference_log_prob = reference_policy.log_prob(prompt, response)
            else:
                reference_log_prob = current_log_prob
            kl_increment = current_log_prob - reference_log_prob

            advantage = predicted_reward - baseline - kl_coef * kl_increment
            grad_w, grad_b = policy.grad_log_prob(prompt, response)

            grad_w_accum += advantage * grad_w
            grad_b_accum += advantage * grad_b

            predicted_rewards.append(predicted_reward)
            true_rewards.append(true_reward)
            kl_terms.append(kl_increment)

        grad_w_accum /= batch_size
        grad_b_accum /= batch_size

        policy.weights += lr * grad_w_accum
        policy.bias += lr * grad_b_accum

        batch_mean_reward = float(np.mean(predicted_rewards))
        baseline = baseline * baseline_momentum + batch_mean_reward * (1.0 - baseline_momentum)

        history.append(
            {
                "episode": float(episode),
                "predicted_reward": batch_mean_reward,
                "true_reward": float(np.mean(true_rewards)),
                "kl_term": float(np.mean(kl_terms)),
            }
        )

    return history
