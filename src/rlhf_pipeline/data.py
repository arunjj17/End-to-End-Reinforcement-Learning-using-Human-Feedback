from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .features import build_features


DEFAULT_PROMPTS: Sequence[str] = (
    "Summarize the key idea in one sentence.",
    "Provide a polite follow-up question for the user.",
    "Offer constructive feedback on a draft paragraph.",
    "Suggest a creative title for a short story.",
    "Explain a technical concept for a beginner.",
    "Brainstorm two ways to save time on a daily task.",
    "Write a friendly reminder message.",
    "List the pros and cons of working remotely.",
)


@dataclass(frozen=True)
class PromptSet:
    """Container for synthetic prompts."""

    texts: list[str]
    features: np.ndarray  # shape: (num_prompts, prompt_dim)


@dataclass(frozen=True)
class PreferenceDataset:
    """Pairwise preferences derived from the hidden environment."""

    prompts: np.ndarray  # shape: (num_pairs, prompt_dim)
    response_a: np.ndarray  # shape: (num_pairs, response_dim)
    response_b: np.ndarray  # shape: (num_pairs, response_dim)
    labels: np.ndarray  # shape: (num_pairs,)


@dataclass
class SyntheticEnvironment:
    """Hidden ground-truth mapping from prompts to ideal responses."""

    true_matrix: np.ndarray  # shape: (response_dim, prompt_dim)
    true_bias: np.ndarray  # shape: (response_dim,)
    reward_weights: np.ndarray  # shape: (feature_dim,)
    reward_noise: float = 0.0

    def ideal_response(self, prompts: np.ndarray) -> np.ndarray:
        prompts_2d = np.atleast_2d(prompts)
        response = prompts_2d @ self.true_matrix.T + self.true_bias
        return response if prompts_2d.shape[0] > 1 else response[0]

    def reward(
        self,
        prompts: np.ndarray,
        responses: np.ndarray,
        *,
        stochastic: bool = False,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        prompts_2d = np.atleast_2d(prompts)
        responses_2d = np.atleast_2d(responses)
        features = np.stack(
            [build_features(p, r) for p, r in zip(prompts_2d, responses_2d, strict=True)],
            axis=0,
        )
        score = features @ self.reward_weights
        score = np.clip(score, -5.0, 5.0)
        if stochastic and self.reward_noise > 0:
            if rng is None:
                rng = np.random.default_rng()
            score = score + rng.normal(scale=self.reward_noise, size=score.shape)
        return score if score.shape[0] > 1 else score[0]


def create_environment(
    prompt_dim: int,
    response_dim: int,
    rng: np.random.Generator,
    *,
    mapping_scale: float = 0.8,
    bias_scale: float = 0.5,
    reward_noise: float = 0.1,
    reward_scale: float = 0.4,
) -> SyntheticEnvironment:
    matrix = rng.normal(
        scale=mapping_scale / max(1.0, np.sqrt(prompt_dim)),
        size=(response_dim, prompt_dim),
    )
    bias = rng.normal(scale=bias_scale, size=(response_dim,))
    feature_dim = build_features(
        np.zeros(prompt_dim, dtype=np.float64),
        np.zeros(response_dim, dtype=np.float64),
    ).size
    reward_weights = rng.normal(scale=reward_scale / np.sqrt(feature_dim), size=feature_dim)
    return SyntheticEnvironment(matrix, bias, reward_weights, reward_noise=reward_noise)


def generate_prompts(
    num_prompts: int,
    prompt_dim: int,
    rng: np.random.Generator,
) -> PromptSet:
    texts: list[str] = []
    vectors: list[np.ndarray] = []
    for idx in range(num_prompts):
        base = DEFAULT_PROMPTS[idx % len(DEFAULT_PROMPTS)]
        variant = idx // len(DEFAULT_PROMPTS)
        text = base if variant == 0 else f"{base} (variant {variant})"
        texts.append(text)
        vectors.append(rng.normal(size=(prompt_dim,)))
    features = np.stack(vectors, axis=0)
    return PromptSet(texts=texts, features=features)


def build_supervised_dataset(
    env: SyntheticEnvironment,
    prompt_features: np.ndarray,
    rng: np.random.Generator,
    *,
    samples_per_prompt: int,
    demonstration_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    prompts_repeated = np.repeat(prompt_features, samples_per_prompt, axis=0)
    ideal = np.repeat(
        np.atleast_2d(env.ideal_response(prompt_features)),
        samples_per_prompt,
        axis=0,
    )
    noisy_targets = ideal + rng.normal(
        scale=demonstration_noise,
        size=ideal.shape,
    )
    return prompts_repeated, noisy_targets


def build_preference_dataset(
    env: SyntheticEnvironment,
    prompt_features: np.ndarray,
    rng: np.random.Generator,
    *,
    pairs_per_prompt: int,
    strong_noise: float,
    weak_noise: float,
) -> PreferenceDataset:
    prompts_repeated = np.repeat(prompt_features, pairs_per_prompt, axis=0)
    ideal = np.repeat(
        np.atleast_2d(env.ideal_response(prompt_features)),
        pairs_per_prompt,
        axis=0,
    )

    noise_strong = rng.normal(scale=strong_noise, size=ideal.shape)
    noise_weak = rng.normal(scale=weak_noise, size=ideal.shape)

    response_a = ideal + noise_weak  # typically closer to ideal
    response_b = ideal + noise_strong

    reward_a = env.reward(prompts_repeated, response_a)
    reward_b = env.reward(prompts_repeated, response_b)

    labels = (reward_a > reward_b).astype(np.float64)
    ties = np.isclose(reward_a, reward_b)
    if np.any(ties):
        labels[ties] = rng.uniform(size=np.count_nonzero(ties)) > 0.5

    return PreferenceDataset(
        prompts=prompts_repeated,
        response_a=response_a,
        response_b=response_b,
        labels=labels,
    )
