from __future__ import annotations

import numpy as np

from dataclasses import dataclass

from .features import build_features


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class RewardModel:
    """Pairwise preference model trained via logistic regression."""

    theta: np.ndarray  # shape: (feature_dim,)

    @classmethod
    def create_from_example(
        cls,
        prompt: np.ndarray,
        response: np.ndarray,
    ) -> "RewardModel":
        feature_dim = cls.featurize(prompt, response).size
        return cls(theta=np.zeros(feature_dim, dtype=np.float64))

    @staticmethod
    def featurize(prompt: np.ndarray, response: np.ndarray) -> np.ndarray:
        return build_features(prompt, response)

    def score(self, prompt: np.ndarray, response: np.ndarray) -> float:
        features = self.featurize(prompt, response)
        value = float(self.theta.dot(features))
        return float(np.clip(value, -5.0, 5.0))

    def predict_preference(
        self,
        prompts: np.ndarray,
        response_a: np.ndarray,
        response_b: np.ndarray,
    ) -> np.ndarray:
        logits = np.array(
            [
                self.score(prompt, a) - self.score(prompt, b)
                for prompt, a, b in zip(prompts, response_a, response_b, strict=True)
            ],
            dtype=np.float64,
        )
        return sigmoid(logits)

    def train(
        self,
        prompts: np.ndarray,
        response_a: np.ndarray,
        response_b: np.ndarray,
        labels: np.ndarray,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
        l2_penalty: float = 1e-2,
        rng: np.random.Generator,
    ) -> list[dict[str, float]]:
        num_samples = labels.shape[0]
        indices = np.arange(num_samples)

        # Pre-compute feature matrices for efficiency.
        feat_a = np.stack(
            [self.featurize(p, r) for p, r in zip(prompts, response_a, strict=True)],
            axis=0,
        )
        feat_b = np.stack(
            [self.featurize(p, r) for p, r in zip(prompts, response_b, strict=True)],
            axis=0,
        )

        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                logits = np.sum(
                    (feat_a[batch_idx] - feat_b[batch_idx]) * self.theta,
                    axis=1,
                )
                probs = sigmoid(logits)
                targets = labels[batch_idx]
                loss = -np.mean(
                    targets * np.log(probs + 1e-9)
                    + (1.0 - targets) * np.log(1.0 - probs + 1e-9)
                )
                epoch_loss += loss * len(batch_idx)

                error = probs - targets
                grad = (error[:, None] * (feat_a[batch_idx] - feat_b[batch_idx])).mean(axis=0)
                grad += l2_penalty * self.theta

                self.theta -= lr * grad

            epoch_loss /= num_samples
            accuracy = float(
                np.mean(
                    (np.sum((feat_a - feat_b) * self.theta, axis=1) > 0) == labels
                )
            )
            history.append({"epoch": float(epoch), "loss": float(epoch_loss), "accuracy": accuracy})

        return history


@dataclass
class AgentRewardModel:
    """Trajectory-level pairwise reward model for tool-using agents."""

    theta: np.ndarray  # shape: (trajectory_feature_dim,)

    @classmethod
    def initialize(cls, feature_dim: int) -> "AgentRewardModel":
        return cls(theta=np.zeros(feature_dim, dtype=np.float64))

    def score_features(self, features: np.ndarray) -> float:
        value = float(self.theta.dot(features))
        return float(np.clip(value, -6.0, 6.0))

    def train(
        self,
        preferred_features: np.ndarray,
        rejected_features: np.ndarray,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
        l2_penalty: float,
        rng: np.random.Generator,
    ) -> list[dict[str, float]]:
        num_samples = preferred_features.shape[0]
        indices = np.arange(num_samples)
        deltas = preferred_features - rejected_features
        history: list[dict[str, float]] = []

        for epoch in range(1, epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                logits = deltas[batch_idx] @ self.theta
                probs = sigmoid(logits)
                loss = -np.mean(np.log(probs + 1e-9))
                epoch_loss += loss * len(batch_idx)

                grad = ((probs - 1.0)[:, None] * deltas[batch_idx]).mean(axis=0)
                grad += l2_penalty * self.theta
                self.theta -= lr * grad

            logits_all = deltas @ self.theta
            accuracy = float(np.mean(logits_all > 0.0))
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": epoch_loss / max(1, num_samples),
                    "accuracy": accuracy,
                }
            )

        return history
