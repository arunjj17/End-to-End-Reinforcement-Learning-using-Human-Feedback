from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class Policy:
    """Gaussian policy that maps prompt features to response features."""

    weights: np.ndarray  # shape: (response_dim, prompt_dim)
    bias: np.ndarray  # shape: (response_dim,)
    log_std: float  # shared log standard deviation across response dims

    @classmethod
    def initialize(
        cls,
        prompt_dim: int,
        response_dim: int,
        rng: np.random.Generator,
        *,
        weight_scale: float = 0.1,
        init_std: float = 0.6,
    ) -> "Policy":
        weights = rng.normal(
            scale=weight_scale / max(1.0, np.sqrt(prompt_dim)),
            size=(response_dim, prompt_dim),
        )
        bias = np.zeros(response_dim, dtype=np.float64)
        log_std = float(np.log(init_std))
        return cls(weights=weights, bias=bias, log_std=log_std)

    def clone(self) -> "Policy":
        return Policy(
            weights=self.weights.copy(),
            bias=self.bias.copy(),
            log_std=float(self.log_std),
        )

    @property
    def std(self) -> float:
        return float(np.exp(self.log_std))

    @property
    def variance(self) -> float:
        return float(np.exp(2.0 * self.log_std))

    def mean(self, prompt: np.ndarray) -> np.ndarray:
        prompt_vec = np.asarray(prompt, dtype=np.float64)
        return self.weights @ prompt_vec + self.bias

    def sample(
        self,
        prompt: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        mean = self.mean(prompt)
        noise = rng.normal(scale=self.std, size=mean.shape)
        return mean + noise

    def log_prob(self, prompt: np.ndarray, response: np.ndarray) -> float:
        mean = self.mean(prompt)
        diff = np.asarray(response, dtype=np.float64) - mean
        variance = self.variance
        normalization = mean.size * np.log(2.0 * np.pi * variance)
        quad = np.sum(diff**2) / variance
        return -0.5 * (normalization + quad)

    def grad_log_prob(
        self,
        prompt: np.ndarray,
        response: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        prompt_vec = np.asarray(prompt, dtype=np.float64)
        diff = np.asarray(response, dtype=np.float64) - self.mean(prompt_vec)
        variance = self.variance
        factor = diff / variance
        grad_weights = np.outer(factor, prompt_vec)
        grad_bias = factor
        return grad_weights, grad_bias

    def supervised_finetune(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
        l2_penalty: float = 1e-4,
        rng: np.random.Generator,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        num_samples = inputs.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(1, epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = inputs[batch_idx]
                y_batch = targets[batch_idx]
                preds = x_batch @ self.weights.T + self.bias
                residual = preds - y_batch

                # Mean squared error loss (per-sample average)
                batch_loss = 0.5 * np.mean(np.sum(residual**2, axis=1))
                epoch_loss += batch_loss * len(batch_idx)

                grad_weights = (residual.T @ x_batch) / len(batch_idx)
                grad_weights += l2_penalty * self.weights
                grad_bias = residual.mean(axis=0)

                self.weights -= lr * grad_weights
                self.bias -= lr * grad_bias

            epoch_loss /= num_samples
            history.append({"epoch": float(epoch), "loss": float(epoch_loss)})

        return history

    def evaluate_mse(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        preds = inputs @ self.weights.T + self.bias
        residual = preds - targets
        return float(np.mean(np.sum(residual**2, axis=1) / 2.0))
