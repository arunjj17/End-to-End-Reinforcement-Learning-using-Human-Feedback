from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .agent_env import ACTIONS, TASK_TYPES, TOOL_ACTIONS, SyntheticTask, ToolCall
from .tool_registry import recommended_tool_actions


PROMPT_FLAG_NAMES: tuple[str, ...] = (
    "asks_current_info",
    "contains_math",
    "mentions_profile",
    "looks_ambiguous",
    "looks_unsafe",
    "asks_comparison",
    "looks_static_fact",
)


def _one_hot(value: str, choices: tuple[str, ...]) -> np.ndarray:
    out = np.zeros(len(choices), dtype=np.float64)
    if value in choices:
        out[choices.index(value)] = 1.0
    return out


def prompt_flags(task: SyntheticTask) -> np.ndarray:
    text = task.prompt.lower()
    flags = np.array(
        [
            any(word in text for word in ("latest", "current", "status", "news feed")),
            any(word in text for word in ("calculate", "+", "-", "*", "/", "result")),
            any(word in text for word in ("my profile", "my ", "given my")),
            any(word in text for word in ("it", "that", "best option")),
            any(word in text for word in ("bypass", "steal", "malware", "api keys")),
            any(word in text for word in ("compare", "recommend one", " versus ", " vs ")),
            any(word in text for word in ("what is", "who wrote", "what gas")),
        ],
        dtype=np.float64,
    )
    return flags


def state_features(
    task: SyntheticTask,
    actions: tuple[str, ...],
    tool_outputs: tuple[ToolCall, ...],
    *,
    max_steps: int,
) -> np.ndarray:
    action_counts = np.array([actions.count(action) for action in ACTIONS], dtype=np.float64)
    if max_steps > 0:
        action_counts /= float(max_steps)

    last_action_choices = ("<none>",) + ACTIONS
    last_action = actions[-1] if actions else "<none>"

    tool_actions_with_output = {tool_call.action for tool_call in tool_outputs}
    recommended_actions = set(recommended_tool_actions(task.task_type))
    tool_flags = np.array(
        [
            "search_tool" in tool_actions_with_output,
            "calculator_tool" in tool_actions_with_output,
            "memory_lookup" in tool_actions_with_output,
            "verify_answer" in tool_actions_with_output,
            any("error" in tool_call.output.lower() for tool_call in tool_outputs),
        ],
        dtype=np.float64,
    )
    recommended_flags = np.array(
        [1.0 if action in recommended_actions else 0.0 for action in ACTIONS],
        dtype=np.float64,
    )
    memory_flags = np.array(
        [
            1.0 if task.memory else 0.0,
            1.0 if task.task_type == "personalized_answer" else 0.0,
            1.0 if task.requirements.get("memory_relevant", False) else 0.0,
        ],
        dtype=np.float64,
    )
    tool_cost = sum(tool_call.cost for tool_call in tool_outputs)
    tool_latency = sum(tool_call.latency for tool_call in tool_outputs)

    return np.concatenate(
        (
            np.ones(1, dtype=np.float64),
            _one_hot(task.task_type, TASK_TYPES),
            prompt_flags(task),
            memory_flags,
            recommended_flags,
            action_counts,
            _one_hot(last_action, last_action_choices),
            tool_flags,
            np.array(
                [
                    len(actions) / max(1.0, float(max_steps)),
                    tool_cost,
                    tool_latency,
                ],
                dtype=np.float64,
            ),
        )
    )


def state_feature_dim(max_steps: int) -> int:
    dummy_task = SyntheticTask(
        task_id=-1,
        task_type=TASK_TYPES[0],
        prompt="What is the capital of France?",
        memory={},
        requirements={},
    )
    return state_features(dummy_task, (), (), max_steps=max_steps).size


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


@dataclass
class AgentPolicy:
    """Softmax linear policy over discrete agent actions."""

    weights: np.ndarray  # shape: (num_actions, state_feature_dim)

    @classmethod
    def initialize(
        cls,
        feature_dim: int,
        rng: np.random.Generator,
        *,
        weight_scale: float = 0.02,
    ) -> "AgentPolicy":
        weights = rng.normal(
            scale=weight_scale / max(1.0, np.sqrt(feature_dim)),
            size=(len(ACTIONS), feature_dim),
        )
        return cls(weights=weights)

    def clone(self) -> "AgentPolicy":
        return AgentPolicy(weights=self.weights.copy())

    def probabilities_from_features(self, features: np.ndarray) -> np.ndarray:
        return softmax(self.weights @ features)

    def action_probabilities(
        self,
        task: SyntheticTask,
        actions: tuple[str, ...],
        tool_outputs: tuple[ToolCall, ...],
        *,
        max_steps: int,
    ) -> np.ndarray:
        features = state_features(task, actions, tool_outputs, max_steps=max_steps)
        return self.probabilities_from_features(features)

    def choose_action(
        self,
        task: SyntheticTask,
        actions: tuple[str, ...],
        tool_outputs: tuple[ToolCall, ...],
        rng: np.random.Generator,
        *,
        max_steps: int,
        greedy: bool,
    ) -> str:
        probs = self.action_probabilities(task, actions, tool_outputs, max_steps=max_steps)
        if greedy:
            return ACTIONS[int(np.argmax(probs))]
        action_idx = int(rng.choice(len(ACTIONS), p=probs))
        return ACTIONS[action_idx]

    def supervised_finetune(
        self,
        inputs: np.ndarray,
        action_labels: np.ndarray,
        *,
        epochs: int,
        lr: float,
        batch_size: int,
        l2_penalty: float,
        rng: np.random.Generator,
    ) -> list[dict[str, float]]:
        history: list[dict[str, float]] = []
        num_samples = inputs.shape[0]
        indices = np.arange(num_samples)

        for epoch in range(1, epochs + 1):
            rng.shuffle(indices)
            epoch_loss = 0.0
            epoch_correct = 0
            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = inputs[batch_idx]
                y_batch = action_labels[batch_idx]
                logits = x_batch @ self.weights.T
                logits -= np.max(logits, axis=1, keepdims=True)
                exp = np.exp(logits)
                probs = exp / np.sum(exp, axis=1, keepdims=True)

                epoch_loss += float(
                    -np.sum(np.log(probs[np.arange(len(batch_idx)), y_batch] + 1e-9))
                )
                epoch_correct += int(np.sum(np.argmax(probs, axis=1) == y_batch))

                targets = np.zeros_like(probs)
                targets[np.arange(len(batch_idx)), y_batch] = 1.0
                grad = ((probs - targets).T @ x_batch) / len(batch_idx)
                grad += l2_penalty * self.weights
                self.weights -= lr * grad

            history.append(
                {
                    "epoch": float(epoch),
                    "loss": epoch_loss / max(1, num_samples),
                    "accuracy": epoch_correct / max(1, num_samples),
                }
            )

        return history

    def grad_log_prob(self, features: np.ndarray, action_idx: int) -> np.ndarray:
        probs = self.probabilities_from_features(features)
        indicator = np.zeros(len(ACTIONS), dtype=np.float64)
        indicator[action_idx] = 1.0
        return (indicator - probs)[:, None] * features[None, :]

    def log_prob_from_features(self, features: np.ndarray, action_idx: int) -> float:
        probs = self.probabilities_from_features(features)
        return float(np.log(probs[action_idx] + 1e-12))

    def entropy_from_features(self, features: np.ndarray) -> float:
        probs = self.probabilities_from_features(features)
        return float(-np.sum(probs * np.log(probs + 1e-12)))


def tool_actions(actions: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(action for action in actions if action in TOOL_ACTIONS)
