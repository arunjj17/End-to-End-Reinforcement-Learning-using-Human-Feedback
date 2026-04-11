from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np

from . import data, rl
from .policy import Policy
from .reward_model import RewardModel


def _format_history_tail(history: Iterable[dict[str, float]], tail: int) -> str:
    entries = list(history)[-tail:]
    formatted_lines = []
    for item in entries:
        epoch = item.get("epoch") or item.get("episode")
        metrics = [
            f"{key}={value:.4f}"
            for key, value in item.items()
            if key not in {"epoch", "episode"}
        ]
        formatted_lines.append(f"  {epoch:.0f}: " + ", ".join(metrics))
    return "\n".join(formatted_lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end synthetic RLHF pipeline.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--prompt-dim", type=int, default=6, help="Dimensionality of prompt features.")
    parser.add_argument("--response-dim", type=int, default=6, help="Dimensionality of response features.")
    parser.add_argument("--num-prompts", type=int, default=8, help="Number of unique prompts.")

    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=4,
        help="Number of supervised demonstrations per prompt.",
    )
    parser.add_argument(
        "--demonstration-noise",
        type=float,
        default=0.3,
        help="Noise scale applied to supervised demonstrations.",
    )
    parser.add_argument(
        "--sft-epochs",
        type=int,
        default=120,
        help="Supervised fine-tuning epochs.",
    )
    parser.add_argument(
        "--sft-lr",
        type=float,
        default=0.05,
        help="Learning rate for supervised fine-tuning.",
    )
    parser.add_argument(
        "--sft-batch-size",
        type=int,
        default=16,
        help="Batch size for supervised fine-tuning.",
    )

    parser.add_argument(
        "--pairs-per-prompt",
        type=int,
        default=12,
        help="Preference pairs per prompt.",
    )
    parser.add_argument(
        "--preference-weak-noise",
        type=float,
        default=0.35,
        help="Noise applied to preferred responses.",
    )
    parser.add_argument(
        "--preference-strong-noise",
        type=float,
        default=0.9,
        help="Noise applied to less preferred responses.",
    )
    parser.add_argument(
        "--reward-epochs",
        type=int,
        default=150,
        help="Reward model training epochs.",
    )
    parser.add_argument(
        "--reward-lr",
        type=float,
        default=0.1,
        help="Reward model learning rate.",
    )
    parser.add_argument(
        "--reward-batch-size",
        type=int,
        default=64,
        help="Reward model batch size.",
    )

    parser.add_argument(
        "--rl-episodes",
        type=int,
        default=160,
        help="Number of RL fine-tuning episodes.",
    )
    parser.add_argument(
        "--rl-batch-size",
        type=int,
        default=64,
        help="Batch size per RL episode.",
    )
    parser.add_argument(
        "--rl-lr",
        type=float,
        default=0.012,
        help="Learning rate for RL updates.",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.015,
        help="Coefficient for KL penalty against the reference policy.",
    )
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.9,
        help="Momentum term for the moving baseline during RL.",
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=64,
        help="Number of samples per prompt for reward evaluation.",
    )
    parser.add_argument(
        "--env-reward-noise",
        type=float,
        default=0.05,
        help="Observation noise applied to the hidden environment.",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    prompt_set = data.generate_prompts(args.num_prompts, args.prompt_dim, rng)
    environment = data.create_environment(
        args.prompt_dim,
        args.response_dim,
        rng,
        reward_noise=args.env_reward_noise,
    )

    sft_inputs, sft_targets = data.build_supervised_dataset(
        environment,
        prompt_set.features,
        rng,
        samples_per_prompt=args.samples_per_prompt,
        demonstration_noise=args.demonstration_noise,
    )

    policy_model = Policy.initialize(
        args.prompt_dim,
        args.response_dim,
        rng,
    )

    sft_history = policy_model.supervised_finetune(
        sft_inputs,
        sft_targets,
        epochs=args.sft_epochs,
        lr=args.sft_lr,
        batch_size=max(1, args.sft_batch_size),
        rng=rng,
    )

    baseline_true_reward = rl.estimate_true_reward(
        policy_model,
        prompt_set.features,
        environment,
        rng,
        samples_per_prompt=max(1, args.eval_samples),
    )

    preference_data = data.build_preference_dataset(
        environment,
        prompt_set.features,
        rng,
        pairs_per_prompt=args.pairs_per_prompt,
        strong_noise=args.preference_strong_noise,
        weak_noise=args.preference_weak_noise,
    )

    reward_model = RewardModel.create_from_example(
        prompt_set.features[0],
        preference_data.response_a[0],
    )
    reward_history = reward_model.train(
        preference_data.prompts,
        preference_data.response_a,
        preference_data.response_b,
        preference_data.labels,
        epochs=args.reward_epochs,
        lr=args.reward_lr,
        batch_size=max(1, args.reward_batch_size),
        rng=rng,
    )

    baseline_eval = rl.evaluate_policy(
        policy_model,
        prompt_set.features,
        environment,
        reward_model,
        rng,
        samples_per_prompt=max(1, args.eval_samples),
    )

    reference_policy = policy_model.clone()

    rl_history = rl.run_policy_gradient(
        policy_model,
        reward_model,
        prompt_set.features,
        environment,
        rng,
        episodes=args.rl_episodes,
        batch_size=max(1, args.rl_batch_size),
        lr=args.rl_lr,
        kl_coef=args.kl_coef,
        reference_policy=reference_policy,
        baseline_momentum=args.baseline_momentum,
    )

    final_eval = rl.evaluate_policy(
        policy_model,
        prompt_set.features,
        environment,
        reward_model,
        rng,
        samples_per_prompt=max(1, args.eval_samples),
    )

    print("=== Synthetic RLHF Pipeline ===")
    print(f"Prompts: {args.num_prompts}, prompt_dim={args.prompt_dim}, response_dim={args.response_dim}")
    print()
    print("Supervised fine-tuning (last 5 epochs):")
    print(_format_history_tail(sft_history, tail=min(5, len(sft_history))))
    print(f"Average true reward after SFT: {baseline_true_reward:.4f}")
    print()
    print("Reward model training (last 5 epochs):")
    print(_format_history_tail(reward_history, tail=min(5, len(reward_history))))
    print(f"Reward model final accuracy: {reward_history[-1]['accuracy']:.3f}")
    print()
    print("Policy optimization via RLHF (last 5 episodes):")
    print(_format_history_tail(rl_history, tail=min(5, len(rl_history))))
    print()
    print(
        f"Baseline predicted reward: {baseline_eval.mean_predicted_reward:.4f} | "
        f"Baseline true reward: {baseline_eval.mean_true_reward:.4f}"
    )
    print(
        f"Final predicted reward:    {final_eval.mean_predicted_reward:.4f} | "
        f"Final true reward:    {final_eval.mean_true_reward:.4f}"
    )
    print(
        f"True reward improvement: {final_eval.mean_true_reward - baseline_eval.mean_true_reward:.4f}"
    )


if __name__ == "__main__":
    main()
