from __future__ import annotations

import argparse
import sys
from typing import Iterable

import numpy as np

from .agent_env import ACTIONS, generate_synthetic_tasks, visible_memory_entries
from .benchmark import ModelBenchmarkResult, benchmark_ollama_models
from .ollama_client import OllamaClient
from .train_reward import train_agent_reward_model
from .train_rl import AgentEvaluation, evaluate_agent_policy, rollout_policy, run_agent_policy_gradient
from .train_sft import train_agent_sft_policy


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


def _format_eval(title: str, result: AgentEvaluation) -> str:
    return "\n".join(
        [
            f"{title}:",
            f"- true reward: {result.true_reward:.4f}",
            f"- predicted reward: {result.predicted_reward:.4f}",
            f"- tool accuracy: {result.tool_accuracy:.3f}",
            f"- memory accuracy: {result.memory_usage_accuracy:.3f}",
            f"- style match: {result.style_match_score:.3f}",
            f"- safety accuracy: {result.safety_accuracy:.3f}",
            f"- avg trajectory length: {result.avg_trajectory_length:.2f}",
            f"- avg tool cost: {result.avg_tool_cost:.3f}",
            f"- avg tool latency: {result.avg_tool_latency:.3f}",
            f"- unnecessary tool calls: {result.unnecessary_tool_calls:.3f}",
        ]
    )


def _format_actions(actions: tuple[str, ...]) -> str:
    return " -> ".join(actions)


def _format_memory(memory: dict[str, str]) -> str:
    if not memory:
        return "none"
    return "; ".join(f"{key}={value}" for key, value in memory.items())


def _format_benchmark_table(results: list[ModelBenchmarkResult]) -> str:
    lines = [
        "model                 tasks reward  tool  memory style safety len  cost latency unnec parse",
        "--------------------- ----- ------- ----- ------ ----- ------ ---- ---- ------- ----- -----",
    ]
    for result in results:
        if not result.available:
            lines.append(f"{result.model[:21]:21} {'skip':>5} unavailable")
            continue
        lines.append(
            f"{result.model[:21]:21} "
            f"{result.num_tasks:5d} "
            f"{result.true_reward:7.3f} "
            f"{result.tool_accuracy:5.3f} "
            f"{result.memory_accuracy:6.3f} "
            f"{result.style_match:5.3f} "
            f"{result.safety_accuracy:6.3f} "
            f"{result.avg_trajectory_length:4.2f} "
            f"{result.avg_tool_cost:4.2f} "
            f"{result.avg_tool_latency:7.2f} "
            f"{result.unnecessary_tool_calls:5.2f} "
            f"{result.parse_success_rate:5.3f}"
        )
    return "\n".join(lines)


def build_benchmark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rlhf-pipeline benchmark-models",
        description="Benchmark local open-source Ollama models on synthetic agentic tasks.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local Ollama model names to benchmark, for example qwen2.5:3b phi3:mini.",
    )
    parser.add_argument("--num-tasks", type=int, default=50, help="Number of synthetic tasks.")
    parser.add_argument("--seed", type=int, default=17, help="Synthetic task seed.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum actions per model trajectory.")
    parser.add_argument(
        "--ollama-base-url",
        type=str,
        default="http://localhost:11434",
        help="Local Ollama HTTP API base URL.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=12.0,
        help="Per-request Ollama timeout in seconds.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Ollama sampling temperature for action selection.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=2,
        help="Examples to print per available model.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AgentRLHF with synthetic tool-using agent trajectories. "
            "Use `rlhf-pipeline benchmark-models --models ...` for local Ollama benchmarks."
        ),
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument("--num-tasks", type=int, default=42, help="Number of synthetic user tasks.")
    parser.add_argument("--max-steps", type=int, default=4, help="Maximum actions per trajectory.")

    parser.add_argument(
        "--sft-demos-per-task",
        type=int,
        default=5,
        help="Noisy supervised demonstrations per task.",
    )
    parser.add_argument(
        "--sft-noise-rate",
        type=float,
        default=0.34,
        help="Probability of corrupting an otherwise good SFT demonstration.",
    )
    parser.add_argument("--sft-epochs", type=int, default=90, help="SFT training epochs.")
    parser.add_argument("--sft-lr", type=float, default=0.22, help="SFT learning rate.")
    parser.add_argument("--sft-batch-size", type=int, default=48, help="SFT batch size.")

    parser.add_argument(
        "--pairs-per-task",
        type=int,
        default=16,
        help="Trajectory preference pairs per task.",
    )
    parser.add_argument("--reward-epochs", type=int, default=120, help="Reward model epochs.")
    parser.add_argument("--reward-lr", type=float, default=0.18, help="Reward model learning rate.")
    parser.add_argument("--reward-batch-size", type=int, default=96, help="Reward batch size.")

    parser.add_argument("--rl-episodes", type=int, default=170, help="RL policy-gradient episodes.")
    parser.add_argument("--rl-batch-size", type=int, default=80, help="Trajectories per RL batch.")
    parser.add_argument("--rl-lr", type=float, default=0.08, help="RL learning rate.")
    parser.add_argument("--kl-coef", type=float, default=0.08, help="KL penalty against SFT policy.")
    parser.add_argument(
        "--baseline-momentum",
        type=float,
        default=0.9,
        help="Moving reward baseline momentum.",
    )

    parser.add_argument(
        "--eval-samples",
        type=int,
        default=20,
        help="Sampled trajectories per task for evaluation.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=3,
        help="Number of before/after task examples to print.",
    )

    parser.add_argument(
        "--use-ollama",
        action="store_true",
        help="Optionally ask local Ollama to verbalize final answers for examples.",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default="llama3.2",
        help="Local Ollama model to use when --use-ollama is set.",
    )
    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=4.0,
        help="Short Ollama HTTP timeout in seconds.",
    )

    # Backward-compatible aliases or ignored knobs from the original continuous demo.
    parser.add_argument("--num-prompts", dest="num_tasks", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--samples-per-prompt", dest="sft_demos_per_task", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--pairs-per-prompt", dest="pairs_per_task", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--demonstration-noise", dest="sft_noise_rate", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--prompt-dim", dest="_ignored_prompt_dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--response-dim", dest="_ignored_response_dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--preference-weak-noise", dest="_ignored_weak_noise", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--preference-strong-noise", dest="_ignored_strong_noise", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--env-reward-noise", dest="_ignored_env_noise", type=float, help=argparse.SUPPRESS)

    return parser


def run_benchmark_command(argv: list[str]) -> None:
    parser = build_benchmark_parser()
    args = parser.parse_args(argv)
    results = benchmark_ollama_models(
        args.models,
        num_tasks=args.num_tasks,
        seed=args.seed,
        max_steps=args.max_steps,
        base_url=args.ollama_base_url,
        timeout=args.ollama_timeout,
        temperature=args.temperature,
        example_count=max(0, args.example_count),
    )

    print("=== Open-Source Local LLM Agent Benchmark ===")
    print("Backend: local Ollama only")
    print(f"Tasks: {args.num_tasks}, max_steps={args.max_steps}, seed={args.seed}")
    print()
    print(_format_benchmark_table(results))

    warnings = [result for result in results if result.warning]
    if warnings:
        print()
        print("Warnings:")
        for result in warnings:
            print(f"- {result.model}: {result.warning}")

    available_results = [result for result in results if result.available]
    if not available_results:
        print()
        print("No models were benchmarked. Start Ollama and pull the requested local models first.")
        return

    print()
    print("Examples:")
    for result in available_results:
        print(f"{result.model}:")
        for example in result.examples:
            print(f"  prompt: {example.task.prompt}")
            print(f"  task type: {example.task.task_type}")
            print(f"  memory: {_format_memory(visible_memory_entries(example.task))}")
            print(f"  actions: {_format_actions(example.actions)}")
            print(
                f"  reward/tool cost/latency: "
                f"{example.trajectory.true_reward:.3f}/"
                f"{example.trajectory.total_tool_cost:.3f}/"
                f"{example.trajectory.total_tool_latency:.3f}"
            )
            print(f"  parse ok: {example.parse_ok}")


def main(argv: list[str] | None = None) -> None:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if raw_args and raw_args[0] == "benchmark-models":
        run_benchmark_command(raw_args[1:])
        return

    parser = build_parser()
    args = parser.parse_args(raw_args)

    rng = np.random.default_rng(args.seed)
    tasks = generate_synthetic_tasks(args.num_tasks, rng)

    policy, sft_history = train_agent_sft_policy(
        tasks,
        rng,
        max_steps=args.max_steps,
        demos_per_task=args.sft_demos_per_task,
        noise_rate=float(np.clip(args.sft_noise_rate, 0.0, 1.0)),
        epochs=args.sft_epochs,
        lr=args.sft_lr,
        batch_size=args.sft_batch_size,
    )

    reward_model, reward_history, _preferences = train_agent_reward_model(
        tasks,
        rng,
        max_steps=args.max_steps,
        pairs_per_task=args.pairs_per_task,
        epochs=args.reward_epochs,
        lr=args.reward_lr,
        batch_size=args.reward_batch_size,
    )

    baseline_eval = evaluate_agent_policy(
        policy,
        reward_model,
        tasks,
        np.random.default_rng(args.seed + 101),
        max_steps=args.max_steps,
        samples_per_task=max(1, args.eval_samples),
        greedy=False,
    )

    reference_policy = policy.clone()
    rl_history = run_agent_policy_gradient(
        policy,
        reward_model,
        tasks,
        rng,
        max_steps=args.max_steps,
        episodes=args.rl_episodes,
        batch_size=max(1, args.rl_batch_size),
        lr=args.rl_lr,
        kl_coef=args.kl_coef,
        reference_policy=reference_policy,
        baseline_momentum=args.baseline_momentum,
    )

    final_eval = evaluate_agent_policy(
        policy,
        reward_model,
        tasks,
        np.random.default_rng(args.seed + 101),
        max_steps=args.max_steps,
        samples_per_task=max(1, args.eval_samples),
        greedy=False,
    )

    print("=== AgentRLHF: Synthetic RLHF for Tool-Using AI Agents ===")
    print("Model policy: local NumPy softmax policy")
    print("LLM backend: synthetic/template fallback by default; optional local Ollama only")
    print(f"Tasks: {args.num_tasks}, actions={len(ACTIONS)}, max_steps={args.max_steps}")
    print()
    print("SFT training (last 5 epochs):")
    print(_format_history_tail(sft_history, tail=min(5, len(sft_history))))
    print()
    print("Reward model training (last 5 epochs):")
    print(_format_history_tail(reward_history, tail=min(5, len(reward_history))))
    print()
    print("Policy optimization via Agent RLHF (last 5 episodes):")
    print(_format_history_tail(rl_history, tail=min(5, len(rl_history))))
    print()
    print(_format_eval("SFT baseline", baseline_eval))
    print()
    print(_format_eval("After Agent RLHF", final_eval))
    print()
    print(f"True reward improvement: {final_eval.true_reward - baseline_eval.true_reward:.4f}")

    example_rows = []
    for task in tasks:
        for attempt in range(12):
            example_seed = args.seed + task.task_id * 31 + attempt
            sft_trace = rollout_policy(
                reference_policy,
                task,
                np.random.default_rng(example_seed),
                max_steps=args.max_steps,
                greedy=False,
            ).trajectory
            rl_trace = rollout_policy(
                policy,
                task,
                np.random.default_rng(example_seed),
                max_steps=args.max_steps,
                greedy=False,
            ).trajectory
            example_rows.append(
                (rl_trace.true_reward - sft_trace.true_reward, task, sft_trace, rl_trace)
            )
    example_rows.sort(
        key=lambda item: (item[0], item[3].true_reward, -len(item[3].actions)),
        reverse=True,
    )
    distinct_examples = []
    seen_task_types: set[str] = set()
    for row in example_rows:
        task_type = row[1].task_type
        if task_type in seen_task_types:
            continue
        distinct_examples.append(row)
        seen_task_types.add(task_type)

    ollama_client = None
    printed_warnings: set[str] = set()
    if args.use_ollama:
        ollama_client = OllamaClient(model=args.ollama_model, timeout=args.ollama_timeout)

    print()
    print("Example tasks:")
    for idx, (improvement, task, sft_trace, rl_trace) in enumerate(
        distinct_examples[: max(1, args.example_count)],
        start=1,
    ):
        print(f"{idx}. prompt: {task.prompt}")
        print(f"   task type: {task.task_type}")
        print(f"   memory: {_format_memory(visible_memory_entries(task))}")
        print(f"   SFT trajectory: {_format_actions(sft_trace.actions)} ({sft_trace.true_reward:.3f})")
        print(f"   Agent RLHF trajectory: {_format_actions(rl_trace.actions)} ({rl_trace.true_reward:.3f})")
        print(
            f"   Agent RLHF tool cost/latency: "
            f"{rl_trace.total_tool_cost:.3f}/{rl_trace.total_tool_latency:.3f}"
        )
        print(f"   reward improvement: {improvement:.3f}")
        if ollama_client is not None:
            answer = ollama_client.generate_final_answer(task, rl_trace)
            if answer.warning and answer.warning not in printed_warnings:
                print(f"   warning: {answer.warning}")
                printed_warnings.add(answer.warning)
            source = "Ollama" if answer.used_ollama else "synthetic/template"
            print(f"   final answer ({source}): {answer.text}")


if __name__ == "__main__":
    main()
