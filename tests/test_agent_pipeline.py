from __future__ import annotations

import contextlib
import io
import unittest

import numpy as np

from rlhf_pipeline.agent_env import (
    generate_synthetic_tasks,
    ideal_action_sequence,
    run_actions,
    run_tool,
    trajectory_diagnostics,
)
from rlhf_pipeline.benchmark import parse_action_response
from rlhf_pipeline.cli import main as cli_main
from rlhf_pipeline.ollama_client import OllamaClient
from rlhf_pipeline.tool_registry import LOCAL_TOOL_REGISTRY
from rlhf_pipeline.train_reward import train_agent_reward_model
from rlhf_pipeline.train_rl import evaluate_agent_policy, run_agent_policy_gradient
from rlhf_pipeline.train_sft import train_agent_sft_policy
from rlhf_pipeline.trajectories import generate_preference_pairs


class AgentPipelineTest(unittest.TestCase):
    def test_tool_registry_contains_required_metadata(self) -> None:
        required = {"search_tool", "calculator_tool", "memory_lookup", "verify_answer"}
        self.assertEqual(required, set(LOCAL_TOOL_REGISTRY))

        for action, tool in LOCAL_TOOL_REGISTRY.items():
            metadata = tool.metadata
            self.assertEqual(action, metadata.action)
            self.assertTrue(metadata.name)
            self.assertTrue(metadata.description)
            self.assertGreaterEqual(metadata.cost, 0.0)
            self.assertGreaterEqual(metadata.latency, 0.0)
            self.assertIn(metadata.risk, {"low", "medium", "high"})
            self.assertTrue(metadata.recommended_task_types)

    def test_memory_generation_is_deterministic(self) -> None:
        tasks_a = generate_synthetic_tasks(14, np.random.default_rng(7))
        tasks_b = generate_synthetic_tasks(14, np.random.default_rng(7))
        memories_a = [task.memory for task in tasks_a if task.task_type == "personalized_answer"]
        memories_b = [task.memory for task in tasks_b if task.task_type == "personalized_answer"]

        self.assertEqual(memories_a, memories_b)
        self.assertTrue(memories_a)
        self.assertIn("preferred_language", memories_a[0])

    def test_memory_lookup_returns_relevant_entries(self) -> None:
        task = next(
            task
            for task in generate_synthetic_tasks(7, np.random.default_rng(5))
            if task.task_type == "personalized_answer"
        )

        tool_call = run_tool(task, "memory_lookup")

        self.assertIsNotNone(tool_call)
        assert tool_call is not None
        self.assertEqual(tool_call.tool_name, "memory_lookup")
        self.assertIn("answer_style", tool_call.output)
        self.assertIn("preferred_language", tool_call.output)
        self.assertNotIn("_style_key", tool_call.output)

    def test_ideal_trajectories_beat_bad_direct_answers(self) -> None:
        rng = np.random.default_rng(3)
        tasks = generate_synthetic_tasks(14, rng)

        for task in tasks:
            ideal = run_actions(task, ideal_action_sequence(task))
            if task.task_type in {
                "current_info_required",
                "math_calculation",
                "personalized_answer",
                "comparison_task",
            }:
                bad = run_actions(task, ("answer_directly",))
                self.assertGreater(ideal.true_reward, bad.true_reward, task.task_type)
            elif task.task_type == "unsafe_request":
                bad = run_actions(task, ("answer_directly",))
                self.assertGreater(ideal.true_reward, bad.true_reward, task.task_type)
            elif task.task_type == "ambiguous_request":
                bad = run_actions(task, ("answer_directly",))
                self.assertGreater(ideal.true_reward, bad.true_reward, task.task_type)

    def test_personalized_tasks_reward_memory_and_style(self) -> None:
        task = next(
            task
            for task in generate_synthetic_tasks(7, np.random.default_rng(13))
            if task.task_type == "personalized_answer"
        )

        memory_aware = run_actions(task, ("memory_lookup", "answer_directly"))
        direct = run_actions(task, ("answer_directly",))

        self.assertGreater(memory_aware.true_reward, direct.true_reward)
        self.assertGreater(memory_aware.style_match_score, direct.style_match_score)

    def test_non_personalized_tasks_penalize_unnecessary_memory_lookup(self) -> None:
        task = next(
            task
            for task in generate_synthetic_tasks(7, np.random.default_rng(21))
            if task.task_type == "factual_static"
        )

        direct = run_actions(task, ("answer_directly",))
        with_memory = run_actions(task, ("memory_lookup", "answer_directly"))
        diagnostics = trajectory_diagnostics(task, with_memory.actions)

        self.assertGreater(direct.true_reward, with_memory.true_reward)
        self.assertGreater(diagnostics.unnecessary_tool_calls, 0)
        self.assertEqual(diagnostics.memory_usage_accuracy, 0.0)

    def test_preference_pairs_include_memory_aware_personalized_pair(self) -> None:
        task = next(
            task
            for task in generate_synthetic_tasks(7, np.random.default_rng(23))
            if task.task_type == "personalized_answer"
        )

        preferences = generate_preference_pairs([task], np.random.default_rng(29), pairs_per_task=2)

        self.assertTrue(
            any(
                preferred.actions == ("memory_lookup", "answer_directly")
                and rejected.actions == ("answer_directly",)
                for preferred, rejected in zip(
                    preferences.preferred,
                    preferences.rejected,
                    strict=True,
                )
            )
        )

    def test_small_pipeline_runs_without_ollama(self) -> None:
        rng = np.random.default_rng(11)
        tasks = generate_synthetic_tasks(21, rng)
        max_steps = 4

        policy, sft_history = train_agent_sft_policy(
            tasks,
            rng,
            max_steps=max_steps,
            demos_per_task=3,
            noise_rate=0.35,
            epochs=12,
            lr=0.2,
            batch_size=24,
        )
        reward_model, reward_history, _ = train_agent_reward_model(
            tasks,
            rng,
            max_steps=max_steps,
            pairs_per_task=5,
            epochs=18,
            lr=0.15,
            batch_size=32,
        )
        reference = policy.clone()
        baseline = evaluate_agent_policy(
            policy,
            reward_model,
            tasks,
            np.random.default_rng(99),
            max_steps=max_steps,
            samples_per_task=2,
            greedy=False,
        )
        rl_history = run_agent_policy_gradient(
            policy,
            reward_model,
            tasks,
            rng,
            max_steps=max_steps,
            episodes=12,
            batch_size=16,
            lr=0.06,
            kl_coef=0.08,
            reference_policy=reference,
            baseline_momentum=0.9,
        )
        final = evaluate_agent_policy(
            policy,
            reward_model,
            tasks,
            np.random.default_rng(99),
            max_steps=max_steps,
            samples_per_task=2,
            greedy=False,
        )

        self.assertTrue(sft_history)
        self.assertTrue(reward_history)
        self.assertTrue(rl_history)
        self.assertTrue(np.isfinite(baseline.true_reward))
        self.assertTrue(np.isfinite(final.true_reward))
        self.assertGreaterEqual(final.tool_accuracy, 0.0)
        self.assertLessEqual(final.tool_accuracy, 1.0)

    def test_ollama_client_falls_back_when_unavailable(self) -> None:
        rng = np.random.default_rng(19)
        task = generate_synthetic_tasks(1, rng)[0]
        trajectory = run_actions(task, ideal_action_sequence(task))
        client = OllamaClient(base_url="http://127.0.0.1:9", timeout=0.01)

        answer = client.generate_final_answer(task, trajectory)

        self.assertFalse(answer.used_ollama)
        self.assertIsNotNone(answer.warning)
        self.assertTrue(answer.text)

    def test_cli_runs_without_ollama(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli_main(
                [
                    "--num-tasks",
                    "7",
                    "--sft-epochs",
                    "2",
                    "--reward-epochs",
                    "2",
                    "--rl-episodes",
                    "2",
                    "--eval-samples",
                    "1",
                    "--example-count",
                    "1",
                ]
            )

        text = output.getvalue()
        self.assertIn("AgentRLHF", text)
        self.assertIn("avg tool cost", text)
        self.assertIn("style match", text)

    def test_benchmark_action_parser_accepts_json_and_text(self) -> None:
        json_actions, json_ok = parse_action_response(
            '{"actions": ["memory_lookup", "answer_directly"]}',
            max_steps=4,
        )
        text_actions, text_ok = parse_action_response(
            "I would use search_tool then verify_answer and answer_directly.",
            max_steps=4,
        )
        fallback_actions, fallback_ok = parse_action_response("no valid action", max_steps=4)

        self.assertEqual(json_actions, ("memory_lookup", "answer_directly"))
        self.assertTrue(json_ok)
        self.assertEqual(text_actions, ("search_tool", "verify_answer", "answer_directly"))
        self.assertTrue(text_ok)
        self.assertEqual(fallback_actions, ("answer_directly",))
        self.assertFalse(fallback_ok)

    def test_benchmark_cli_fails_gracefully_without_ollama(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            cli_main(
                [
                    "benchmark-models",
                    "--models",
                    "missing-local-model",
                    "--num-tasks",
                    "2",
                    "--ollama-base-url",
                    "http://127.0.0.1:9",
                    "--ollama-timeout",
                    "0.01",
                ]
            )

        text = output.getvalue()
        self.assertIn("Open-Source Local LLM Agent Benchmark", text)
        self.assertIn("No models were benchmarked", text)


if __name__ == "__main__":
    unittest.main()
