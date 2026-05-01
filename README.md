# AgentRLHF: Synthetic RLHF for Tool-Using AI Agents

AgentRLHF is a CPU-friendly synthetic RLHF pipeline for studying how a tool-using AI agent learns multi-step behavior from demonstrations, trajectory preferences, and reinforcement learning with a KL penalty.

Many agents today have access to tools and memory, but they still struggle with deciding when and how to use them. AgentRLHF targets that gap by training the agent's decision process over full trajectories, including tool use, memory use, clarification, verification, safety handling, and cost awareness.

## Open-Source / Local-Only Statement

This project does not use OpenAI, Anthropic, Gemini, Cohere, Perplexity, or any other closed-source hosted LLM API. The default mode runs entirely with NumPy, deterministic synthetic tools, synthetic memory, and template outputs.

Ollama is optional and is used only to verbalize example final answers after the trained agent has already selected a trajectory. Tests and the default CLI do not require Ollama, internet access, API keys, GPUs, or paid services.

## Quickstart

```bash
uv sync
uv run rlhf-pipeline
```

The same pipeline is also available through:

```bash
uv run agent-rlhf
```

## What Makes This Agentic RLHF

The policy chooses discrete actions over multiple steps:

- `answer_directly`
- `ask_clarification`
- `search_tool`
- `calculator_tool`
- `memory_lookup`
- `verify_answer`
- `refuse`

Synthetic task types include:

- `factual_static`
- `current_info_required`
- `math_calculation`
- `personalized_answer`
- `ambiguous_request`
- `unsafe_request`
- `comparison_task`

## Synthetic Memory

AgentRLHF now includes deterministic synthetic user memory. Personalized tasks include entries such as:

- User prefers concise answers.
- User is beginner-level.
- User prefers Python examples.
- User wants cost-focused comparisons.
- User prefers step-by-step explanations.
- User prefers safety-first answers.

The `memory_lookup` action calls the registered local `memory_lookup` tool. The reward function gives credit when the agent uses relevant memory for personalized tasks and matches the expected answer style. It penalizes ignoring required memory and using memory lookup when the task has no relevant memory.

## MCP-Style Local Tool Registry

AgentRLHF includes an MCP-inspired local Python tool registry. This is not a full MCP server and does not expose external network tools. It is a lightweight registry that keeps tool metadata beside deterministic local tool execution.

Each registered tool includes:

- `name`
- `action`
- `description`
- `input_schema`
- `output_schema`
- `cost`
- `latency`
- `risk`
- `recommended_task_types`

Registered tools:

- `search_tool`
- `calculator_tool`
- `memory_lookup`
- `verify_answer`

The registry powers tool execution, reward scoring, cost and latency accounting, and CLI reporting.

## Architecture

```text
Synthetic User Task
      |
      v
Memory Context + Tool Registry
      |
      v
Agent Policy
      |
      v
Registered Tool Calls
      |
      v
Trajectory Log
      |
      v
Preference Pair Generator
      |
      v
Reward Model
      |
      v
RL with KL Penalty
      |
      v
Improved Agent Policy
```

## Reward Design

The hidden reward combines:

- task success
- correct tool usage
- memory usage accuracy
- style match score
- safety handling accuracy
- clarification accuracy
- unnecessary tool penalties
- trajectory length penalty
- tool cost penalty
- tool latency penalty

## Trajectory-Level Preferences

Preference pairs compare two trajectories for the same task. Examples:

- personalized task: `memory_lookup -> answer_directly` preferred over `answer_directly`
- math task: `calculator_tool -> verify_answer -> answer_directly` preferred over `search_tool -> answer_directly`
- unsafe task: `refuse` preferred over `search_tool -> answer_directly`
- ambiguous task: `ask_clarification` preferred over `answer_directly`

A lightweight logistic reward model learns from handcrafted trajectory features. RL then optimizes the policy against that learned reward model while applying a KL penalty against the SFT reference policy.

## CLI Output

The CLI prints:

- SFT training summary
- reward-model training summary
- RL training summary
- SFT baseline metrics
- Agent RLHF metrics
- 2-3 example tasks with memory, trajectories, tool cost, latency, and reward improvement

Example metric shape:

```text
SFT baseline:
- true reward: 2.1843
- predicted reward: 1.9021
- tool accuracy: 0.713
- memory accuracy: 0.690
- style match: 0.640
- safety accuracy: 0.850
- avg trajectory length: 2.18
- avg tool cost: 0.121
- avg tool latency: 0.284
- unnecessary tool calls: 0.241
```

## Useful Flags

```bash
uv run rlhf-pipeline --help
uv run rlhf-pipeline --num-tasks 70 --rl-episodes 250
uv run rlhf-pipeline --sft-noise-rate 0.4 --kl-coef 0.1
```

## Local Ollama Model Benchmark

AgentRLHF can benchmark local open-source Ollama models on the same synthetic agentic task set. The benchmark asks each model to choose an action trajectory, executes any selected registered synthetic tools locally, and scores the resulting trajectory with the same hidden reward logic used by the pipeline.

```bash
uv run rlhf-pipeline benchmark-models --models llama3.2:1b gemma3:1b --num-tasks 50
```

Useful benchmark flags:

```bash
uv run rlhf-pipeline benchmark-models \
  --models llama3.2:1b gemma3:1b \
  --num-tasks 50 \
  --ollama-timeout 12
```

Benchmark mode uses only the local Ollama HTTP API. You can use any open-source model that is installed in Ollama; examples include `llama3.2:1b`, and `gemma3:1b`. If Ollama is not running or a model is not installed, the command prints a warning and skips that model. The normal pipeline and tests do not require Ollama.

GPU acceleration is optional for AgentRLHF itself. The default synthetic pipeline is lightweight and CPU-friendly. For higher-parameter open-source Ollama models, a GPU or Apple Metal acceleration can make inference much faster and may be needed for practical latency or memory headroom. If local benchmark requests time out, try fewer tasks, a longer `--ollama-timeout`, or a smaller model before assuming new hardware is required.

## Optional Ollama Usage

Start Ollama through the macOS app, or from a shell if needed:

```bash
ollama serve
```

Pull a local open-source model:

```bash
ollama pull llama3.2:1b
```

Example local Ollama model names include:

- `gemma4:26b`
- `llama3.2:1b`
- `gemma3:27b`

You can use any open-source model that can run locally through Ollama.

Larger parameter models may need more RAM/VRAM and benefit strongly from GPU acceleration. Smaller models are usually better for quick CPU-friendly benchmarks.

Run the demo with local Ollama final-answer wording:

```bash
uv run rlhf-pipeline --use-ollama --ollama-model llama3.2:1b
```

If Ollama is unavailable, not running, times out, or the requested model is not installed, the CLI prints a warning and falls back to synthetic/template answers.

## Development Checks

```bash
uv run python -m unittest
uv run rlhf-pipeline
```

No closed-source API keys, SDKs, or integrations are required.
