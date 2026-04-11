RLHF Pipeline (Synthetic Data)
===============================

This project demonstrates an end-to-end reinforcement learning from human feedback (RLHF) workflow using fully synthetic data. The pipeline covers:

1. Supervised fine-tuning (SFT) of a policy on noisy demonstrations.  
2. Reward-model training from preference pairs generated with a hidden ground-truth environment.  
3. Reinforcement learning with a KL penalty (policy gradient) that optimizes the policy against the learned reward model.

Everything runs on lightweight NumPy models so you can study the workflow without downloading large neural networks.

Quickstart
----------

```bash
# 1. Create and sync the uv environment.
uv sync

# 2. Run the pipeline end to end with the default configuration.
uv run rlhf-pipeline
```

The CLI prints summaries for each stage (SFT, reward modeling, RL) plus baseline and final reward estimates. You can adjust any hyper-parameter by passing CLI flags; run `uv run rlhf-pipeline --help` for the full list.

Architecture Highlights
-----------------------

- **Synthetic environment** — A hidden linear mapping defines the ideal response features for each prompt, while the ground-truth reward is a clipped linear score over shared prompt/response features.  
- **Policy** — A Gaussian policy maps prompt features to response features. It starts from SFT (mean-squared error) and is fine-tuned with policy gradient updates.  
- **Reward model** — Logistic regression learns to order preference pairs using the same handcrafted joint features as the environment, with L2 regularisation to keep scores well behaved.  
- **RL stage** — Uses a moving reward baseline and a KL penalty to keep the updated policy close to the SFT reference.

Tuning Knobs
------------

- `--prompt-dim`, `--response-dim`: feature dimensionality.  
- `--samples-per-prompt`, `--pairs-per-prompt`: dataset sizes for SFT and reward modeling.  
- `--sft-*`: SFT training schedule.  
- `--reward-*`: reward-model optimizer settings.  
- `--rl-*`: RL episodes, learning rates, and KL regularization strength.

Try increasing `--rl-episodes` or decreasing `--kl-coef` to watch how aggressive policy updates affect true reward relative to the baseline.
