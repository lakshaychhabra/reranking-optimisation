# Training Plan: Qwen3.5-2B

Decision date: 2026-09-26

## Decision

Use `Qwen/Qwen3.5-2B` as the primary model. Train it with Unsloth using BF16
LoRA for both the supervised warm-up and GRPO stages.

Do not use 4-bit QLoRA initially. Unsloth's Qwen3.5 guidance reports larger
than normal quantization differences for this family and recommends BF16 LoRA.

## Why this model is feasible

- It is a post-trained 2B-parameter Qwen model intended for prototyping and
  task-specific fine-tuning.
- Its official BF16 checkpoint is approximately 4.55 GB on disk.
- Unsloth reports approximately 5 GB VRAM for Qwen3.5-2B BF16 LoRA under its
  benchmark configuration.
- Our task is text-only, short-output, and narrow: rank 4-9 supplied tariffs
  and emit three tariff IDs as strict JSON.
- The 2B model remains substantially smaller than the supplied Qwen3.5-9B
  open-source baseline.

Qwen3.5 is nevertheless a unified multimodal model with a vision encoder and a
hybrid Gated DeltaNet/full-attention text backbone. We will freeze visual
components and train only language-side LoRA adapters.

## Stack

- PyTorch with CUDA
- Current Unsloth and `unsloth_zoo`
- Transformers v5, required for Qwen3.5
- TRL `SFTTrainer` and `GRPOTrainer`
- PEFT LoRA adapters
- Hugging Face Datasets
- TensorBoard-compatible local logs plus repository JSON metrics
- Matplotlib for final static plots

Package versions will be locked only after the one-step SFT and GRPO smoke
tests pass. The exact successful environment will then be captured with
`pip freeze`, CUDA version, GPU model, and model revision.

## Tracking decision

Weights & Biases is not required for this take-home. Every run must persist:

- the Trainer `trainer_state.json` and local TensorBoard event file;
- a compact machine-readable metrics JSON file;
- model revision, dataset hashes, seed, hyperparameters, GPU, elapsed time,
  peak VRAM, and estimated cost;
- validation loss, JSON-validity rate, Top-1 agreement, Top-3 recall, and mean
  reward ratio where available; and
- a record in `documentation/experiment-runs.json` after the result is final.

Matplotlib will read those local metrics to produce report figures. W&B should
only be added if the experiment count grows enough to require remote comparison
or live monitoring.

## SFT configuration

Dataset:

- Train: `artifacts/splits/sft_train.jsonl` (300 examples)
- Validation: `artifacts/splits/sft_validation.jsonl` (90 examples)
- Prompt: canonical ranking-only prompt from `src/prompts.py`
- Target: strict `top_3` JSON containing rank and tariff ID only
- Teacher rationales remain in the source data but are not completion targets

Initial settings:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-2B` |
| Precision | BF16 |
| Method | LoRA |
| LoRA rank / alpha | 16 / 16 |
| LoRA dropout | 0 |
| Target modules | `q/k/v/o`, `gate/up/down` projections |
| Maximum sequence length | 2,048, subject to tokenizer audit |
| Epochs | 3 |
| Learning rate | `2e-4` initial run |
| Loss | Assistant completion only |
| Gradient checkpointing | Unsloth |
| Model selection | SFT validation, never golden data |

Thinking output must remain disabled. The desired completion is a short JSON
ranking, not a reasoning trace.

## GRPO configuration envelope

Dataset:

- Train: `artifacts/splits/grpo_train.jsonl` (810 prompts, no teacher labels)
- Validation: `artifacts/splits/grpo_validation.jsonl` (90 held-out labels)
- Reward: frozen weights from
  `artifacts/reward_calibration/best_weights.json`

Initial resource-limited settings:

| Setting | Initial value |
|---|---|
| Initialization | Best SFT LoRA adapter |
| Precision | BF16 LoRA |
| Generations per prompt | 4 |
| Maximum completion | 128 tokens |
| Prompt/sequence ceiling | 2,048 tokens |
| Fast vLLM inference | Disabled for Qwen3.5 compatibility |
| Generation backend | Unsloth inference |
| Training duration | Conservative 50-step cap; save every 10 and select using held-out validation |

GRPO separately logs exact JSON validity, distinct offered-tariff validity,
frozen calibrated Top-1 utility, and discounted calibrated Top-3 utility. Their
fixed composition weights are 0.10, 0.10, 0.40, and 0.40 respectively. These
outer shaping weights are fixed before training and are not tuned on GRPO
validation or golden data. The underlying business scores still come directly
from the frozen calibrated `RewardWeights` configuration.

## GPU recommendation

A single 24 GB NVIDIA GPU is the recommended target. An RTX 4090, L4, or A10G
has enough margin for BF16 LoRA, four short GRPO completions, compilation, and
evaluation without forcing 4-bit training.

| GPU tier | SFT estimate | GRPO estimate | Assessment |
|---|---:|---:|---|
| 16 GB T4 | 40-90 min | 8-15 h | Possible but slow and tight; avoid if prices are similar |
| 24 GB L4/A10G | 20-45 min | 4-8 h | Recommended cloud tier |
| 24 GB RTX 4090 | 10-25 min | 2-5 h | Best price/performance when available |
| 40/80 GB A100 | 8-20 min | 1.5-4 h | Comfortable but unnecessary for 2B |

These are planning ranges, not benchmark claims. They assume approximately
810 prompt groups, four short rollouts per prompt, a 2,048-token ceiling, and
one primary GRPO pass. Kernel compilation, provider CPU speed, output length,
and checkpoint/evaluation frequency can materially change runtime.

The first paid machine must run a 10-step benchmark. Its measured seconds per
step and peak allocated VRAM will replace these estimates before the bounded
50-step run. Going beyond 100 steps requires explicit evidence from held-out
validation and an explicit CLI acknowledgement.

## Compatibility gates

Qwen3.5 support has evolved quickly and earlier Unsloth/TRL releases had GRPO
issues around partial rotary embeddings, multimodal position IDs, and text-only
re-forward passes. Current Unsloth documentation declares Qwen3.5 SFT and RL
support, but our run proceeds only if all of these gates pass:

1. Load the official checkpoint and tokenizer at the recorded revision.
2. Run one SFT forward/backward/update step.
3. Save and reload the LoRA adapter; confirm identical deterministic output.
4. Run one GRPO step with four completions and the real reward function.
5. Run ten GRPO steps while logging reward components and checking for NaNs.
6. Freeze the exact working dependency versions before renting the full run.

For Qwen3.5 GRPO, use the documented Unsloth path with fast vLLM inference
disabled. If the current package combination fails these gates, first try plain
TRL with `use_vllm=False`. If that also fails, the declared fallback is
`Qwen/Qwen3-1.7B`; do not patch third-party training kernels during a one-week
take-home exercise.
