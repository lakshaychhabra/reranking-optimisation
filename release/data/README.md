---
pretty_name: WG Recommendation Training and Development Data
language:
- de
license: other
task_categories:
- text-generation
tags:
- insurance
- recommendation
- sft
- grpo
- qwen
size_categories:
- 1K<n<10K
configs:
- config_name: sft
  data_files:
  - split: train
    path: sft/train.jsonl
  - split: validation
    path: sft/validation.jsonl
- config_name: grpo
  data_files:
  - split: train
    path: grpo/train.jsonl
  - split: validation
    path: grpo/validation.jsonl
- config_name: eval_sft
  data_files:
  - split: validation
    path: eval/sft_validation.jsonl
- config_name: eval_grpo
  data_files:
  - split: validation
    path: eval/grpo_validation.jsonl
---

# WG Recommendation Training and Development Data

This dataset supports reproducible fine-tuning of a small language model to rank
German Wohngebäude (residential-building insurance) tariffs. Given a building
risk, risk context, and a shuffled set of priced quotes, the model returns a
ranked top three in strict JSON.

It contains the prepared data used for supervised fine-tuning (SFT), the optional
GRPO continuation, and development-set evaluation. It does **not** contain the
held-out 50-scenario golden benchmark, golden labels, or per-scenario golden
predictions.

## Configurations and splits

| Configuration | Split | Rows | Purpose |
|---|---|---:|---|
| `sft` | `train` | 300 | Conversational prompt/completion examples for SFT |
| `sft` | `validation` | 90 | SFT checkpoint selection and development monitoring |
| `grpo` | `train` | 810 | Prompts with frozen per-tariff business rewards |
| `grpo` | `validation` | 90 | GRPO development monitoring; teacher ranking retained offline |
| `eval_sft` | `validation` | 90 | Raw development scenarios corresponding to SFT validation |
| `eval_grpo` | `validation` | 90 | Raw development scenarios corresponding to GRPO validation |

Load a configuration with `datasets`:

```python
from datasets import load_dataset

sft = load_dataset("lakshaychhabra/wg-recommendation-data", "sft")
grpo = load_dataset("lakshaychhabra/wg-recommendation-data", "grpo")
```

## Record formats

### SFT

- `scenario_id`: deterministic identifier for the normalized risk.
- `prompt`: TRL conversational messages created by the canonical prompt builder.
- `completion`: assistant message containing the ranked top-three target.

Teacher rationales are not included in the target.

### GRPO

- `scenario_id`: deterministic scenario identifier.
- `prompt`: TRL conversational messages using the same canonical prompt format.
- `reward_by_tariff_id`: frozen business reward for each valid tariff.
- `teacher_top3`: validation-only field for offline monitoring; absent from
  training and never passed to the GRPO reward function.

### Development evaluation

The `eval_*` configurations retain the structured risk, requested coverage,
quotes, risk context, and development teacher annotation needed to run the
repository evaluator. These are development records, not the golden benchmark.

## Data construction and leakage controls

- Scenarios were generated independently from the held-out benchmark, priced
  through the cached quote-engine client, and split deterministically by
  `scenario_id`.
- Training and validation are disjoint within each training stage.
- SFT targets contain three distinct tariff IDs that exist in their quote sets.
- GRPO training contains no teacher labels; rewards were precomputed from frozen
  reward weights.
- The manifests record row counts, prompt statistics, source hashes, reward
  configuration, and validation checks.
- The golden benchmark was used only for final aggregate evaluation and is not
  distributed here.

## Reproducibility

The companion project contains the canonical prompt builder, preparation
scripts, training entry points, exact resolved configurations, adapter weights,
and evaluation commands. Use the repository's `HOW_TO_RUN.md` as the primary
runbook. The selected result is a Qwen3.5-2B SFT LoRA with 0.84 top-1 agreement
on the held-out benchmark; the GRPO continuation is included as an optional
ablation and was not selected.

## Scope and license

This dataset is published for reproducing the associated take-home experiment.
It is not a general insurance-advice dataset and must not be used as a substitute
for qualified brokerage, underwriting, or legal review. The repository does not
grant additional rights over third-party tariff or quote-engine data; downstream
users are responsible for confirming that their use is permitted.
