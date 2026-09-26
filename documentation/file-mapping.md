# Added Python Files: Purpose and Data Flow

This document maps the Python files added during the dataset, teacher-annotation,
splitting, and reward-calibration work. It intentionally excludes the Python
files supplied with the take-home assignment.

## Scope

The following files were supplied originally and are **not** described as our
new implementation:

- `scripts/build_golden_dataset.py`
- `scripts/regenerate_with_coverage.py`
- `scripts/run_eval.py`
- `scripts/topup_and_rescore.py`
- `tools/assess_wg_risk_context.py`
- `tools/calculate_quotes_wg.py`
- `tools/reward.py`

Some supplied files, especially `scripts/run_eval.py`, may have been adjusted
to integrate with the new shared modules. They remain part of the original
codebase rather than being counted as newly created files.

## End-to-end flow

```text
scripts/generate_scenarios.py
        │ uses src/scenarios/generate.py
        ▼
synthetic risk scenarios
        │
scripts/collect_quotes.py
        │ uses src/engine/client.py
        │ wraps the supplied risk and quote tools
        ▼
quoted scenarios + hash-addressed cache
        ├──────────────► scripts/summarize_quoted_dataset.py
        │                 └─ distribution/funnel reports
        │
        ▼
scripts/annotate_teacher_batch.py
        │ uses src/prompts.py
        ▼
validated GPT-6 Sol silver annotations
        │
scripts/create_training_splits.py
        ├─ reward-calibration split
        ├─ SFT train/validation splits
        └─ GRPO train/validation splits
                │
                ├────────► reward_calibration/calibrate.py
                │            └─ frozen reward weights
                │
                └────────► reward_calibration/validate_frozen.py
                             └─ negative-control, subgroup, and golden checks
```

The supplied golden dataset is used only where explicitly intended:

- Scenario generation reads golden **risk inputs only** to prevent exact
  duplicates; it does not read their Fable labels.
- Reward calibration uses only the dedicated 200-example silver-label split.
- Frozen-weight validation evaluates the already selected weights on held-out
  data and the supplied golden set without changing them.

## Runnable scripts

### `scripts/generate_scenarios.py`

Entry point for producing reproducible synthetic building-risk profiles.

- Calls `src.scenarios.generate.generate_scenarios` for the actual generation.
- Generates deterministic `scenario_id` values from canonical risk hashes.
- Balances hazard, building-age, building-type, and construction attributes.
- Validates every record and excludes exact risk matches from the supplied
  golden dataset.
- Makes no network or live quote-engine calls.
- Refuses to write into the protected `data/` and `tools/` directories.

Default outputs:

- `artifacts/scenarios/scenarios.jsonl`
- `artifacts/scenarios/generate_config.json`
- `artifacts/scenarios/summary.json`

The `--validate-only` mode regenerates and checks scenarios in memory without
writing files.

### `scripts/collect_quotes.py`

Turns generated risks into complete recommendation episodes by collecting the
risk context, available quotes, and deterministic quote scores.

- Uses `src.engine.client.QuoteEngineClient` for all live-engine access.
- Defaults to a dry run; live access requires `--execute-live`.
- Enforces the engine's minimum three-second delay.
- Maintains one cache record per scenario ID, including unsuccessful outcomes,
  so interrupted runs can resume without repeating completed calls.
- Uses a collector lock to prevent concurrent processes from bypassing the
  shared cache and rate limit.
- Writes only successful scenarios with at least three quotes to the public
  quoted-dataset output.
- Supports an explicit `--retry-transient-errors` option; other cached outcomes
  remain terminal.

Default outputs:

- `artifacts/quoted/quoted_scenarios.jsonl`
- `artifacts/quoted/collect_config.json`
- `artifacts/quoted/summary.json`
- Per-scenario cache records under `artifacts/cache/quotes/`

### `scripts/summarize_quoted_dataset.py`

Produces report-ready evidence about the dataset that survived live quote
collection.

- Summarizes scenario classes, hazard and age strata, quote counts, prices,
  coverage scores, deductibles, and insured amounts.
- Reads cache records to report the complete collection funnel, including
  no-quote, insufficient-quote, and error outcomes.
- Calculates subgroup collection success rates and useful numeric statistics.
- Makes no API calls and does not alter the quoted dataset.

Default outputs:

- `artifacts/quoted/distribution_summary.json`
- `artifacts/quoted/distribution_report.md`

### `scripts/annotate_teacher_batch.py`

Creates and manages GPT-6 Sol Batch API jobs that label quoted scenarios with
an expert-style top-three ranking and rationales. These labels are the silver
standard used for SFT and offline reward calibration.

It exposes five subcommands:

1. `prepare` validates the input and creates offline JSONL request shards.
2. `submit` uploads and submits one selected shard after explicit confirmation.
3. `status` refreshes and persists the selected batch's provider status.
4. `download` retrieves one completed shard and validates every annotation.
5. `merge` joins all validated shards into one annotation file.

Important behavior:

- Uses the canonical broker prompt from `src/prompts.py`, keeping teacher and
  evaluation instructions aligned.
- Uses structured output constraints and validates ranks, tariff IDs, and
  schema before accepting a response.
- Stores batch IDs and state in the work directory so execution can resume on a
  later day or machine with the same artifacts.
- Keeps source quoted data immutable and records hashes for traceability.
- Only `submit`, `status`, and `download` contact OpenAI.

Default work area: `artifacts/teacher/`. The final merged output is
`artifacts/teacher/annotations.jsonl`.

### `scripts/create_training_splits.py`

Joins quoted scenarios with teacher annotations and creates deterministic,
disjoint experiment splits.

- First stratifies all 1,490 examples into reward-calibration, SFT, and GRPO
  pools.
- Then independently re-stratifies the SFT and GRPO pools into train and
  validation partitions.
- Balances hazard tier, age stratum, building type, construction class,
  natural-hazard request, and owner occupancy.
- Verifies one-to-one scenario/annotation matching and rejects overlap with the
  supplied golden risks.
- Removes teacher annotations from GRPO training records; validation retains
  them strictly for offline measurement.
- Writes distribution evidence, source hashes, sizes, and split invariants.

Current default split sizes:

| Split | Examples | Teacher labels included |
|---|---:|---|
| Reward calibration | 200 | Yes |
| SFT training | 300 | Yes |
| SFT validation | 90 | Yes |
| GRPO training | 810 | No |
| GRPO validation | 90 | Yes, evaluation only |

Default output directory: `artifacts/splits/`.

## Supporting modules

### `src/scenarios/generate.py`

Contains the reusable scenario-generation and validation logic used by the
scenario CLI.

Responsibilities include:

- Defining the canonical risk schema and accepted enum values.
- Normalizing risks and deriving SHA-256 scenario IDs.
- Validating construction year, dwelling units, rebuild sum, and cross-field
  constraints.
- Generating reproducible, diverse building profiles from a seed.
- Using known-valid German postal codes and classifying their hazard strata
  through the supplied risk-context tool.
- Reading only golden risk fields to construct a deduplication set.

### `src/engine/client.py`

Provides the safe integration boundary around the two supplied engine tools and
the supplied reward scorer.

For each scenario it:

1. Normalizes and hashes the risk.
2. Reuses the matching on-disk cache record when available.
3. Assesses the risk context.
4. Chooses the requested coverage policy from that context.
5. Calls the live quote engine subject to the delay floor.
6. Scores returned quotes and stores an atomic cache record.
7. Classifies the result as success, no quotes, insufficient quotes, transient
   error, or permanent error.

It also sanitizes error messages so credentials and URL query parameters cannot
be persisted accidentally.

### `src/prompts.py`

Single source of truth for the broker system instruction and user prompt.

- Selects the quote fields exposed to the model.
- Formats the building, assessed risk context, and available tariffs.
- Supports the benchmark's ranking-only response and the teacher's ranking plus
  rationale response.
- Prevents the teacher and evaluator from silently drifting to different task
  definitions.

### Package marker files

The following intentionally small files make their directories importable
Python packages:

- `src/__init__.py`
- `src/engine/__init__.py`
- `src/scenarios/__init__.py`
- `reward_calibration/__init__.py`

They contain no pipeline logic. `reward_calibration/__init__.py` additionally
provides a short package description.

## Reward-calibration package

### `reward_calibration/calibrate.py`

Performs offline random search for reward weights using the 200-example
reward-calibration split and the teacher's rankings.

- Samples candidate combinations for coverage, insured amount, deductible,
  price, and supplied coverage-score weights.
- Scores the already collected quotes locally; it makes no engine or LLM calls.
- Uses deterministic stratified folds and a nested held-out estimate.
- Selects primarily by top-one agreement, then top-three recall and pairwise
  ordering, with proximity to the supplied default reward as a final tie-break.
- Cross-checks its optimized scorer against `tools/reward.py` before searching.
- Freezes the selected configuration so downstream validation cannot retune it.

Default outputs under `artifacts/reward_calibration/`:

- `best_weights.json`
- `calibration_summary.json`
- `calibration_report.md`
- `candidate_results.jsonl`

### `reward_calibration/validate_frozen.py`

Audits the frozen weights without selecting or changing them.

It performs three separate checks:

1. A permuted-label negative control to show that the search does not produce
   the same apparent accuracy when teacher rankings are randomized.
2. Subgroup evaluation on the held-out GRPO validation split.
3. Locked evaluation against the original Fable-annotated golden dataset.

It compares the calibrated reward with the original default reward, writes
per-scenario golden results, and explicitly records that the weights were not
reselected. All work is local and deterministic; no LLM calls are made.

Default outputs under `artifacts/reward_calibration/validation/`:

- `validation_summary.json`
- `validation_report.md`
- `golden_per_scenario.jsonl`

## Training and experiment package

### `training/common.py`

Dependency-light shared helpers for training data, local evaluation, hashing,
atomic JSON/JSONL output, teacher-label extraction, strict ranking targets, and
lenient/strict model-output parsing.

### `training/prepare_sft_data.py`

Transforms the stratified 300/90 SFT splits into TRL conversational
prompt-completion data. It validates that every teacher selection exists in the
corresponding quote panel, removes rationales from the completion target, checks
split disjointness, and writes hashes plus schema statistics. It never reads the
golden dataset.

### `training/train_sft.py`

Runs Qwen3.5-2B text-only SFT with Unsloth and LoRA. It performs a tokenizer
length audit before allocating the full run, selects the checkpoint by held-out
validation loss, saves the portable adapter, and records resolved model/package
versions, hyperparameters, GPU details, elapsed time, peak VRAM, Trainer history,
and output paths in `run_record.json`.

### `training/grpo/prepare_data.py`

Transforms the 810/90 GRPO split into self-contained TRL conversational
datasets. It verifies that training has no teacher labels, uses the canonical
prompt, applies `tools/reward.py` with the frozen `best_weights.json`, and saves
only per-tariff scores needed by rollout rewards. Validation teacher IDs are
retained solely for offline monitoring.

### `training/grpo/rewards.py`

Defines four bounded rollout signals: exact JSON schema, valid distinct offered
tariffs, frozen calibrated utility of rank 1, and discounted calibrated utility
for the complete Top-3. It has no model or GPU dependency and can be checked
offline before paid training.

### `training/grpo/train.py`

Loads the completed SFT LoRA adapter and runs GRPO through TRL/Unsloth. It
enforces data/manifest checksums, forbids teacher labels in training, audits
token lengths, validates group-size divisibility, records reward components and
environment provenance, supports one-step and ten-step smoke gates, and saves a
portable final adapter plus intermediate checkpoints.

### `training/run_eval.py`

One deterministic local-checkpoint evaluator for the unchanged base model, SFT
adapter, and future GRPO adapter. It uses the canonical benchmark prompt and the
supplied per-quote reward unchanged. Alongside Top-1, Top-3 recall, and reward
ratio, it records strict JSON, valid-tariff, token, latency, throughput, and
optional GPU-cost metrics. Golden execution requires an explicit confirmation
flag and must not be used for checkpoint selection.

### `training/plot_run.py`

Reads a completed `run_record.json` and renders train loss, validation loss, and
learning-rate curves with Matplotlib. It does not contact an external tracking
service.

### `training/create_runpod_bundle.py`

Builds an explicit, credential-free archive for the GPU machine. It includes
the code, prepared SFT data, validation and GRPO splits, frozen reward, golden
evaluation file, and reproducibility documentation while excluding `.env`,
caches, raw provider responses, pilots, old data, and prior checkpoints. It
writes an embedded file manifest plus an external SHA-256 checksum.

### `training/__init__.py`

Marks `training` as an importable package. It contains no pipeline logic.

## Which files to modify for future stages

- For scenario diversity or validation rules, change
  `src/scenarios/generate.py`; keep `scripts/generate_scenarios.py` as the thin
  command-line entry point.
- For caching, retry, throttling, or live-engine behavior, change
  `src/engine/client.py`; do not duplicate engine calls in another script.
- For broker instructions or output format, change `src/prompts.py` so teacher
  generation and evaluation remain aligned.
- For split sizes or stratification, change
  `scripts/create_training_splits.py` and regenerate the manifest.
- The reward weights in `best_weights.json` are frozen. Training code should
  load them, not rerun calibration or tune them on GRPO validation/golden data.
- For SFT/GRPO preparation, training, checkpoint evaluation, or plots, work under
  `training/`; generated adapters and run records stay under
  `artifacts/training/`.
