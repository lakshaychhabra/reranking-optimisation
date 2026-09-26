# Reproducing the WG Training-Data Pipeline

This document records the exact data-generation and quote-collection sequence
used for the take-home. Commands are intended to be run from the repository
root. It distinguishes completed work from planned later stages so the final
report does not accidentally claim an experiment that was never run.

## Safety and data boundaries

- `data/` is the supplied held-out benchmark. Never write to it and never use
  its Fable annotations for training, reward calibration, prompt tuning, or
  checkpoint selection.
- `tools/` contains the supplied digital twin. It is imported but not modified.
- Candidate-generated data is stored under `artifacts/`.
- Credentials are passed through environment variables only. Never put them in
  a command, config, log, or committed file.
- The live quote engine is called only through `src/engine/client.py`, which
  performs direct SHA-256 cache lookup and enforces at least three seconds
  between uncached calls.

## Runtime requirements

- Python 3.11 or newer.
- No third-party dependency is required for scenario generation, quote
  collection, validation, or distribution summaries.
- For live pricing, set credentials in the shell environment:

```bash
export MM_ID="<provided broker id>"
export MM_PA="AFO"
```

Do not source or inspect a repository `.env` file.

## 1. Generate the final candidate pool

Command used:

```bash
python3 scripts/generate_scenarios.py \
  --n 2400 \
  --seed 20260925 \
  --output artifacts/scenarios/final_candidates.jsonl \
  --config-output artifacts/scenarios/final_generate_config.json \
  --summary-output artifacts/scenarios/final_generate_summary.json
```

This stage is local and makes no network calls. It:

- generates 2,400 deterministic synthetic building risks;
- samples only from the supplied list of known-valid German postcodes;
- stratifies hazard, age, building type, construction class, occupancy, and
  deductible fields;
- hashes canonical risk JSON to form `scenario_id`;
- excludes exact risk matches with the 50 held-out golden scenarios; and
- writes the resolved configuration and distribution summary.

Observed generation result:

- 2,400 scenarios and 2,400 unique scenario IDs;
- hazard: 937 moderate, 741 elevated, 722 high;
- age: 815 old, 785 middle, 800 modern;
- building types: 498 single-family, 456 two-family, 475 multi-family,
  487 semi-detached, and 484 terraced;
- construction classes: 484 BAK1, 487 BAK2, 477 FHG1, 476 FHG2, and 476 FHG3.

Generated artifacts:

```text
artifacts/scenarios/final_candidates.jsonl
artifacts/scenarios/final_generate_config.json
artifacts/scenarios/final_generate_summary.json
```

## 2. Optional preflight validation

Validate deterministic generation and schema without writing or calling the
engine:

```bash
python3 scripts/generate_scenarios.py \
  --n 2400 \
  --seed 20260925 \
  --golden-path data/golden_wg_recommendations.jsonl \
  --validate-only
```

Preview the collection plan without live calls:

```bash
python3 scripts/collect_quotes.py \
  --input artifacts/scenarios/final_candidates.jsonl \
  --output artifacts/quoted/final_quoted_scenarios.jsonl \
  --cache-dir artifacts/cache/final_quotes \
  --target 1600 \
  --max-live-calls 2000 \
  --seed 20260925 \
  --dry-run
```

## 3. Collect live quotes

Command launched for the final collection:

```bash
python3 scripts/collect_quotes.py \
  --input artifacts/scenarios/final_candidates.jsonl \
  --output artifacts/quoted/final_quoted_scenarios.jsonl \
  --cache-dir artifacts/cache/final_quotes \
  --config-output artifacts/quoted/final_collect_config.json \
  --summary-output artifacts/quoted/final_collect_summary.json \
  --target 1600 \
  --max-live-calls 2000 \
  --seed 20260925 \
  --execute-live
```

Interpretation of the limits:

- `--target 1600` requests 1,600 successful scenarios with at least three
  priceable tariffs.
- `--max-live-calls 2000` allows at most 2,000 previously uncached attempts in
  this invocation.
- Collection stops early if it reaches 1,600 successes.
- Every attempt is written immediately to an atomic cache file.
- The success-only combined JSONL and final summary are written when the
  invocation exits normally.

Cache layout:

```text
artifacts/cache/final_quotes/<sha256-scenario-id>.json
```

The scenario ID is the SHA-256 hash of canonical normalized risk JSON. Lookup
is by exact filename; the collector does not scan the cache to find a risk.
Cached `success`, `no_quotes`, `insufficient_quotes`, and error results are all
reused. A transient error is retried only when explicitly requested with
`--retry-transient-errors`.

### Resuming after interruption

Rerun the exact same live command. Completed cache records are reused, failed
or insufficient scenarios are not repeated, and only uncached scenarios call
the engine. If the combined JSONL was not written before interruption, cached
successes reconstruct it during the resumed run.

### Non-invasive progress check

While collection is running, count completed attempts without changing state:

```bash
find artifacts/cache/final_quotes -maxdepth 1 -name '*.json' -type f | wc -l
```

Do not start a second live collector against the same cache. A lock prevents
concurrent execution because independent processes could otherwise violate the
rate limit or duplicate calls.

## 4. Validate the completed quoted dataset

After collection exits and writes the combined JSONL:

```bash
python3 scripts/collect_quotes.py \
  --output artifacts/quoted/final_quoted_scenarios.jsonl \
  --validate-only
```

This verifies:

- success-only rows;
- unique scenario and tariff IDs;
- at least three quotes per scenario;
- positive prices;
- reward and reward-component presence;
- descending reward order;
- valid reward recommendation IDs; and
- absence of credential values or authenticated URLs.

Check the final row count:

```bash
wc -l artifacts/quoted/final_quoted_scenarios.jsonl
```

The intended result is 1,600 rows. A lower count is valid if the 2,000-attempt
limit was exhausted first; resume with an explicitly approved additional call
budget if necessary.

## 5. Produce final distribution evidence

```bash
python3 scripts/summarize_quoted_dataset.py \
  --input artifacts/quoted/final_quoted_scenarios.jsonl \
  --cache-dir artifacts/cache/final_quotes \
  --output artifacts/quoted/final_distribution_summary.json \
  --markdown-output artifacts/quoted/final_distribution_report.md \
  --seed 20260925
```

The JSON is machine-readable experiment evidence. The Markdown is formatted for
reuse in the final report. Together they cover:

- the full attempt/success/failure funnel;
- success rates by hazard, age, building type, and construction class;
- final successful-scenario balance;
- quote, insurer, product, price, quality, and reward distributions; and
- within-scenario variation of each supplied reward feature.

## 6. Freeze the dataset

Before teacher annotation, preserve file hashes:

```bash
shasum -a 256 \
  artifacts/scenarios/final_candidates.jsonl \
  artifacts/quoted/final_quoted_scenarios.jsonl \
  artifacts/quoted/final_distribution_summary.json \
  > artifacts/quoted/final_dataset_sha256.txt
```

Do not modify or reorder the frozen quoted dataset afterward. Create train,
reward-calibration, and development splits as new files.

## 7. GPT-6 Sol teacher annotation through Batch API

`scripts/annotate_teacher_batch.py` implements the complete teacher workflow.
It uses `gpt-6-sol` with low reasoning and strict structured output. Its broker
system prompt and scenario formatting come from the same canonical builder used
by `scripts/run_eval.py`; the teacher variant adds only the rationale fields
needed by the annotation schema. The prompt contains only risk, risk context,
and sanitized quote facts.
It cannot contain `reward`, `reward_components`, or
`reward_recommended_tariff_id` because the request builder selects an explicit
field allow-list.

Downloaded labels are checked against the actual tariff IDs. Insurer and product
are copied from the source quote rather than trusted from generated text. Final
records match the existing `data/annotations.jsonl` schema, and the frozen
quoted dataset is never modified.

The 2,000-call collection budget produced 1,490 validated successful records.
After validating and freezing that completed dataset, prepare the offline
request files:

```bash
python3 scripts/annotate_teacher_batch.py prepare
```

This creates two equal shards by default (745 and 745 requests) and prints a
token/cost estimate. Sharding keeps each submission below the likely Tier-1
queued-token limit. Submit only one shard at a time:

```bash
python3 scripts/annotate_teacher_batch.py submit --shard 0 --confirm-submit
python3 scripts/annotate_teacher_batch.py status --shard 0
python3 scripts/annotate_teacher_batch.py download --shard 0
```

Wait until shard 0 has completed and downloaded, then repeat those three
commands for shard 1. Batch completion may take up to 24 hours. Finally:

```bash
python3 scripts/annotate_teacher_batch.py merge
```

The final silver labels are written to
`artifacts/teacher/annotations.jsonl`. Raw provider responses, validation
errors, actual token counts, exact cost, file hashes, batch IDs, and the merged
summary remain under `artifacts/teacher/`.

The final offline preparation step records the estimate calculated from all
1,490 actual prompts in `artifacts/teacher/batch_plan.json`. Budget $6-$8 for
normal output-length variation and targeted retries.

For the completed dataset, preparation produced two 745-request files of about
3.0 MB each. The recorded estimate is 1,778,126 input tokens plus 670,500
output tokens, or $5.1306 at GPT-6 Sol Batch prices. This is an estimate; the
download and merge stages record actual provider-reported usage and cost.

### Observed teacher cost

Shard 0 completed successfully with all 745 annotations valid and no validation
errors. Provider-reported usage was 774,013 input tokens and 209,640 output
tokens. At Batch prices this cost exactly $1.822213:

```text
input:   774,013 / 1,000,000 * $1 = $0.774013
output:  209,640 / 1,000,000 * $5 = $1.048200
total:                                  $1.822213
```

This was approximately $0.00245 per annotation and accurately projected the
final total.

Both shards completed and merged successfully. The authoritative final result
is 1,490/1,490 valid annotations, 1,546,926 input tokens, 417,797
output/reasoning tokens, and a total Batch cost of $3.635911 (approximately
$0.00244 per annotation). The merged file hash is
`b4d473f7abf3c7414ead2e0b680de910f4e48cd0d3720ed2b18274fbb50d49cf`.
These values are preserved in `artifacts/teacher/annotation_summary.json`.

Teacher-generated rankings are a silver standard for training. The supplied
golden Fable annotations remain held-out evaluation labels and are never copied
into the training pipeline.

## 8. Create hierarchical stratified splits

Run:

```bash
python scripts/create_training_splits.py
```

The command validates the one-to-one quote/annotation join, validates every
teacher top-three against its available tariff panel, checks for exact-risk
overlap with the golden set, and performs two deterministic stratification
stages using seed `20260925`:

```text
1,490 synthetic scenarios
  -> reward calibration: 200
  -> SFT pool: 390 -> train: 300, validation: 90
  -> GRPO pool: 900 -> train: 810, validation: 90
```

Both stages balance hazard tier, age stratum, building type, construction
class, requested natural-hazard coverage, and owner occupancy. Outputs are:

```text
artifacts/splits/reward_calibration.jsonl
artifacts/splits/sft_train.jsonl
artifacts/splits/sft_validation.jsonl
artifacts/splits/grpo_train.jsonl
artifacts/splits/grpo_validation.jsonl
artifacts/splits/split_assignments.jsonl
artifacts/splits/split_manifest.json
artifacts/splits/split_distribution_report.md
```

Teacher annotations are included under `teacher_annotation` in all files that
need labels for training or offline metrics. They are deliberately omitted from
`grpo_train.jsonl`, where optimization must use only the frozen reward. The
manifest records source and output hashes, distributions, deviations from each
parent pool, and disjointness/exhaustiveness checks. The Markdown report
contains report-ready tables for both the three first-stage pools and all five
final splits.

## 9. Calibrate and freeze the reward

Run the deterministic offline search:

```bash
python reward_calibration/calibrate.py
```

The command uses only `artifacts/splits/reward_calibration.jsonl`. It does not
call an LLM and does not read the golden benchmark or GRPO validation split. It
performs the following work:

- validates all quote panels and teacher tariff IDs;
- creates five 40-example stratified folds;
- verifies its optimized scorer against `tools.reward.score_quotes` on every
  default-weight scenario;
- evaluates 20,000 deterministic random/anchor configurations;
- measures nested held-out selection performance; and
- freezes the winning weights.

Outputs are:

```text
artifacts/reward_calibration/best_weights.json
artifacts/reward_calibration/calibration_summary.json
artifacts/reward_calibration/candidate_results.jsonl
artifacts/reward_calibration/calibration_report.md
```

The completed seed-`20260926` run selected candidate `15957`. The supplied
default reward achieved 49.00% teacher Top-1 agreement; the selected reward and
the aggregated nested five-fold estimate both achieved 94.00%. The held-out
fold range was 92.5%-97.5%, and every fold selected the same candidate using
only its other four folds. The frozen weights are recorded authoritatively in
`best_weights.json`; do not tune them using GRPO validation or the golden set.

## 10. Validate the frozen reward

After `best_weights.json` has been frozen, run:

```bash
python reward_calibration/validate_frozen.py
```

This command does not update or reselect the weights. It performs:

1. a 20,000-candidate, five-fold negative control using deterministically
   permuted within-scenario teacher rankings;
2. overall and subgroup evaluation on the 90-example GRPO validation split; and
3. the locked deterministic baseline evaluation on all 50 supplied Fable
   scenarios.

Outputs are:

```text
artifacts/reward_calibration/validation/validation_summary.json
artifacts/reward_calibration/validation/validation_report.md
artifacts/reward_calibration/validation/golden_per_scenario.jsonl
```

Observed results:

- permutation control: 20.00% nested held-out Top-1 versus 16.31% theoretical
  random expectation; 48.33% Top-3 recall versus 48.92% expected; control passed;
- held-out silver validation: 93.33% Top-1 and 82.96% Top-3 recall, with every
  reported subgroup improving over the supplied default; and
- locked Fable evaluation: 84.00% Top-1, 79.33% Top-3 recall, and 81.17%
  supplied-reward ratio, versus 38.00% Top-1 for the supplied default reward.

The golden result is now evidence, not a tuning input. Do not alter the frozen
weights in response to it. Future model results should be compared against this
deterministic baseline under the same evaluation definitions.

## 11. Prepare Qwen3.5-2B SFT data

Run locally:

```bash
python training/prepare_sft_data.py
python training/train_sft.py --validate-only
```

This produces 300 training and 90 validation prompt-completion records under
`artifacts/training/sft_data/`. The prompt is exactly the canonical benchmark
prompt. The completion contains only ranks and tariff IDs, matching evaluation;
teacher rationales are retained in source data but are not completion targets.
The manifest records source/output hashes and confirms that no golden examples
were used.

## 12. Benchmark the unchanged Qwen3.5-2B model

On the GPU machine, install the training environment:

```bash
python -m pip install -r training/requirements.txt
```

Then run the unchanged open-weight baseline exactly once:

```bash
python training/run_eval.py \
  --stage base \
  --run-name golden-base-qwen35-2b \
  --confirm-golden-eval
```

This uses the original per-quote reward values in the golden file; it does not
apply the calibrated reward. The model prompt cannot see reward fields or Fable
labels. Do not use this golden result to change SFT hyperparameters.

## 13. Run Qwen3.5-2B SFT

First execute the compatibility smoke test:

```bash
python training/train_sft.py \
  --run-name sft-qwen35-2b-smoke \
  --smoke-test \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1
```

Verify the saved adapter can be reloaded and generate one validation result:

```bash
python training/run_eval.py \
  --stage sft \
  --adapter artifacts/training/runs/sft-qwen35-2b-smoke/final_adapter \
  --dataset artifacts/splits/sft_validation.jsonl \
  --limit 1 \
  --run-name sft-smoke-reload
```

After finite train/evaluation losses and a successful reload, run:

```bash
python training/train_sft.py \
  --run-name sft-qwen35-2b-seed3407
```

The run writes the final adapter, checkpoints, TensorBoard events, resolved
configuration, package/GPU provenance, loss history, runtime, and peak VRAM
under `artifacts/training/runs/sft-qwen35-2b-seed3407/`.

Validate the fixed adapter against the 90-example SFT validation set before
opening golden results:

```bash
python training/run_eval.py \
  --stage sft \
  --adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --dataset artifacts/splits/sft_validation.jsonl \
  --limit 90 \
  --run-name sft-validation-qwen35-2b
```

Once fixed, run its single golden evaluation:

```bash
python training/run_eval.py \
  --stage sft \
  --adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name golden-sft-qwen35-2b \
  --confirm-golden-eval
```

## 14. Prepare and run GRPO

Prepare self-contained prompt/reward records locally:

```bash
python training/grpo/prepare_data.py
```

The output contains 810 training and 90 validation prompts. Training contains
no teacher labels. Per-tariff business scores are precomputed using the frozen
reward weights and protected by hashes in the manifest.

After the SFT adapter is selected, run the mandatory one-step and ten-step
gates:

```bash
python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-smoke \
  --smoke-test

python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-10step \
  --max-steps 10 \
  --eval-steps 10 \
  --save-steps 10
```

Only after finite rewards/losses, nonzero within-group reward variance, a saved
adapter reload, and acceptable measured VRAM/runtime, run the bounded 50-step
job:

```bash
python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-seed3407
```

This saves checkpoints every 10 steps. Compare steps 10/20/30/40/50 against
the frozen SFT baseline on GRPO validation and prefer the earliest checkpoint
that improves the intended reward without degrading format or expert ranking.
Do not extend beyond 100 steps without explicit validation evidence and the
`--allow-long-run` flag.

Select using the untouched 90-example GRPO validation split, then perform one
locked golden evaluation with `training/run_eval.py --stage grpo`. This makes
base, SFT, and GRPO outputs directly comparable.

## 15. Reproduce the released model from a fresh Git clone

The reviewer-facing entry point is `HOW_TO_RUN.md`. The tracked, allowlisted
`release/` directory contains both portable adapters, prepared non-golden
training data, development evaluation records, exact configs, aggregate final
summaries, and checksums. Rebuild it from complete local experiment state with
`python scripts/create_release_bundle.py --force`, or verify a clone with
`python scripts/create_release_bundle.py --verify-only`.

The completed GPU artifacts are preserved as two independent run families:

```text
artifacts/training/runs/sft-qwen35-2b-seed3407/
├── checkpoints/checkpoint-14/
├── checkpoints/checkpoint-21/
└── final_adapter/

artifacts/training/runs/grpo-qwen35-2b-10step/
├── checkpoints/checkpoint-10/
└── final_adapter/
```

The local checkpoint directories include optimizer, scheduler, RNG, and trainer
state for resumption. Their final adapters are copied byte-for-byte into
`release/model/sft-final/` and `release/model/grpo-10step/`. Full optimizer
checkpoints remain local; the tracked release contains everything needed for
inference or clean retraining. Release hashes are recorded in
`release/SHA256SUMS` and `release/MANIFEST.json`. The complete remote experiment
journal is preserved at `documentation/runpod_training_journal.md`.

The base weights are deliberately not duplicated in the repository. The
reproduction driver downloads public `Qwen/Qwen3.5-2B` at immutable revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` and verifies the published model
weight SHA-256 before training or evaluation.

Inspect the plan without network or GPU work:

```bash
python scripts/reproduce_model.py \
  --action sft-grpo \
  --seed 3407
```

On a compatible CUDA machine, reproduce SFT and the optional ten-step GRPO
continuation:

```bash
# Begin with Python 3.12, torch 2.8.0+cu128, and torchvision 0.23.0+cu128.
python -m pip install -r training/requirements-runpod-lock.txt

python scripts/reproduce_model.py \
  --action sft-grpo \
  --seed 3407 \
  --execute
```

`training/requirements.txt` remains the portable compatibility specification;
`training/requirements-runpod-lock.txt` captures the exact successful RTX
A6000 user-space package versions. The causal-conv1d wheel in the lock is
specific to Linux x86_64, Python 3.12, PyTorch 2.8, CUDA 12, and the recorded
CXX11 ABI. Use a matching wheel on a different platform.

Use `--action sft` to stop after supervised LoRA training. The exact recorded
SFT configuration uses per-device batch 48, three epochs, learning rate `2e-4`,
LoRA rank/alpha 16, BF16, and seed 3407. GRPO starts from that newly produced
SFT adapter and runs ten optimizer steps with four generations per prompt,
learning rate `5e-6`, and KL coefficient `0.001`.

To download the base model and evaluate the already imported adapters only:

```bash
python scripts/reproduce_model.py \
  --action evaluate-imported \
  --seed 3407 \
  --execute
```

All ordinary reproduction evaluations use the development splits. Golden
evaluation is omitted by default and must not be used for checkpoint or
hyperparameter selection. `--confirm-golden-eval` exists only for a separately
authorized final replication after the checkpoint is fixed; the archived final
golden results should normally be reused instead of rerun.

The repository's canonical prompt implementation is `src/prompts.py`. The
singular `src/prompt.py` path in the older project status instructions is stale;
training and evaluation both use the same plural module and no second prompt
builder is introduced.

## Historical pilot

Before final collection, a 20-call pilot was run to validate mechanics. It
produced 16 usable scenarios, two `no_quotes`, two `insufficient_quotes`, and no
engine errors. The success-only JSONL therefore correctly contained 16 lines.
Pilot artifacts and earlier iterations are retained separately from the final
dataset and are not silently merged into it.
