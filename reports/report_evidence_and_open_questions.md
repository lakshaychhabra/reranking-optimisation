# Report Evidence and Open Questions

This register accompanies the four-page submission brief and the detailed
technical report. Exact numbers should be changed only after checking the
primary artifact listed below.

## Evidence hierarchy

1. Machine-readable run summaries, manifests, configs, trainer state, and
   checksums control exact metrics and settings.
2. `documentation/runpod_training_journal.md` controls the execution chronology.
3. Stage reports under `artifacts/` control dataset and reward interpretation.
4. `DECISIONS.md` controls recorded choices and reversal conditions.
5. `gpt.md` supplies the narrative spine but may contain superseded plans.
6. `HOW_TO_RUN.md`, `release/`, and `documentation/macos-verification.md`
   support reproduction claims.

## Claim map

| Claim area | Primary evidence | Status |
|---|---|---|
| 2,400 generated risks | `artifacts/scenarios/final_generate_summary.json` | Verified |
| 2,000 quote attempts and 1,490 successes | `artifacts/quoted/final_collect_summary.json` | Verified |
| 9,632 quotes and portfolio distributions | `artifacts/quoted/final_distribution_summary.json` | Verified |
| Teacher usage and cost | `artifacts/teacher/annotation_summary.json` | Verified |
| Leakage-safe 200 300 90 810 90 split | `artifacts/splits/split_manifest.json` and `split_distribution_report.md` | Verified |
| Reward search and frozen weights | `artifacts/reward_calibration/calibration_summary.json` and `best_weights.json` | Verified |
| Permutation silver validation and locked reward checks | `artifacts/reward_calibration/validation/validation_summary.json` | Verified |
| SFT settings runtime memory and loss | `artifacts/training/runs/sft-qwen35-2b-seed3407/run_record.json` | Verified |
| GRPO settings stability runtime and memory | `artifacts/training/runs/grpo-qwen35-2b-10step/run_record.json` | Verified |
| Base SFT and GRPO golden metrics | `release/evaluation/*-golden-summary.json` | Verified |
| SFT and GRPO development comparison | `release/evaluation/*grpo-validation-summary.json` | Verified |
| Serving throughput latency utilization and cost | `artifacts/serving_benchmark/qwen35-2b-sft-vllm-rtx4090-sustained-20260926/summary.json` | Verified |
| RunPod total bill | Owner-provided figure of `$4.00` for training evaluation and benchmarking | Owner confirmed |
| End-to-end cash spend | About `$3.80` teacher API plus `$4.00` RunPod equals about `$7.80` | Verified plus owner confirmed |
| Exact Apple Silicon reproduction | `documentation/macos-verification.md` | Verified |
| Adapter hashes and portable release | `release/MANIFEST.json`, `release/SHA256SUMS`, and `HOW_TO_RUN.md` | Verified |

## Owner wording to confirm

1. Should Fable be described as an expert reference, a frontier-model reference,
   or both? The repository uses both ideas in different places.
2. Should the title page include the author's full name, repository URL, and
   public Hugging Face dataset URL?
3. Will the technical report be uploaded as supplementary material or linked
   from the repository only? The four-page brief remains self-contained either
   way.
4. How much internal evaluation chronology should appear in the submitted
   version? The detailed report currently states the safeguards and final frozen
   comparisons without reproducing every operational detour in the journal.

## Deliberate qualifications

- Ten GRPO steps do not establish that GRPO cannot improve. They establish only
  that this run did not improve the selected golden metrics broadly enough to
  replace SFT.
- GPT-6 Sol labels are silver supervision. They are not equivalent to Fable or
  human broker ground truth.
- The calibrated scorer's 84 percent golden Top-1 is an interpretable baseline,
  not a trained-model result and not an independent label source.
- The measured cost covers GPU runtime at the configured hourly rate. It excludes
  storage, networking, startup, redundancy, taxes, monitoring, and engineering.
- Recorded experiment spend is approximately `$7.80`: about `$3.80` of teacher
  API usage and an owner-confirmed `$4.00` RunPod bill. The RunPod bill is not
  allocated across individual training and evaluation stages.
- The 50-scenario golden set is too small for strong conclusions from one-scenario
  differences.
