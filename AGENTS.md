# Afori WG Recommendation — Take-home

## Goal
Fine-tune Qwen3.5-9B (LoRA) to recommend Wohngebäude insurance tariffs: given a building risk
and its priced quotes, output a ranked top-3.

Pipeline: scenario generation → quote crawl (cached) → teacher labels (Fable, Batch API)
→ SFT → GRPO → evaluation on the held-out golden set.

Target: move untuned Qwen (top-1 agreement 0.42) toward frontier (~0.80) at ~€0.19 / 1k recs.

## Status
<!-- Update this line at the start of each session -->
Current stage: final reporting — the completed RunPod SFT and 10-step GRPO runs
have been imported and the portable reproduction flow has been verified. SFT is
the selected model (golden top-1 0.84); GRPO matched top-1 and slightly reduced
top-3 recall. The active work is the four-page submission brief and the detailed
technical experiment report. Do not run another golden evaluation for checkpoint
selection.

## Hard rules (never break these)
1. `data/golden_wg_recommendations.jsonl` is HELD OUT.
   - Never train on it.
   - Never tune weights, hyperparameters or prompts on it.
   - Never select checkpoints on it.
   - Allowed: reading the schema, feature (NOT label) distributions, and final evaluation via `src/eval/` only.
   - Never read `fable_annotation` fields outside `src/eval/`.
2. The live quote engine is called ONLY through `src/engine/client.py` (disk-cached, >= 3s between calls).
   - Never call `tools/calculate_quotes_wg.py` directly in a loop.
   - Never make more than 20 live engine calls without my explicit approval.
3. Secrets (`MM_ID`, `MM_PA`, `ANTHROPIC_API_KEY`) come from environment variables only.
   - Never print, log, commit or write them to files. Never read `.env`.
4. Never modify anything in `tools/` or `data/`. Wrap or copy instead.
5. ONE prompt format: every model prompt (teacher, eval, SFT, GRPO) is built by `src/prompt.py`.
   No other code constructs prompts.
6. Don't spend money without asking.
   - No more than 20 Anthropic API requests without approval.
   - No Batch API submission without approval.
   - No GPU jobs longer than 10 minutes without approval.
7. If a provided tool looks wrong, ambiguous or surprising, STOP and tell me.
   Don't silently work around it.

## Conventions
- Python 3.11, type hints, small modules. No notebooks for pipeline code.
- Every script:
  - uses argparse with `--seed` (default 42);
  - writes outputs under `artifacts/<stage>/`;
  - saves its resolved config as JSON next to its outputs.
- Data: JSONL, one scenario per line, keyed by `scenario_id` (hash of the normalized risk JSON).
- Quote order: prompts reference tariffs as Q1..Qn, shuffled with a seeded RNG.
  The Qn → tariff_id mapping is stored in the scenario record.
- Model output: JSON `{"top3": ["Qi", "Qj", "Qk"]}`, with distinct, valid indices only.
- Tests: `pytest tests/` must pass before anything that costs money or GPU time.
  Metric code needs unit tests with hand-computed cases.
- Dependencies: ask before adding heavy ones. Check the installed library version's docs
  for argument names (TRL/vLLM/Unsloth APIs change) instead of guessing.

## Layout
```
src/engine/      cached client for calculate_quotes_wg + assess_wg_risk_context
src/scenarios/   risk generator (+ dedupe against golden risks)
src/prompt.py    single prompt serializer + output parser
src/reward/      reward variants: R0 (wraps tools/reward.py), R1 (+w_score), R2 (+risk-aware), R3 (learned)
src/labels/      teacher labelling (sync pilot + Anthropic Batch API, resumable)
src/train/       sft.py, grpo.py, dpo.py (configs in configs/*.yaml)
src/eval/        metrics.py, bootstrap.py, run_eval.py
scripts/         entry points (crawl, diagnostics, smoke tests)
artifacts/       outputs (gitignored except small summaries)
DECISIONS.md     decision log
```

## Metrics (`src/eval/metrics.py`)
- `top1`: model rank-1 tariff_id == Fable rank-1 tariff_id
- `hit3`: model rank-1 is in Fable's top-3
- `reward_ratio`: reward(model rank-1) / max reward in the scenario, under R0 and R1
  (confirm this matches how BENCHMARK.md computes it)
- `invalid_rate`: unparseable output, unknown index, or duplicate picks
- Confidence intervals: bootstrap with 10k resamples; paired bootstrap for model-vs-model comparisons
- Validation gate: must reproduce the BENCHMARK.md baseline rows from `results/` exactly

## Splits
- Own scenarios: ~1,400 train / 200 dev (seeded split by scenario_id)
- Checkpoint selection, reward weights, hyperparameters: dev only
- Golden: final reported numbers only

## DECISIONS.md
Whenever a design choice is made or proposed, append an entry with four parts:
options considered / choice / why / what evidence would change it.
