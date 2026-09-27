# WG Recommendation Model - Working Log

This file records what we are building, why each stage exists, the evidence
collected so far, and the decisions that will matter in the final take-home
report. It is a working document: update it after every material experiment.
The exact command sequence and recovery procedure are maintained separately in
`replicate.md`.

## Objective

Fine-tune a small open-source model to rank the top three Wohngebäude insurance
tariffs for a building. The final model should move toward the expert/Fable
ranking quality of frontier models while retaining open-source inference cost.

The provided 50-scenario golden dataset is held out. It is used only for final
evaluation, never for training, reward selection, prompt tuning, or checkpoint
selection.

## Intended pipeline

```text
synthetic risks
    -> live quotes + heuristic risk context
    -> cached, scored training episodes
    -> teacher top-3 annotations
    -> reward calibration on a separate calibration split
    -> SFT warm-up
    -> GRPO alignment
    -> development-set checkpoint selection
    -> one locked golden-set evaluation
```

## Stage 1 - Synthetic scenario generation

### Why this stage exists

The supplied golden scenarios are evaluation-only, so the training pipeline
needs its own building risks. Synthetic generation also lets us deliberately
cover different hazard tiers, building ages, construction classes, occupancy
types, deductibles, and building types rather than depending on a small sample
of historical leads.

### Implementation

- `src/scenarios/generate.py` generates deterministic risks using a seeded RNG.
- A scenario ID is the SHA-256 hash of canonical normalized risk JSON.
- Exact matches with golden risks are excluded using only the golden `risk`
  field; golden annotations are not read by the generator.
- Marginal distributions are stratified rather than sampled independently.
- Generated files are written under `artifacts/scenarios/`; `data/` is not
  modified.
- Runtime validation checks schema, enums, construction years, dwelling-unit
  rules, rebuild-sum plausibility, uniqueness, golden collisions, and basic
  diversity.

### Current generated batch

Seed 42 produced 100 unique, non-golden scenarios:

| Dimension | Observed distribution |
|---|---|
| Hazard | 40 moderate, 29 elevated, 31 high |
| Age | 35 old, 36 middle, 29 modern |
| Building type | 21 single-family, 23 two-family, 17 multi-family, 20 semi-detached, 19 terraced |
| Owner occupied | 76 true, 24 false |
| Deductible requested | 48 true, 52 false |

The built-in deterministic and structural validation passed.

### Post-pilot postcode correction

The first 100-scenario batch synthesized suffixes from a hazard-tier prefix.
After review, the generator was changed to sample only the known-valid German
postcodes supplied by the original benchmark generator, then classify them via
the risk-context tool. The original pilot remains useful for validating the
mechanics and reward fields, but future training-data collection uses the
corrected postcode source. The complete pre-fix pilot was preserved under
`artifacts/pilots/2026-09-25/`; active artifact paths were reset and regenerated
so future default commands cannot mix the two dataset versions.

## Stage 2 - Quote collection and caching

### Why this stage exists

Training examples need the tariffs a broker would actually see. The provided
Mr-Money wrapper calls a live engine, so calls are rate-limited, potentially
costly, and should never be repeated unnecessarily. A file cache makes runs
resumable and preserves unsuccessful attempts for diagnosis.

### Implementation

- `src/engine/client.py` is the new entry point for live quote calls.
- Cache records are stored one per scenario under
  `artifacts/cache/quotes/<scenario_id>.json`.
- Cached records of every status are reused; automatic reruns do not repeat a
  failed or insufficient request.
- Uncached calls are separated by at least three seconds.
- Fire and glass are always requested. Natural-hazard/Elementar coverage is
  requested for elevated or high flood exposure.
- Quotes are scored with the provided default reward and sorted descending.
- Only scenarios with at least three quotes enter
  `artifacts/quoted/quoted_scenarios.jsonl`; all attempts remain in the cache.
- Atomic file replacement is used for cache and JSONL output writes.

### Pilot results - 2026-09-25

The first live pilot made 20 uncached calls:

| Outcome | Count |
|---|---:|
| Successful scenarios (at least 3 quotes) | 15 |
| Insufficient quote sets | 5 |
| No quotes | 0 |
| Engine errors | 0 |
| Success rate | 75% |

Successful scenarios contained 4-9 quotes. All 20 recorded call timestamps were
more than three seconds apart. Output validation passed: no duplicate scenario
IDs, no invalid prices or tariff IDs, rewards were present and sorted, only
successful records were included, and no credential values or authenticated
URLs were found.

Within the 15 successful scenarios, annual price and `coverage_score` varied in
all 15, and insured amount varied in 13. Deductible and `num_coverages` varied in
none. Every returned quote had deductible 0, including risks generated with
`with_deductible: true`. This reinforces the supplied reward-literacy warning:
for the current engine output, the default reward is driven mainly by price and
insured amount, while the optional `coverage_score` signal is available but has
zero default weight. The deductible behavior should be investigated before
claiming that it is a meaningful training signal.

Pilot success by construction class was BAK1 3/3, BAK2 5/5, FHG1 5/5, FHG2
2/3, and FHG3 0/4. This sample is too small to conclude that FHG3 is generally
unpriceable, but a larger bounded batch should verify the pattern before
large-scale collection. Removing FHG3 based only on four attempts would be
premature and would reduce intended construction diversity.

At the observed 75% success rate, obtaining 1,600 usable episodes would require
approximately 2,134 attempts before adding a safety margin. This estimate must
be revisited after a larger pilot because 20 attempts have high uncertainty.

### Example cache outcome

Scenario `4c15a613...` was structurally valid and classified as elevated flood
exposure, so Elementar was requested. The engine returned one Baloise tariff.
The cache correctly stored it as `insufficient_quotes`, with no error, and the
collector correctly excluded it from the success-only quoted dataset.

## Post-pilot fixes completed

Before scale-up, the implementation was hardened as follows:

1. `--cache-dir` now receives the same protected-path check as all other output
   paths.
2. Downstream quoted episodes expose a stable `risk` field; `normalized_risk`
   remains an internal cache field.
3. Collector inputs receive full structural validation before any live call.
4. Cached transient failures can be retried only with the explicit
   `--retry-transient-errors` flag.
5. A non-blocking file lock prevents two live collectors using the same cache
   directory concurrently.
6. `--seed` now deterministically shuffles collection order.
7. Future generated batches use known-valid supplied postcodes rather than
   synthetic suffixes.
8. Generated JSONL and cache files, Python bytecode, and environment files are
   excluded through `.gitignore`.

Quoted artifacts intentionally retain reward fields for calibration. Teacher
prompts must later exclude `reward`, `reward_components`, and
`reward_recommended_tariff_id`.

## Next stage - Dataset scale-up

Before scaling to 1,600 successful episodes:

1. Generate a fresh corrected candidate pool while keeping the seed/config
   recorded.
2. Collect another bounded batch and compare success rates by hazard tier,
   building type, and construction class.
3. Ensure the final successful dataset does not become skewed because one
   stratum produces fewer quotes.
4. Scale through resumable, explicitly approved batches until the target number
   of successful episodes is reached.
5. Freeze the collected dataset and create deterministic train, reward-
   calibration, and development splits before teacher annotation.

## Later stages and rationale

### Teacher annotation

The default numeric reward captures value-for-money but matches Fable's golden
top-1 only 38% of the time. New training scenarios therefore need independent
teacher top-3 rankings to learn the quality-first judgment missing from reward
argmax. Teacher prompts must see risk, risk context, and sanitized quotes only.

GPT-6 Sol was selected as the silver-label teacher after a complete run on the
50 held-out Fable scenarios using low reasoning and strict structured output. It
answered 50/50 scenarios with 84% top-1 agreement, 86.67% top-3 recall, and an
83.22% reward ratio. The synchronous selection run consumed 47,362 input and
6,895 output tokens and cost approximately $0.1637. These results support using
GPT-6 Sol as a scalable silver teacher, not as unquestionable ground truth.

`scripts/annotate_teacher_batch.py` prepares, submits, monitors, downloads,
validates, and merges GPT-6 Sol Batch API shards. It imports the same canonical
broker system prompt and scenario formatter as `scripts/run_eval.py`, adding
only the rationale output contract. Reward fields are excluded through an
explicit field allow-list, and annotations are stored separately from the
immutable quoted scenarios. Insurer and product names are copied from source
quotes during validation so the final output exactly matches the supplied
annotation schema without trusting the model to reproduce metadata.

The completed quote collection exhausted its 2,000-call budget with 1,490
successful scenarios (74.5%). Teacher preparation therefore uses two equal
745-request shards rather than the originally planned 1,600 examples.

The final quoted dataset contains 9,632 quotes, averaging 6.46 per scenario.
Hazard tiers remain diverse (38.79% moderate, 30.74% elevated, 30.47% high),
the three age strata are nearly equal, and the five building types range from
16.85% to 23.22%. BAK1, BAK2, FHG1, and FHG2 each contribute roughly one
quarter. FHG3 is absent because all 397 attempted FHG3 scenarios returned fewer
than three usable quotes; this is a quote-engine support limitation and must be
reported rather than silently repaired. The requested-deductible input is
balanced 50/50, although every returned tariff has deductible zero, so that
field provides no within-scenario ranking signal.

Offline teacher preparation generated two verified 745-line Batch API files.
Across all actual prompts it estimates 1,778,126 input and 670,500 output tokens,
for an expected Batch cost of $5.1306 before retries.

Both teacher shards completed and merged with 1,490/1,490 valid annotations and
zero schema problems. Actual provider usage was 1,546,926 input tokens and
417,797 output/reasoning tokens, costing $3.635911 through Batch, or about
$0.00244 per annotation. The difference from the $5.1306 pre-run estimate comes
mainly from actual output averaging about 280 tokens per scenario instead of
the conservative 450-token assumption. The merged annotation SHA-256 is
`b4d473f7abf3c7414ead2e0b680de910f4e48cd0d3720ed2b18274fbb50d49cf`.

### Reward calibration

Use only the new reward-calibration split to compare the provided reward with a
coverage-score-aware and potentially risk-aware reward. Freeze the chosen reward
before GRPO. Treat it as a proxy, not absolute truth.

Reward calibration was completed with `reward_calibration/calibrate.py` using
only the 200-example calibration split. The dataset contains 1,311 quotes. Price
and `coverage_score` vary within all 200 panels, insured amount varies in 168,
and both `num_coverages` and deductible vary in zero. Because the latter two
signals are not separately identifiable, the search combines them into a
static-protection term and maps it back to the supplied `w_cov:w_ded` ratio of
4:3.

The deterministic seed-`20260926` search evaluated 20,000 candidates. It uses a
simplex over static protection, insured amount, and coverage score; samples the
price penalty from 0.05 to 5.0 plus explicit zero-price anchors; and selects
lexicographically by teacher Top-1 agreement, Top-3 recall, and pairwise
ranking accuracy. Five stratified folds estimate selection stability without
touching GRPO validation or the golden benchmark.

The supplied default reward achieved 49.00% teacher Top-1, 68.83% Top-3 recall,
and 76.07% pairwise accuracy. Frozen candidate `15957` achieved 94.00%, 84.17%,
and 92.24%, respectively. The nested five-fold held-out aggregate was identical
at 94.00% Top-1, with fold results ranging from 92.5% to 97.5%; all five
four-fold training subsets independently selected the same candidate. Its
frozen weights are:

```text
w_cov   = 0.0928034111891159
w_ins   = 0.4102710203778362
w_ded   = 0.06960255839183692
w_score = 0.42732301004121104
w_price = 1.793611631132337
```

The authoritative machine-readable configuration is
`artifacts/reward_calibration/best_weights.json`. The golden set and GRPO
validation set were not used, and the calibration made no LLM calls. These
weights approximate the GPT-6 Sol silver teacher and must remain fixed during
GRPO; they are not independent ground truth.

### Frozen reward validation and falsification

After the weights were frozen, `reward_calibration/validate_frozen.py` ran three
predeclared checks without changing them.

First, a permutation negative control replaced each calibration annotation with
a deterministic random ordering of valid tariff IDs and repeated the complete
20,000-candidate nested selection. The theoretical random Top-1 expectation was
16.31%; nested held-out Top-1 was 20.00% with a 95% Wilson interval of
15.05%-26.09%. Top-3 recall was 48.33% versus a 48.92% random expectation, and
pairwise accuracy was 49.91%. The control passed, providing evidence against
label leakage or a mechanically inflated evaluator.

Second, on the untouched 90-scenario GRPO validation split, the frozen reward
achieved 93.33% teacher Top-1, 82.96% Top-3 recall, and 91.38% pairwise accuracy,
compared with 46.67%, 68.52%, and 73.96% for the supplied default. Every reported
hazard, age, building-type, construction-class, natural-hazard, occupancy, and
quote-count subgroup improved in Top-1. For subgroups with at least ten examples,
frozen Top-1 ranged from 83.33% to 100%.

Third, the authorized locked evaluation applied the unchanged weights to the 50
Fable golden scenarios. The supplied reward reproduced its documented 38.00%
Top-1 baseline. The calibrated reward achieved 84.00% Fable Top-1, 79.33% Top-3
recall, 90.41% pairwise accuracy, and an 81.17% supplied-reward ratio. This
matches GPT-6 Sol and GPT-5.4 on the primary Top-1 metric, matches GPT-5.4's
Top-3 recall, and trails GPT-6 Sol's 86.67% Top-3 recall. It is a deterministic
zero-inference-cost baseline, not a trained-model result. The weights are now
locked permanently and must not be revised using these golden outcomes.

The complete evidence is in
`artifacts/reward_calibration/validation/validation_report.md`, with all 50
per-scenario decisions preserved in `golden_per_scenario.jsonl`.

### Reward audit against the assignment brief

The six-page assignment explicitly confirms that changing the reward in this
way is intended. It identifies price, deductible, insured amount, and optional
coverage count as the supplied value-for-money inputs; exposes
`coverage_score` through an optional `w_score` that is off by default; warns
that `num_coverages` is usually uniform within a scenario; and says a strong
candidate should either weight `coverage_score` or justify a better term, then
evaluate both rewards. It also permits reward changes when they are explicit,
justified, and compared with the original. The calibrated reward therefore
follows the assignment's stated reward-literacy path rather than exploiting an
unintended field.

The exact inputs used by the frozen scorer are only quote attributes, all
min-max normalized within each scenario: `num_coverages`, `insured_amount`,
inverted deductible, `coverage_score`, and annual price. No scenario ID,
teacher label, Fable label, reward field, insurer/product name, building type,
hazard tier, or rationale enters the formula. Risk context influences which
covers were requested and therefore which quotes exist, but is not directly in
the calibrated formula.

An important post-validation correction is that raw variation is not the same
as effective reward variation. `tools/reward.py` maps a field to neutral 0.5
when its within-panel spread is at most 1% of its magnitude. Under that exact
normalization, effective signal availability is:

| Normalized component | Calibration (200) | Silver validation (90) | Golden (50) |
|---|---:|---:|---:|
| Coverage count | 0 | 0 | 0 |
| Insured amount | 2 | 1 | 0 |
| Deductible | 0 | 0 | 0 |
| Coverage score | 200 | 90 | 50 |
| Annual price | 200 | 90 | 50 |

Consequently, the separate `w_cov`, `w_ins`, and `w_ded` values should not be
interpreted causally. In almost every panel they jointly create a constant
protection intercept. On all 50 golden scenarios, the frozen ranking is exactly
equivalent to:

```text
reward = (0.2863385 + 0.4273230 * score_n)
         / (1 + 1.7936116 * price_n)
```

Thus the 0.84 golden Top-1 result is fundamentally a calibrated quality-versus-
price curve. Removing `coverage_score` returns golden Top-1 to the supplied
reward's 0.38; this is the decisive added signal. The assignment itself points
to precisely this blind spot.

Additional leakage checks found zero reward-related fields in the saved teacher
requests. The canonical prompt included `coverage_score`, price, insured amount,
deductible, and covers, exactly as the assignment says a broker/model should
receive. Across all 200 calibration, 90 validation, and 50 golden panels, the
frozen scorer had zero rounded Top-1 ties and zero Top-3-boundary ties. Rankings
were identical with rounded scores, unrounded scores, input-order tie-breaking,
and tariff-ID tie-breaking, ruling out accidental benefit from the original
default-reward quote ordering.

The high result is less mysterious after this audit. The quote engine repeatedly
returns a small stable product catalogue, `coverage_score` and price vary in
every panel, and both GPT-6 Sol and Fable were explicitly shown those fields.
The scorer learned their common, largely consistent quality/price trade-off.
This is valid for the current digital twin and deployment catalogue, but it does
not establish robustness to new insurers, changed tariff definitions, a shifted
`coverage_score`, or richer policy-wording features.

### Calibration provenance to preserve in the final report

The calibration operation was deliberately narrow: it changed configurable
weights only. For each calibration scenario, the script read the raw quote
attributes `num_coverages`, `insured_amount`, `deductible`, `coverage_score`, and
`price_annual`; reapplied the supplied within-scenario normalization, 1% noise
floor, protection-over-price formula, and four-decimal reward behavior; ranked
the resulting tariff IDs; and compared that ranking with GPT-6 Sol's separately
stored teacher Top-3.

The 20,000-candidate search did not read the quote's pre-existing `reward`,
`reward_components`, or `reward_recommended_tariff_id` when selecting weights.
Those fields remain in the source JSONL for provenance. The stored default
recommendation was consulted only by a parity assertion proving that the fast
calibration implementation exactly reproduces `tools.reward.score_quotes` under
the supplied default weights. It had no role in candidate scoring or selection.

No quote value, teacher output, scenario feature, normalization rule, or reward
equation was modified. The existing optional `w_score` term was activated and
the configurable weights were searched. The teacher request builder used an
explicit quote-field allow-list, so GPT-6 Sol saw the same risk, risk context,
and raw tariff facts used by evaluation, but no numeric reward or recommendation
derived from it.

The chronological evidence chain was:

```text
raw live-engine quote attributes
    -> unchanged supplied reward implementation with candidate weights
    -> candidate argmax and ranked Top-3
    -> comparison with GPT-6 Sol silver labels on calibration only
    -> five-fold stability check
    -> immutable best_weights.json
    -> untouched 90-scenario silver validation
    -> permutation and subgroup checks
    -> locked 50-scenario Fable evaluation
```

This chronology matters for the report: the 0.84 Fable result was observed only
after the weights were frozen. The weights must not be changed in response to
the validation or golden outcomes. Future SFT and GRPO work must load the frozen
configuration from `artifacts/reward_calibration/best_weights.json` and treat
the deterministic scorer as a fixed comparison baseline.

### Final-report headline: the target was reached before fine-tuning

The most important experimental finding is that reward calibration alone
reached the assignment's primary quality target. The task frames success as
moving an inexpensive open-source system from roughly 0.42 Top-1 agreement to
frontier-level agreement near 0.80. Before any SFT or GRPO, the frozen
deterministic scorer achieved 0.84 Fable Top-1 on the same 50-scenario golden
benchmark, equal to the reported GPT-5.4 and GPT-6 Sol results. It has no token
usage, model-serving latency, or per-request inference cost.

This should be presented as a substantive result rather than hidden as a
preprocessing detail: for the currently exposed quote features, much of the
frontier teacher's ranking policy can be distilled into an interpretable
value/quality formula. The movement from the supplied reward's 0.38 golden
Top-1 to 0.84 came primarily from assigning meaningful weight to
`coverage_score` while retaining a price penalty and protection intercept. Its 0.8117
supplied-reward ratio also shows the intended trade-off: it gives up some of the
original price-oriented reward to match expert quality preferences.

The claim must remain precise. This is a zero-cost deterministic baseline, not
a fine-tuned language model, and it does not by itself satisfy a deliverable
that specifically requires demonstrating SFT/GRPO. It also produces no natural
language rationale and depends on receiving the structured quote attributes at
inference time. The model experiments therefore remain useful for testing
whether a small open-source model can internalize the policy, emit valid ranked
JSON directly, and preserve the quality/cost result without explicitly running
the scorer.

Recommended final-report framing:

1. Establish the supplied reward and untuned Qwen baselines.
2. Show that offline reward calibration produces a 0.84 Top-1, zero-inference-
   cost symbolic baseline before model training.
3. Present the permutation control, held-out silver validation, subgroup
   stability, and locked golden evaluation as evidence that this is not leakage
   or calibration-set memorization.
4. Compare SFT and SFT+GRPO against both the original model baselines and the
   stronger calibrated-reward baseline.
5. Conclude whether fine-tuning adds deployable language/format capabilities or
   ranking quality beyond what the interpretable scorer already provides.

### SFT and GRPO

SFT teaches valid ranked JSON output and teacher-like decisions. GRPO then uses
the reward formula selected on the separate calibration split, plus structural
format/validity checks. Teacher labels are not exposed during GRPO training;
teacher agreement is measured only on the held-out GRPO validation split.

### Frozen training splits

The 1,490 silver-labeled synthetic scenarios are partitioned hierarchically
with `scripts/create_training_splits.py`. The first stratified stage creates a
200-example reward-calibration pool, a 390-example SFT pool, and a 900-example
GRPO pool. The SFT and GRPO pools are then independently stratified again into
300/90 and 810/90 train/validation splits. The supplied 50-example golden set
remains outside this process and is reserved for final evaluation.

Stratification simultaneously balances hazard tier, building-age stratum,
building type, construction class, requested natural-hazard coverage, and
owner occupancy. The split is deterministic under seed `20260925`, disjoint,
exhaustive, and checked for exact-risk overlap with the golden set. Teacher
labels are intentionally omitted from `grpo_train.jsonl`; they remain present
in reward calibration, SFT train/validation, and GRPO validation for offline
metrics and checkpoint selection. `artifacts/splits/split_distribution_report.md`
preserves report-ready count/share tables for both hierarchy levels.

### Final evaluation

After all prompts, weights, and checkpoints are frozen, evaluate once on
`data/golden_wg_recommendations.jsonl` using the provided evaluation metrics:
top-1 agreement, top-3 recall, reward ratio, token usage, and estimated cost.

## 2026-09-26 — GRPO implementation prepared

GRPO is isolated under `training/grpo/` so it can be copied to the GPU VM as a
single code directory. The VM also needs shared `training/common.py`, the
existing `training/requirements.txt`, and `artifacts/training/grpo_data/`.

`training/grpo/prepare_data.py` converted the disjoint 810/90 raw splits into
self-contained conversational data. The 810 training records contain no
teacher labels. The 90 validation records retain teacher Top-3 IDs only for
offline monitoring; none of the rollout reward functions accepts or reads that
field. The preparation step did not read the 50-example golden set.

The preparation step reapplied `tools/reward.py` using immutable candidate
`15957` from `artifacts/reward_calibration/best_weights.json`, then stored a
per-tariff score map in every prepared record. This avoids recomputing or
recalibrating the business formula on the GPU. The manifest records source,
output, and frozen-configuration hashes.

The GRPO objective has four separately logged bounded components:

| Component | Fixed weight | Purpose |
|---|---:|---|
| Strict JSON | 0.10 | Exact `top_3` schema with ranks 1-3 |
| Valid tariffs | 0.10 | Distinct IDs must belong to the offered panel |
| Calibrated Top-1 | 0.40 | Normalized frozen business utility of rank 1 |
| Calibrated Top-3 | 0.40 | Position-weighted utility using discounts 1, 0.5, 0.25 |

The 0.10/0.10/0.40/0.40 values are fixed reward-composition choices, not a
second calibration search. They must not be tuned on GRPO validation or golden
results. The underlying calibrated business weights remain unchanged.

`training/grpo/train.py` starts from the completed SFT LoRA adapter and defaults
to four sampled completions, 128 maximum completion tokens, learning rate
`5e-6`, KL coefficient `0.001`, Dr. GRPO loss, no reward-standard-deviation
scaling, and no vLLM. It records the resolved environment, checksums, token
audit, trainable parameters, component rewards, runtime, peak VRAM, and final
adapter. Full training is gated by one-step and ten-step smoke runs.

Offline preparation and validation succeeded with 810 unique train records, 90
unique validation records, disjoint scenario IDs, 3-9 offered tariffs per
scenario, and no teacher labels in training. A direct reward sanity check gave
the ideal frozen ranking full Top-1 and ranking reward, a reversed ranking lower
reward, and unknown tariff IDs zero validity/business reward.

After SFT reached 83/90 (92.22%) Top-1, 87.04% Top-3 recall, 100% strict JSON,
and 100% complete valid rankings on the held-out GRPO validation split, the
planned GRPO duration was reduced. The model is already only one Top-1 scenario
behind the frozen calibrated scorer on that split and exceeds its Top-3 recall,
so prolonged reward optimization has more downside risk than before SFT.

The first real GRPO run is therefore capped at 50 optimizer steps, with adapter
checkpoints every 10 steps. The 10/20/30/40/50 checkpoints must be compared on
GRPO validation and the earliest non-degrading checkpoint is preferred. The
trainer refuses runs above 100 steps unless `--allow-long-run` is supplied
explicitly. This is an operational safety guard, not evidence that 50 is
universally optimal; the decision is based on the observed strength of the SFT
policy and the assignment's time-boxed scope.

The SFT smoke mode was also tightened before the GPU run. It continues to
validate and tokenize-audit all 300 training and 90 validation records, but the
one-step forward/backward compatibility gate now trains and evaluates on a tiny
subset. This avoids running the complete 90-example validation pass twice for a
checkpoint that has received only one optimizer update. Full SFT continues to
use all 300/90 records unchanged.

### Evaluation token-ceiling correction

The first local base-model golden run used `max_new_tokens=128` and returned a
mean of 126.3 completion tokens. This shows that the generation ceiling bound
almost every response. It achieved 22% Top-1 and 27.33% Top-3 recall, but only
2% strict JSON and 2% complete valid Top-3 output; lenient parsing nevertheless
found a valid first tariff in all 50 responses. These measurements are retained
as a useful capped diagnostic, not overwritten.

The supplied `scripts/run_eval.py` allows Qwen up to 700 tokens. The local
checkpoint evaluator now uses the same 700-token default. Although the desired
JSON target is only 95-98 characters, a generous common ceiling removes output
truncation as a confound. The official base result must therefore be rerun under
a new run name with 700 tokens, and the exact same ceiling must be used for SFT
and GRPO comparisons. Generation remains deterministic with thinking disabled.

## Reporting evidence to preserve

For the final report, retain:

- generator and collector resolved configs;
- scenario and quote distribution summaries;
- cache hit, success, insufficient, no-quote, and error counts;
- actual teacher token usage and annotation cost;
- reward-search configurations and calibration results;
- SFT/GRPO configs and checkpoints;
- development results used for selection;
- the single final golden evaluation and cost assumptions;
- limitations, including heuristic hazard context and synthetic-data bias.

## Dataset distribution reporting

`scripts/summarize_quoted_dataset.py` converts the success-only quoted JSONL and
the complete cache into two reusable artifacts: a detailed JSON summary and a
Markdown report suitable for the final write-up. It reports scenario balance,
the collection funnel and subgroup success rates, quote/insurer diversity,
numeric price-quality-reward distributions, and within-scenario feature
variation. Run it after every meaningful collection batch and preserve the
final outputs with the experiment artifacts.

## 2026-09-26 — RunPod training artifacts imported

Imported the completed GPU bundle without overwriting `data/` or `tools/`.
The exact as-run training sources now live under `training/`. The full remote
journal is preserved at `documentation/runpod_training_journal.md` rather than
replacing this local pipeline journal.

Important checkpoints remain separate. SFT retains checkpoints 14 and 21 plus
its portable final adapter under
`artifacts/training/runs/sft-qwen35-2b-seed3407/`. GRPO retains checkpoint 10
and its portable final adapter under
`artifacts/training/runs/grpo-qwen35-2b-10step/`. A 59-file checksum inventory
is stored at `artifacts/training/important_checkpoints.sha256`.

The selected adapter is SFT. Its final golden metrics were Top-1 `0.84`, Top-3
recall `0.866667`, strict JSON `1.0`, and supplied reward ratio `0.811724`.
Ten-step GRPO matched Top-1 and reward ratio but reduced Top-3 recall to
`0.860000`, so it is retained as an optional ablation rather than promoted.

Added `scripts/reproduce_model.py` as the single orchestration entry point. It
pins the Hugging Face base revision and weight checksum, reproduces the actual
batch-48 SFT run with seed 3407, optionally continues for ten GRPO steps, and
uses development splits by default. Golden evaluation remains separately
gated and is not part of ordinary reproduction.

## 2026-09-26 — Git-contained evaluator release

Added a deterministic, allowlisted `release/` build so a private Git clone is
self-contained for model evaluation and clean retraining. The 85 MB release
contains the selected SFT adapter, optional GRPO adapter, prepared SFT/GRPO
data, non-golden evaluation records, exact configs, aggregate evaluation
summaries, and integrity metadata. It excludes golden rows and predictions,
cache entries, raw provider traffic, credentials, base weights, and optimizer
checkpoints.

`scripts/create_release_bundle.py` rebuilds or verifies the release and rejects
sensitive paths, Fable fields in released data, credential-like text, unsafe
destinations, and checksum drift. `HOW_TO_RUN.md` is the reviewer entry point
for Git LFS checkout, exact environment setup, direct `training/run_eval.py`
usage, expected SFT metrics, development evaluation, and SFT/optional-GRPO
retraining. `scripts/reproduce_model.py` now consumes only tracked `release/`
inputs for training and development evaluation.

## 2026-09-26 — Apple Silicon reproduction completed

Installed the pinned inference-only environment in Conda `work` and extended
`training/run_eval.py` with an explicit Transformers backend while preserving
Unsloth as the automatic CUDA choice. The local Qwen3.5-2B weight hash and SFT
adapter hash matched their recorded values.

A one-row MPS smoke test matched the CUDA raw output exactly. The subsequent
50-row golden portability run reproduced 42/50 Top-1, `0.866667` Top-3 recall,
`1.0` strict JSON, and `0.811724` reward ratio. All 50 rows matched on exact raw
output, Top-3 ordering, selection, correctness, validity, rewards, and token
counts. Only generation timing differed. Evidence and the exact command are in
`documentation/macos-verification.md`.

## 2026-09-26 — Production serving benchmark and measured cost model

### Why this benchmark was necessary

The first cost-quality draft had no measured Qwen3.5-2B serving price. It used a
parameter-scaled sensitivity range anchored to the managed Qwen3.5-9B estimate:
€0.046-€0.093 per 1,000 recommendations, with a €0.066 display midpoint. That
was explicitly labelled as an estimate, but it was still not a defensible
production cost because self-hosted inference does not scale linearly with
parameter count. GPU-hour price, batching, latency target, token lengths, idle
capacity, and serving software determine the actual unit economics.

The estimate was therefore superseded by a controlled vLLM benchmark. The
benchmarking philosophy was:

1. Serve the selected trained adapter, not a synthetic proxy or base model.
2. Use the real recommendation prompt/token profile on non-golden records.
3. Measure an entire traffic curve rather than report one context-free number.
4. Separate the saturated hardware floor from a conservative fleet-utilization
   planning assumption.
5. Preserve request-level data, GPU telemetry, logs, package versions, hashes,
   and resolved configuration so every number is auditable.
6. Measure latency and output validity alongside throughput; a cheap but slow or
   malformed response is not useful capacity.

### RunPod provisioning and runtime

The preferred A5000 and A6000 inventory was unavailable. Community and secure
A4000/3090/4090 alternatives were checked; the available machine was a secure
cloud RTX 4090 in Romania at `$0.74/hour`. The pod exposed 24,564 MiB physical
VRAM (23.52 GiB usable in vLLM), NVIDIA driver `580.178.04`, and a 450 W power
limit.

The first dependency installation was mistakenly placed on the network-backed
`/workspace` volume and was stopped when its filesystem phase proved slow. The
disposable vLLM environment and benchmark inputs were then placed on the pod's
local root disk. Only result artifacts were retained for download. The final
runtime was:

| Setting | Value |
|---|---|
| Serving stack | vLLM 0.29.0 |
| PyTorch | 2.13.0+cu130 |
| Base model | `Qwen/Qwen3.5-2B` |
| Base revision | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Adapter | selected `release/model/sft-final` LoRA |
| Precision | BF16 |
| Context ceiling | 2,048 tokens |
| Max active sequences | 64 |
| GPU memory target | 90% |
| Mode | text/language-model only |
| Thinking | disabled |
| Response ceiling | 64 tokens |
| API | OpenAI-compatible chat completions on pod loopback |

LoRA was served directly under the model name `wg-sft`; it was not merged or
re-quantized. The API bound only to `127.0.0.1`, so no public inference endpoint
or internet latency entered the measurements. Startup, model download, CUDA
graph compilation, and warm-up were completed before benchmark timing.

RunPod and Hugging Face credentials were sourced from environment variables
only and were never written to benchmark artifacts. The pod was deleted after
the complete artifact set was downloaded, and the RunPod CLI returned an empty
active-pod list.

### Workload and leakage boundary

The client is `scripts/benchmark_vllm_serving.py`. It uses only
`release/data/grpo/validation.jsonl`, never the 50-row golden set. Requests use
the actual prompt text and the selected adapter's chat template, with thinking
disabled. The observed workload averaged approximately:

- 1,023 input tokens per recommendation;
- 41 completion tokens per recommendation;
- 1,065 total tokens per recommendation.

The client records every request's latency, HTTP/error status, token counts,
raw output validity, strict-JSON status, and valid distinct top-3 status. A
parallel `nvidia-smi` sampler records utilization, VRAM, power, clocks,
temperature, and estimated energy throughout each level.

### Two-phase benchmark design

The first phase swept concurrency `1, 2, 4, 8, 16, 32, 64`, with 10 untimed
warm-up requests and 90 timed requests per level. It found the batching curve:

| Concurrency | Requests/s | p50 latency (s) | p95 latency (s) | GPU cost €/1k |
|---:|---:|---:|---:|---:|
| 1 | 3.344 | 0.295 | 0.360 | 0.053899 |
| 2 | 6.360 | 0.313 | 0.333 | 0.028343 |
| 4 | 11.240 | 0.353 | 0.385 | 0.016038 |
| 8 | 18.887 | 0.408 | 0.456 | 0.009544 |
| 16 | 31.309 | 0.497 | 0.543 | 0.005758 |
| 32 | 49.824 | 0.615 | 0.759 | 0.003618 |
| 64 | 58.793 | 1.038 | 1.228 | 0.003066 |

The high-concurrency windows in the first sweep lasted only a few seconds, so
they were not used alone for the final claim. A second sustained phase ran
1,200 requests at each of concurrency 16, 32, and 64, after 20 warm-ups:

| Concurrency | Requests/s | p50 (s) | p95 (s) | Mean GPU util. | Mean power | GPU cost €/1k |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 28.856 | 0.555 | 0.631 | 78.8% | 253.7 W | 0.006247 |
| 32 | 44.597 | 0.717 | 0.807 | 79.9% | 298.9 W | 0.004042 |
| 64 | 63.659 | 1.003 | 1.105 | 90.7% | 355.8 W | 0.002832 |

All 3,600 sustained requests succeeded, and every output was strict JSON with a
valid distinct top-3. At concurrency 64 the run processed about 67,767 total
tokens/s, drew an estimated 1.863 Wh during the 18.85-second window, and reached
100% instantaneous GPU utilization at its sampled maximum. Continuous batching
increased recommendation throughput by about 19x relative to one request at a
time while keeping p95 latency near 1.1 seconds.

### Cost calculation and selected planning assumption

The measured GPU-only cost is derived from achieved recommendation throughput:

```text
EUR per 1k recommendations
    = (GPU USD/hour / USD-per-EUR)
      / (recommendations/second * 3,600)
      * 1,000
```

The benchmark used the recorded conversion `$1.1403 = €1`. For a self-hosted
model, `price_assumption_eur_per_mtok` is not a vendor tariff; it is an effective
infrastructure rate for the measured workload:

```text
effective EUR/Mtok
    = (EUR per 1k recommendations)
      * 1,000
      / total tokens per recommendation
```

The saturated concurrency-64 result is the hardware-specific lower bound, but
production traffic rarely keeps every paid GPU continuously saturated. The
cost model therefore retains multiple utilization scenarios:

| Fleet utilization / operating mode | EUR/1k recommendations | Effective EUR/Mtok |
|---|---:|---:|
| 100% saturated | 0.002832 | 0.002660 |
| 50% | 0.005663 | 0.005320 |
| 25% planning assumption | 0.011327 | 0.010640 |
| 10% | 0.028317 | 0.026601 |
| One request at a time / no batching | 0.053899 | 0.050632 |

The scalar used when a report field requires one value is therefore:

```text
price_assumption_eur_per_mtok = 0.010640
```

This is the conservative 25%-fleet-utilization planning case, equivalent to
€0.011327 per 1,000 recommendations. It is approximately 16.5x below the
existing €0.1869/1k Qwen3.5-9B managed-price estimate, although a raw GPU cost
and a managed endpoint price do not include the same services or margin.

The cost scope is GPU runtime only. It excludes model download/startup,
storage, network, taxes, monitoring, engineering/operations, autoscaling lag,
and redundant failover capacity.

### GPU memory and model-instance interpretation

vLLM's own startup profiler reported:

| GPU allocation | GiB |
|---|---:|
| Weights plus non-PyTorch allocations | 3.91 |
| Peak activation | 0.40 |
| CUDA graphs | 0.14 |
| Available KV cache | 16.85 |

The engine exposed a 554,393-token KV cache and a theoretical 270.7 concurrent
2,048-token-sequence capacity before the configured 64-sequence application
ceiling. Five bare engine footprints fit arithmetically before allocating any
KV cache. A practical estimate is at most four independent instances if each
receives at least 1 GiB of KV cache; this four-instance number was calculated,
not validated by launching four simultaneous servers.

More independent instances on the same GPU do not reduce unit cost. They
duplicate weights, divide compute, and reduce KV-cache capacity. One vLLM
engine already batches requests across clients and demonstrated 64 concurrent
requests. The preferred topology is one continuously batched engine per GPU.
Additional replicas should be introduced for availability, isolation, or
capacity beyond one GPU. More GPUs increase aggregate capacity but normally do
not improve per-recommendation economics unless autoscaling reduces idle time
or a different GPU has better throughput per euro.

Further legitimate optimization candidates are a cheaper GPU benchmark,
quantization with a quality parity check, more efficient kernels, request
queueing that sustains larger batches, and autoscaling/serverless placement
that raises fleet utilization. None of those savings is claimed by the current
number.

### Important limitations

The benchmark is a controlled serving-capacity measurement, not a replay of a
production arrival trace. Requests were local to the pod, so public network and
gateway latency are absent. The sustained 1,200-request levels reused the
finite 90-record validation set. vLLM's server log reached approximately 69%
prefix-cache hit rate. A common system/prompt prefix makes some cache reuse
realistic, but exact repeated scenarios may make the saturated result
optimistic for a fully novel production stream. The 25% planning assumption
adds utilization headroom but is not a formal correction for prefix-cache
effects. A future production decision should replay unique or measured traffic,
including arrival timing, autoscaling, gateway overhead, and availability
requirements.

### Preserved benchmark artifacts

The complete evidence is under `artifacts/serving_benchmark/`:

- `qwen35-2b-sft-vllm-rtx4090-20260926/` contains the seven-level sweep;
- `qwen35-2b-sft-vllm-rtx4090-sustained-20260926/` contains the sustained runs;
- every run includes `resolved_config.json`, `summary.json`, request-level
  `requests.jsonl`, and sampled `gpu_telemetry.csv`;
- `environment/` contains the exact benchmark client, `pip freeze`, vLLM server
  log, NVIDIA XML report, idle-GPU snapshot, CPU/OS information, Prometheus
  metrics, model disk sizes, and adapter checksum;
- `artifacts/serving_benchmark/README.md` records the calculation and planning
  result in report-ready form.

The reproducible client is `scripts/benchmark_vllm_serving.py`; percentile and
output-parsing behavior is covered by `tests/test_serving_benchmark.py`.

## 2026-09-26 — Story figures and price-quality correction

`scripts/create_story_figures.py` now generates report-ready PNG and editable
SVG files under `artifacts/figures/`. The set covers:

1. dataset collection funnel and leakage-safe splits;
2. portfolio distributions across six risk dimensions;
3. supplied versus calibrated reward;
4. SFT loss and learning-rate progression;
5. GRPO reward, KL, and output-format diagnostics;
6. base -> SFT -> GRPO model progress;
7. cost versus Top-1, Top-3, and reward ratio;
8. completion-length and strict-JSON efficiency;
9. serving concurrency, latency, throughput, utilization, and cost.

The original price-quality chart used a logarithmic cost axis and the unmeasured
2B parameter-scaling range. It was replaced with a linear horizontal axis and
the measured RTX 4090 range: €0.002832/1k at sustained concurrency 64 through
€0.053899/1k without batching.

GPT-6 Sol was initially absent because `results/summary.json` stores its cost as
`cost_per_1k_recommendations_usd`, while the chart intentionally filters and
labels its axis in EUR. It is now included after an explicit conversion:

```text
$3.2735 / 1.1403 USD per EUR = €2.8707 per 1,000 recommendations
```

The FX rate and conversion are stated in the figure note; currencies are never
mixed directly. Reward argmax remains absent because it is a deterministic
scorer rather than a served model. Figure source hashes and the complete
serving-cost assumption are frozen in `artifacts/figures/resolved_config.json`.

The final benchmark and figure changes passed the complete local test suite:
`PYTHONPATH=. pytest -q tests/` returned 7 passing tests.
