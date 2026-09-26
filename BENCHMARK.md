# Baseline Benchmark — WG recommendation on the golden set

Reference numbers the candidate's fine-tuned model must beat. Produced by
`scripts/run_eval.py` against `data/golden_wg_recommendations.jsonl` (50 held-out
scenarios). Every model saw the same input — the building (`risk` + `risk_context`)
and its tariffs — and returned a ranked top-3 by `tariff_id`, **blind to the reward
and blind to Fable's answer**. Raw per-scenario records are in `results/<model>.jsonl`;
machine-readable summary in `results/summary.json`.

## Results (50 scenarios, each model answered all 50)

Quotes carry real optional covers (fire + glass on every building; natural-hazard /
Elementar in flood-exposed postcodes), so `num_coverages` (1 or 2) and coverage fit
are live signals in the golden set.

| Model | Source | top-1 vs Fable | top-3 recall | reward-ratio | Mean tokens (in/out) | **€ / 1,000 recs** |
|---|---|---:|---:|---:|---:|---:|
| **reward-argmax** | the scorer itself | 0.38 | — | **1.00** | — | 0.00 |
| **Qwen3.5-9B** | open-source (HF/Together) | 0.42 | 0.66 | **0.98** | 995 / 43 | **0.19** |
| **Gemini 3 Flash** | frontier (Google) | 0.80 | **0.88** | 0.76 | 998 / 236 | 0.89 |
| **gpt-5.4** | frontier (Azure) | **0.84** | 0.79 | 0.80 | 894 / 41 | 1.34 |

### Metrics
- **top-1 vs Fable** — fraction of scenarios where the model's #1 pick equals Fable's #1 (the expert gold). The primary quality metric.
- **top-3 recall** — mean overlap between the model's top-3 and Fable's top-3 (∩ / 3).
- **reward-ratio** — mean `reward(model's #1) / max reward in the scenario`. How close the pick is to the value-for-money optimum (1.0 = it picked the reward-argmax).
- **€ / 1,000 recs** — cost to produce 1,000 recommendations at the price assumptions below, from measured tokens.

## What the numbers say (and why this task is interesting)

1. **The reward is not the expert.** Picking `argmax(reward)` — the pure value-for-money
   optimum — agrees with Fable's #1 only **38%** of the time. Fable routinely pays up
   for coverage quality on old / high-hazard / rented buildings, and respects the
   customer's deductible preference, in ways the raw reward does not. So a model that
   only maximises the reward will score badly on expert agreement.

2. **The untuned open-source model is a value-for-money machine, not an expert.**
   Qwen3.5-9B has a **near-perfect reward-ratio (0.98)** — it hugs the cheap/high-reward
   options, essentially the argmax baseline — but **low expert agreement (top-1 0.42)**.
   It captures "value" out of the box and misses the nuance.

3. **Frontier models capture the nuance but cost 5–7× more.** Gemini (0.80) and gpt-5.4
   (0.84) match Fable roughly twice as often as Qwen, at a **lower** reward-ratio (~0.76–0.80,
   i.e. they knowingly trade price for quality, like Fable) — and at **€0.89–1.34 / 1k**
   vs Qwen's **€0.19 / 1k**.

4. **The target.** This is the whole premise of the take-home: **take Qwen3.5-9B from
   top-1 0.42 up to frontier-level agreement (~0.80) — while keeping its ~€0.19 / 1k cost.**
   If a fine-tune reaches frontier-quality recommendations at ~⅕–⅐ the price, owning the
   model wins. Beating frontier *agreement* at open-source *cost* is the bar.

## Price assumptions (transparency)

Cost is measured tokens × the price table in `run_eval.py` (`PRICES_EUR_PER_MTOK`),
best-effort public estimates for Aug-2026 models — **adjust to your own contract**:

| Model | € / 1M input | € / 1M output |
|---|---:|---:|
| Qwen3.5-9B | 0.18 | 0.18 |
| Gemini 3 Flash | 0.30 | 2.50 |
| gpt-5.4 | 1.10 | 8.80 |

Note: a self-hosted open-source model at scale is typically far cheaper than the HF
serverless estimate used here, so Qwen's real-world cost advantage is understated.

## Caveats

- **gpt-5.4 stands in for GPT-5.5**, which is not provisioned in the available Azure
  credentials; 5.4 is the newest working Azure GPT. Labelled as such throughout.
- **Gemma was dropped** — not available as a managed model on the Vertex project
  (would need a self-deployed Model Garden GPU endpoint).
- **Gemini 3 Flash** = `gemini-3-flash-preview` (there is no literal "3.5 flash").
- All models were run **zero-shot** (a single instruction prompt, temperature 0). A
  candidate's fine-tune should be compared against these same zero-shot baselines.
- Reproduce with: `ENV=<.env> python3 scripts/run_eval.py --models qwen gemini gpt54`.
