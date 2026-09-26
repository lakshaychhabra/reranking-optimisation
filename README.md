# Take-Home: Own the Recommendation Model

## Submission quickstart

The selected deliverable is a Qwen3.5-2B SFT LoRA adapter with 0.84 Top-1
agreement on the held-out 50-scenario benchmark. A fresh clone contains the
portable adapter and prepared non-golden training data under `release/`; the
public base model is downloaded at its pinned Hugging Face revision.

Start with **[HOW_TO_RUN.md](HOW_TO_RUN.md)** for exact Git LFS setup, dependency
installation, direct `training/run_eval.py` commands, expected metrics, SFT
reproduction, and the optional GRPO continuation.

The prepared, non-golden training and development data is also published as the
public Hugging Face dataset
**[lakshaychhabra/wg-recommendation-data](https://huggingface.co/datasets/lakshaychhabra/wg-recommendation-data)**.
It contains the 300/90 SFT splits, 810/90 GRPO splits, their manifests, and the
corresponding development-evaluation records. The held-out golden benchmark is
explicitly excluded.

Quick integrity check after cloning:

```bash
python3 scripts/create_release_bundle.py --verify-only
python3 scripts/reproduce_model.py --action verify-imported
```

### An ML-Engineering challenge — fine-tuning a small model to recommend Wohngebäude insurance

> _"A model can only practice a workflow it can actually perform, fail at, and retry."_

Afori is an AI insurance broker for the German market. One of the most valuable —
and most repeated — decisions in the business is deceptively simple: **given a
building and the tariffs a comparison engine returns, which product should we
recommend?** A broker does this hundreds of times a week. It is high-volume,
judgement-heavy, and every recommendation is scored by an outcome we can measure.

That is exactly the shape of task where the winning move is **not** a bigger
frontier prompt. It is to **own the model**: take a small open-source model and
fine-tune it with reinforcement learning against a scored copy of your own
workflow — a *digital twin* — until it matches or beats frontier models on the
task, at a fraction of the cost and with none of the vendor lock-in.

**Your mission:** build the best fine-tuned model you can for **Wohngebäude (WG,
residential building) product recommendation**, trained inside the digital twin
we give you, and prove it against a held-out, expert-annotated benchmark.

---

## The digital twin

We hand you a working copy of the real workflow — the same engine, the same
reward, the same stakes:

| Piece | What it is | File |
|---|---|---|
| **Quote tool** | The live Mr-Money "Rechenkern" pricing engine, WG only. A risk in → real tariffs out. | `tools/calculate_quotes_wg.py` |
| **Risk-context tool** | Turns a risk into natural-hazard exposure + coverage-emphasis hints. Context to *reason* with, not the answer. | `tools/assess_wg_risk_context.py` |
| **Reward / scorer** | The value-for-money reward every recommendation is scored by. This is your RL signal. | `tools/reward.py` |
| **Golden benchmark** | 50 real WG scenarios, priced live, with the **top-3 products ranked by Fable** (a frontier model) as the known-correct answers. **Held out — evaluation only.** | `data/golden_wg_recommendations.jsonl` |

You act inside the twin the way a broker would: pull the risk context, price the
building, weigh coverage against price, and commit a recommendation. The twin
scores it. The gap between your model's choice and the expert answer is the signal
your fine-tune has to close.

See **`PROBLEM_FRAMING.md`** for the full methodology, and **`tools/README.md`**
for how to run each tool.

---

## What you build

A pipeline that produces a **fine-tuned small open-source model** which, given a WG
risk and its priced tariffs, recommends the best product(s).

You choose the recipe. A strong submission will typically include:

1. **Rollout generation** — use `calculate_quotes_wg` + `assess_wg_risk_context`
   to build training episodes (you generate your own training scenarios — the 50
   golden ones are for evaluation only).
2. **A reward-driven fine-tune** — RL / GRPO (or a preference/ranking method you
   justify) on a small open-source base model, optimising the provided reward.
3. **Evaluation** against the held-out golden set: how well does your model's
   recommendation agree with the Fable-ranked top-3, and how does its reward
   compare to (a) the reward-argmax baseline and (b) a frontier model prompted
   zero-shot on the same task?
4. **A cost/quality argument** — tokens/€ per 1,000 recommendations for your model
   vs. the frontier baseline. Owning the model only wins if it's cheaper *and*
   competitive.

---

## Baselines to beat

We ran three models zero-shot against the golden set so you have reference numbers
(details in **`BENCHMARK.md`**, raw data in `results/`):

| Model | top-1 vs Fable | reward-ratio | € / 1,000 recs |
|---|---:|---:|---:|
| reward-argmax (the scorer itself) | 0.38 | 1.00 | 0.00 |
| **Qwen3.5-9B** (open-source, untuned) | **0.42** | 0.98 | **€0.19** |
| Gemini 3 Flash (frontier) | 0.80 | 0.76 | €0.89 |
| gpt-5.4 (frontier) | 0.84 | 0.80 | €1.34 |

The untuned open-source model captures value-for-money (reward-ratio ≈ 1.0 — it
essentially picks the reward optimum) but misses the expert's quality-first judgement
(lowest agreement). Frontier models match the expert ~2× as often, at 5–7× the cost.
**Your target: fine-tune the small open-source model from ~0.42 up toward
frontier-level agreement (~0.80) while keeping its ~€0.19 / 1k cost.** Beating frontier
*quality* at open-source *cost* is the bar.

## Deliverables

- **Code**: the training + evaluation pipeline, reproducible from these tools.
- **The fine-tuned model** (weights or adapter) or a clear path to reproduce it.
- **A short write-up (≤ 4 pages)**: your approach, the reward you optimised (and
  any changes you made to it, justified), results on the golden benchmark vs. the
  two baselines, the cost/quality trade-off, and what you'd do next with more time.
- **An evaluation script** we can run to reproduce your headline numbers against
  `data/golden_wg_recommendations.jsonl`.

## How we evaluate you

| Dimension | What we look for |
|---|---|
| **Methodology** | A real digital-twin loop: rollouts, a reward you understand, honest evaluation. Not a prompt in a trench coat. |
| **Result** | Agreement with the Fable top-3 + reward vs. baselines. Beating frontier at lower cost is the bar. |
| **Reward literacy** | Do you understand what the reward rewards — and where it's blind? (Hint: inspect what `num_coverages` actually varies across a scenario.) |
| **Engineering** | Reproducible, readable, sane data handling. |
| **Judgement** | Sensible scope, clear trade-offs, honest about limitations. |

## Rules & notes

- **Time-box: ~1 week, part-time.** We care about the reasoning and the loop, not
  polish. A smaller model that clearly beats a baseline beats a heroic setup that
  doesn't.
- **The golden set is held out.** Don't train on `data/golden_wg_recommendations.jsonl`.
- **Credentials for the quote engine** are provided separately (env `MM_ID` / `MM_PA`).
  Be gentle with the live engine — cache, and space out calls (~3s).
- You may **change the reward** if you can argue it recommends better products —
  just make the change explicit and evaluate both.
- Use any open-source base model and any RL/fine-tuning stack you like.

Questions are welcome — a good clarifying question is a positive signal, not a
negative one.

## Training-data generation

`src/scenarios/generate.py` + `src/engine/client.py` build the candidate's own
training scenarios and price them — separate from, and never touching,
`data/` or `tools/`. All outputs land under `artifacts/`.

**Generate 100 local scenarios** (stdlib only, zero network calls,
zero live-engine cost):

```bash
python3 scripts/generate_scenarios.py \
  --n 100 \
  --seed 42
```

**Validate generated scenarios** (regenerates in-memory with the same seed,
checks determinism + schema + diversity, writes nothing):

```bash
python3 scripts/generate_scenarios.py \
  --n 100 \
  --seed 42 \
  --validate-only
```

**Dry-run quote collection** (plans the run — cache hits vs. live calls needed —
makes zero engine calls and zero sleeps):

```bash
python3 scripts/collect_quotes.py \
  --input artifacts/scenarios/scenarios.jsonl \
  --target 20 \
  --max-live-calls 20 \
  --dry-run
```

**Summarize the successful quoted dataset and the complete cache funnel:**

```bash
python3 scripts/summarize_quoted_dataset.py \
  --input artifacts/quoted/quoted_scenarios.jsonl \
  --cache-dir artifacts/cache/quotes
```

This writes machine-readable `artifacts/quoted/distribution_summary.json` and a
report-ready `artifacts/quoted/distribution_report.md`. The report covers risk
class balance, collection success rates, quote/market diversity, price and
quality distributions, and which reward inputs actually vary within a scenario.

**The live pilot command** (documented here, **not run** as part of this
change — requires `MM_ID` in the environment and explicit sign-off):

```bash
python3 scripts/collect_quotes.py \
  --input artifacts/scenarios/scenarios.jsonl \
  --target 20 \
  --max-live-calls 20 \
  --execute-live
```

Notes:

- `data/` is held out and is never written to; `tools/` is never modified —
  both CLIs refuse to write under either path.
- Cache records live one-per-scenario under `artifacts/cache/quotes/<scenario_id>.json`.
  Repeated runs reuse these files; a scenario is attempted at most once, ever
  (any cached status counts as a completed attempt, not just `success`).
- The cache keeps every attempt, but `quoted_scenarios.jsonl` only ever
  receives `status == "success"` records — no_quotes/insufficient/error
  scenarios are counted in the summary but never written into the
  training-data output.
- Live calls are throttled to at least 3s apart (`time.monotonic()`-based),
  enforced both by the CLI's `--delay` floor and, independently, inside
  `QuoteEngineClient` itself — so the floor can't be configured away.
- Scenarios sample from the benchmark generator's list of known-valid German
  postcodes, stratified through the supplied risk-context tool; arbitrary
  postcode suffixes are not synthesized.
- Input order is deterministically shuffled with `--seed` before collection.
- Only one live collector may use a cache directory at a time; a file lock
  prevents concurrent processes from bypassing caching or rate limiting.
- Cached transient errors remain terminal by default. Pass
  `--retry-transient-errors` explicitly to retry only that status.
- `--max-live-calls` has no code-enforced ceiling — it's whatever you pass
  (default 20), and applies per invocation, not cumulatively across runs.
  Run volume is a judgment call, not an automatic stop.
- The first live pilot needs explicit approval before running.
