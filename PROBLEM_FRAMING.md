# Problem Framing — Owning the WG Recommendation Model

This document explains the *why* behind the take-home: the business problem, the
digital-twin methodology, the reward, and what a strong solution looks like. The
candidate-facing brief is in `README.md`; the tools are in `tools/`.

---

## 1. The business problem

Afori is an AI insurance broker for the German market. In the **Wohngebäude
(residential building)** line, a lead arrives (a building: postcode, size, age,
construction, roof, usage…), a comparison engine (Mr-Money's *Rechenkern*) returns
a set of tariffs, and someone has to **recommend** the right product. This decision:

- is **high-volume** — it repeats on every WG lead,
- is **judgement-heavy** — cheapest is rarely best; the right answer trades
  coverage, deductible, insured sum and price against the building's actual risk,
- and is **measurable** — we can score a recommendation against an expert's, so it
  is a supervised/RL target, not a vibe.

That combination — repetitive, cognitive, scorable — is precisely where the
literature and practice converge on one deployment pattern: **fine-tune a small
open-source model with reinforcement learning against a scored copy of the
workflow.** It turns knowledge only Afori has (its tools, its data, its notion of a
good recommendation) into a model no vendor API can match, at a fraction of the
per-token cost of prompting a frontier model on every lead.

## 2. The digital twin

> _"We rebuilt the workflow as a digital twin: a simulated environment with the
> same tools, the same data, and the same stakes as the real thing… so every
> episode has a known correct answer to score against."_

We do the same for WG recommendation. The twin has three parts the model can act
in and be scored by:

1. **The real pricing engine** (`calculate_quotes_wg`). Not a mock — the live
   Rechenkern. A WG risk goes in; real tariffs come out, each with insurer,
   product, price, insured amount, deductible, coverage flags and the engine's own
   quality score.
2. **A reasoning tool** (`assess_wg_risk_context`). It converts a risk into
   natural-hazard exposure (flood / storm-hail / surge) and a coverage-emphasis
   hint. This is the analyst's context — *which* covers matter for *this* building
   — and it is what lifts the task above a trivial numeric argmax. It is a
   transparent heuristic the candidate can and should improve.
3. **A scorer** (`reward.py`). The value-for-money reward every recommendation is
   graded by — the RL signal.

An **episode** is one scenario: pull the context, price the building, choose a
recommendation, get scored. The model practices thousands of these; the fine-tune
learns the policy that maximises the reward and agrees with the expert.

## 3. The reward — value-for-money

A good WG recommendation is **protection per euro**. The reward encodes exactly
that. Within a scenario's quote set, each field is min-max normalised (so scenarios
with very different premiums are comparable):

```
protection = w_cov·coverages_n + w_ins·insured_n + w_ded·(1 − deductible_n)
reward     = protection / (1 + w_price · price_n)
```

- more **coverages** and higher **insured amount** raise protection,
- a lower **deductible** raises protection (less customer out-of-pocket),
- a higher **price** penalises the quote (the denominator),
- the recommended quote is `argmax(reward)`.

Defaults: `w_cov 0.4, w_ins 0.3, w_ded 0.3, w_price 1.0`. All configurable.

**How coverage enters (and a subtlety to notice).** The optional covers are real:
every building requests fire + glass, and a building in a flood-exposed postcode
also requests natural-hazard (Elementar) — so `num_coverages` (1 or 2) varies across
the set **with the building's hazard**, and Elementar-bearing tariffs cost visibly
more. But the engine returns only tariffs that satisfy the request, so **within a
single scenario coverage is usually uniform** — meaning the *within-scenario*
differentiators the reward ranks on are price, deductible, and `coverage_score` (the
engine's 0–100 quality score, exposed via the optional `w_score` term, off by
default). A strong candidate notices that the intra-scenario coverage term is flat
and either weights `coverage_score` or argues for a better coverage term, and
evaluates the change. Reward literacy is part of the test.

**Where context enters.** The base reward is risk-blind. The `assess_wg_risk_context`
tool both *drives which covers each building requests* (flood → Elementar) and lets a
policy reason *beyond* the reward — e.g. preferring a tariff whose coverage matches
the building's hazard. Candidates may fold context into a risk-aware reward; if they
do, they should evaluate both.

## 4. The golden benchmark

50 real WG scenarios, priced live through the twin, each annotated by **Fable** (a
frontier model) which independently ranked the **top-3 recommended products with a
broker rationale, blind to the numeric reward**. These are the *known correct
answers*.

- **Held out.** Candidates evaluate against it; they do not train on it.
- **Two references per scenario**: the reward-argmax (`reward_recommended_tariff_id`)
  and Fable's expert top-3 (`fable_annotation.top_3`). Where they disagree is the
  interesting part — a learned policy that captures Fable's judgement *and* scores
  well on reward is the goal.

Schema of each JSONL line: `id`, `risk`, `risk_context`, `quotes` (each with
`reward` + `reward_components`), `reward_recommended_tariff_id`, and
`fable_annotation.top_3` (`rank`, `tariff_id`, `insurer`, `product`, `rationale`).

## 5. What "good" looks like

- A model that **agrees with Fable's top-3** more often than a zero-shot frontier
  prompt, and whose recommendations **score at least as well on reward** as the
  argmax baseline.
- **Cheaper**: a defensible tokens/€ per 1,000 recommendations well below a
  frontier baseline — the whole point of owning the model.
- **Honest evaluation**: baselines included (reward-argmax, frontier zero-shot),
  limitations named, the reward's blind spots acknowledged.

## 6. Extension ideas (not required)

- A **risk-aware reward** that uses `coverage_emphasis` to value the covers that
  matter for the building.
- Replace the heuristic hazard context with real **ZÜRS / GDV** data.
- **Multi-objective / ranking** training (recommend a ranked top-3, not one).
- **Calibration**: have the model express confidence and abstain on close calls.

---

*The bar is not a perfect model. It is a real digital-twin loop, an honest number
against the golden set, and a small model that beats a frontier baseline at a
fraction of the cost.*
