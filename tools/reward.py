#!/usr/bin/env python3
"""
reward.py — the scorer at the centre of the digital twin.

Given a scenario's quote set, it assigns each quote a scalar "value-for-money"
reward in roughly [0, 1] and identifies the recommended quote (argmax). This is
the reward signal an RL fine-tune (e.g. GRPO) optimises against, and the same
function you can use to score a policy's chosen recommendation.

Design: value-for-money = protection ÷ price.

    Within a scenario's quote set, min-max normalise each field:
        insured_norm  = norm(insured_amount)          # higher is better
        deductible_n  = norm(deductible)              # LOWER is better  -> use (1 - deductible_n)
        coverages_n   = norm(num_coverages)           # higher is better
        price_norm    = norm(price_annual)            # lower is better

        protection = w_cov·coverages_n + w_ins·insured_norm + w_ded·(1 - deductible_n)
        reward     = protection / (1 + w_price · price_norm)

Weights are configurable (defaults below). The recommended quote is argmax(reward).
Normalising WITHIN a scenario makes the reward comparable across scenarios with
very different absolute premiums / rebuild sums.

Note on inputs
--------------
`insured_amount` may be the literal "unbegrenzt" (unlimited) for some tariffs; it
is treated as the scenario maximum (best) for the insured-amount term.
`num_coverages` counts the optional add-on covers actually included (glass +
natural-hazard) — driven by what each building requested (fire + glass always;
natural-hazard/Elementar in flood-exposed postcodes), so it varies across scenarios
by hazard. Within one scenario the engine returns only tariffs matching the request,
so coverage is usually uniform there and the within-scenario differentiators are
price, deductible and `coverage_score` (the engine's 0-100 quality score, exposed as
an optional extra term via `w_score`, default 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class RewardWeights:
    w_cov: float = 0.4     # weight on number of coverages (protection)
    w_ins: float = 0.3     # weight on insured amount (protection)
    w_ded: float = 0.3     # weight on (low) deductible (protection)
    w_price: float = 1.0   # strength of the price penalty (denominator)
    w_score: float = 0.0   # optional weight on the engine coverage_score (protection); off by default

    def protection_denominator(self) -> float:
        return self.w_cov + self.w_ins + self.w_ded + self.w_score or 1.0


def _minmax(values: list[float], rel_floor: float = 0.01) -> list[float]:
    """Min-max to [0,1] within the scenario. A spread smaller than `rel_floor` of the
    field's magnitude is treated as NO signal (neutral 0.5) — otherwise near-constant
    fields (e.g. an insured sum that varies by a few hundred euros across tariffs, or
    all-identical coverage flags) get amplified into a full 0→1 swing and dominate the
    reward for no real reason. This keeps the value-for-money shape but ignores noise."""
    lo, hi = min(values), max(values)
    rng = hi - lo
    scale = max(abs(hi), abs(lo), 1.0)
    if rng <= rel_floor * scale:
        return [0.5 for _ in values]  # no meaningful spread → neutral
    return [(v - lo) / rng for v in values]


def _insured_value(q: dict[str, Any], scenario_max: float) -> float:
    v = q.get("insured_amount")
    if v == "unbegrenzt" or v is None:
        return scenario_max  # unlimited (or missing) → treat as best in scenario
    return float(v)


def score_quotes(quotes: list[dict[str, Any]], weights: RewardWeights | None = None) -> list[dict[str, Any]]:
    """Return quotes with an added `reward` and `reward_components`; sorted best-first."""
    w = weights or RewardWeights()
    if not quotes:
        return []

    numeric_insured = [float(q["insured_amount"]) for q in quotes if isinstance(q.get("insured_amount"), (int, float))]
    scenario_max_insured = max(numeric_insured) if numeric_insured else 1.0

    insured = [_insured_value(q, scenario_max_insured) for q in quotes]
    deductible = [float(q.get("deductible") or 0) for q in quotes]
    coverages = [float(q.get("num_coverages") or 0) for q in quotes]
    scores = [float(q.get("coverage_score") or 0) for q in quotes]
    price = [float(q["price_annual"]) for q in quotes]

    insured_n = _minmax(insured)
    deductible_n = _minmax(deductible)
    coverages_n = _minmax(coverages)
    score_n = _minmax(scores)
    price_n = _minmax(price)

    denom = w.protection_denominator()
    out: list[dict[str, Any]] = []
    for i, q in enumerate(quotes):
        protection = (
            w.w_cov * coverages_n[i]
            + w.w_ins * insured_n[i]
            + w.w_ded * (1.0 - deductible_n[i])
            + w.w_score * score_n[i]
        ) / denom
        reward = protection / (1.0 + w.w_price * price_n[i])
        out.append(
            {
                **q,
                "reward": round(reward, 4),
                "reward_components": {
                    "protection": round(protection, 4),
                    "coverages_n": round(coverages_n[i], 3),
                    "insured_n": round(insured_n[i], 3),
                    "deductible_n": round(deductible_n[i], 3),
                    "score_n": round(score_n[i], 3),
                    "price_n": round(price_n[i], 3),
                },
            }
        )
    out.sort(key=lambda x: x["reward"], reverse=True)
    return out


def recommend(quotes: list[dict[str, Any]], weights: RewardWeights | None = None) -> dict[str, Any] | None:
    """The single recommended quote = argmax(reward)."""
    scored = score_quotes(quotes, weights)
    return scored[0] if scored else None


def _main() -> None:
    ap = argparse.ArgumentParser(description="Score a scenario's quotes by value-for-money reward.")
    ap.add_argument("--quotes", help="JSON file with a `quotes` list (default: stdin).")
    for name in ("w_cov", "w_ins", "w_ded", "w_price", "w_score"):
        ap.add_argument(f"--{name}", type=float)
    args = ap.parse_args()
    payload = json.loads(open(args.quotes, encoding="utf-8").read() if args.quotes else sys.stdin.read())
    quotes = payload.get("quotes", payload) if isinstance(payload, dict) else payload
    overrides = {k: getattr(args, k) for k in ("w_cov", "w_ins", "w_ded", "w_price", "w_score") if getattr(args, k) is not None}
    scored = score_quotes(quotes, RewardWeights(**overrides))
    best = scored[0] if scored else None
    print(json.dumps({"recommended_tariff_id": best and best["tariff_id"], "scored": scored}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
