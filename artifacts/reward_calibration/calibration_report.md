# Reward Calibration Report

## Outcome

A deterministic search over 20,000 candidate reward configurations
selected candidate `15957` using only 200 silver-labeled
calibration scenarios. The supplied golden benchmark and GRPO validation split were not used.

| Configuration | Teacher Top-1 | Teacher Top-3 recall | Pairwise accuracy |
| --- | --- | --- | --- |
| Supplied default | 49.00% | 68.83% | 76.07% |
| Selected (in-sample) | 94.00% | 84.17% | 92.24% |
| Nested 5-fold selection | 94.00% | 84.17% | 92.24% |

The nested-fold row is the honest estimate of the weight-selection procedure:
each fold was scored with weights selected using only the other four folds.
The final weights were then selected from all 200 calibration scenarios and frozen.

## Frozen weights

| Parameter | Value |
| --- | --- |
| w_cov | 0.09280341 |
| w_ins | 0.41027102 |
| w_ded | 0.06960256 |
| w_score | 0.42732301 |
| w_price | 1.79361163 |

## Dataset evidence

| Measure | Value |
| --- | --- |
| Scenarios | 200 |
| Quotes | 1311 |
| Mean quotes per scenario | 6.555 |
| Minimum / maximum quotes | 4 / 9 |

### Within-scenario signal availability

| Quote field | Scenarios with variation | Share |
| --- | --- | --- |
| coverage_score | 200 | 100.00% |
| deductible | 0 | 0.00% |
| insured_amount | 168 | 84.00% |
| num_coverages | 0 | 0.00% |
| price_annual | 200 | 100.00% |

Because `num_coverages` and `deductible` never vary within a quote panel,
their individual weights are not identifiable. The search therefore calibrates
their combined static contribution and preserves their default 4:3 ratio.

## Five-fold stability

| Held-out fold | Selected candidate | Top-1 | Top-3 recall | Pairwise | w_score | w_price |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 15957 | 92.50% | 84.17% | 92.02% | 0.4273 | 1.7936 |
| 1 | 15957 | 95.00% | 82.50% | 91.26% | 0.4273 | 1.7936 |
| 2 | 15957 | 97.50% | 83.33% | 92.69% | 0.4273 | 1.7936 |
| 3 | 15957 | 92.50% | 82.50% | 91.43% | 0.4273 | 1.7936 |
| 4 | 15957 | 92.50% | 88.33% | 93.89% | 0.4273 | 1.7936 |

Nested held-out Top-1 95% Wilson interval: 89.81%
to 96.53%.

## Search method

- Protection weights were sampled over a three-part simplex: static baseline, insured amount, and coverage score.
- The price penalty was sampled log-uniformly from 0.05 to 5.0, with explicit zero-price and hand-designed anchor candidates.
- Candidates were ranked lexicographically by Top-1 agreement, Top-3 recall, and pairwise accuracy.
- Distance from the supplied default reward was used only to break otherwise identical results.
- Reward values were rounded to four decimals before ranking, matching `tools/reward.py`.
- The fast calibration scorer was checked against `tools.reward.score_quotes` on all 200 default-weight scenarios.

## Interpretation

These weights approximate the GPT-6 Sol silver teacher; they are not independent ground truth.
Do not adjust them using the golden benchmark. During GRPO, keep them fixed and combine
the ranking reward with structural penalties for invalid JSON, duplicate IDs, or IDs absent
from the scenario's quote panel.
