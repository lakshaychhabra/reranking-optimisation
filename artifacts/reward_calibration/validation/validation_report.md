# Frozen Reward Validation Report

This report evaluates the already-frozen reward. No result in this report
was used to change or reselect its weights.

## 1. Permutation negative control

Teacher rankings were replaced by deterministic random rankings drawn from
each scenario's valid tariff IDs. The full 20,000-candidate, five-fold
selection process was then repeated.

| Control | Top-1 | Top-3 recall |
| --- | --- | --- |
| Theoretical random expectation | 16.31% | 48.92% |
| Best candidate on all shuffled labels (optimistic) | 22.50% | 46.67% |
| Nested held-out selection | 20.00% | 48.33% |

Control result: **PASS**. nested held-out Top-1 <= theoretical random Top-1 + 0.05.

## 2. Held-out silver subgroup validation

| Reward | Top-1 | Top-3 recall | Pairwise |
| --- | --- | --- | --- |
| Supplied default | 46.67% | 68.52% | 73.96% |
| Frozen calibrated reward | 93.33% | 82.96% | 91.38% |

### Subgroups

| Dimension | Category | N | Default Top-1 | Frozen Top-1 | Frozen Top-3 recall |
| --- | --- | --- | --- | --- | --- |
| age_stratum | middle | 30 | 66.67% | 96.67% | 81.11% |
| age_stratum | modern | 30 | 53.33% | 100.00% | 80.00% |
| age_stratum | old | 30 | 20.00% | 83.33% | 87.78% |
| building_type | multi-family | 15 | 33.33% | 100.00% | 84.44% |
| building_type | semi-detached | 18 | 50.00% | 100.00% | 90.74% |
| building_type | single-family | 20 | 55.00% | 90.00% | 76.67% |
| building_type | terraced | 19 | 52.63% | 94.74% | 80.70% |
| building_type | two-family | 18 | 38.89% | 83.33% | 83.33% |
| construction_class | bak1 | 24 | 33.33% | 95.83% | 83.33% |
| construction_class | bak2 | 22 | 18.18% | 95.45% | 86.36% |
| construction_class | fhg1 | 22 | 63.64% | 86.36% | 83.33% |
| construction_class | fhg2 | 22 | 72.73% | 95.45% | 78.79% |
| hazard_tier | elevated | 27 | 33.33% | 92.59% | 79.01% |
| hazard_tier | high | 28 | 57.14% | 92.86% | 80.95% |
| hazard_tier | moderate | 35 | 48.57% | 94.29% | 87.62% |
| natural_hazard_requested | false | 35 | 48.57% | 94.29% | 87.62% |
| natural_hazard_requested | true | 55 | 45.45% | 92.73% | 80.00% |
| owner_occupied | false | 26 | 30.77% | 92.31% | 85.90% |
| owner_occupied | true | 64 | 53.12% | 93.75% | 81.77% |
| quote_count_bucket | 4-5 | 18 | 16.67% | 83.33% | 92.59% |
| quote_count_bucket | 6-7 | 49 | 71.43% | 93.88% | 82.31% |
| quote_count_bucket | 8-9 | 23 | 17.39% | 100.00% | 76.81% |

## 3. Locked Fable golden evaluation

The frozen weights were applied once to the 50 supplied golden scenarios.
The calibration and validation procedure did not use these labels.

| Reward | Fable Top-1 | Fable Top-3 recall | Pairwise | Supplied reward ratio |
| --- | --- | --- | --- | --- |
| Supplied default | 38.00% | 65.33% | 66.02% | 100.00% |
| Frozen calibrated reward | 84.00% | 79.33% | 90.41% | 81.17% |

The golden result is a deterministic non-model baseline. It must not be used
to revise the frozen weights; future SFT and GRPO models should be compared
against it under the same `run_eval.py` definitions.
