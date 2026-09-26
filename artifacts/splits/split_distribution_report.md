# Training Split Distribution Report

This report documents the deterministic hierarchical split of the 1,490
synthetic, teacher-annotated scenarios. The supplied 50-scenario golden
dataset is excluded and remains reserved for final evaluation.

## Split sizes

| Final split | Records | Purpose |
| --- | --- | --- |
| Reward calibration | 200 | Reward-weight selection |
| SFT train | 300 | Supervised fine-tuning |
| SFT validation | 90 | SFT checkpoint selection |
| GRPO train | 810 | Policy optimization |
| GRPO validation | 90 | GRPO checkpoint monitoring/selection |

## Stratification quality

Values are the largest absolute percentage-point difference across all
balanced category marginals. Stage-one pools are compared with the full
source; final train/validation splits are compared with their parent pool.

| Pool or split | Maximum deviation (percentage points) |
| --- | --- |
| Reward calibration | 0.7215 |
| SFT pool | 0.3803 |
| GRPO pool | 0.2229 |
| SFT train | 0.4103 |
| SFT validation | 1.3675 |
| GRPO train | 0.1728 |
| GRPO validation | 1.5556 |

## Stage-one pool distributions

Each cell is `count (share within dataset/pool)`.

### hazard_tier

| Field | Category | Full source | Reward calibration | SFT pool | GRPO pool |
| --- | --- | --- | --- | --- | --- |
| hazard_tier | elevated | 458 (30.74%) | 62 (31.00%) | 119 (30.51%) | 277 (30.78%) |
| hazard_tier | high | 454 (30.47%) | 61 (30.50%) | 119 (30.51%) | 274 (30.44%) |
| hazard_tier | moderate | 578 (38.79%) | 77 (38.50%) | 152 (38.97%) | 349 (38.78%) |

### age_stratum

| Field | Category | Full source | Reward calibration | SFT pool | GRPO pool |
| --- | --- | --- | --- | --- | --- |
| age_stratum | middle | 502 (33.69%) | 67 (33.50%) | 131 (33.59%) | 304 (33.78%) |
| age_stratum | modern | 497 (33.36%) | 67 (33.50%) | 129 (33.08%) | 301 (33.44%) |
| age_stratum | old | 491 (32.95%) | 66 (33.00%) | 130 (33.33%) | 295 (32.78%) |

### building_type

| Field | Category | Full source | Reward calibration | SFT pool | GRPO pool |
| --- | --- | --- | --- | --- | --- |
| building_type | multi-family | 251 (16.85%) | 34 (17.00%) | 66 (16.92%) | 151 (16.78%) |
| building_type | semi-detached | 286 (19.19%) | 38 (19.00%) | 75 (19.23%) | 173 (19.22%) |
| building_type | single-family | 346 (23.22%) | 45 (22.50%) | 90 (23.08%) | 211 (23.44%) |
| building_type | terraced | 313 (21.01%) | 43 (21.50%) | 82 (21.03%) | 188 (20.89%) |
| building_type | two-family | 294 (19.73%) | 40 (20.00%) | 77 (19.74%) | 177 (19.67%) |

### construction_class

| Field | Category | Full source | Reward calibration | SFT pool | GRPO pool |
| --- | --- | --- | --- | --- | --- |
| construction_class | BAK1 | 389 (26.11%) | 52 (26.00%) | 103 (26.41%) | 234 (26.00%) |
| construction_class | BAK2 | 388 (26.04%) | 52 (26.00%) | 102 (26.15%) | 234 (26.00%) |
| construction_class | FHG1 | 362 (24.30%) | 49 (24.50%) | 94 (24.10%) | 219 (24.33%) |
| construction_class | FHG2 | 351 (23.56%) | 47 (23.50%) | 91 (23.33%) | 213 (23.67%) |

### natural_hazard_requested

| Field | Category | Full source | Reward calibration | SFT pool | GRPO pool |
| --- | --- | --- | --- | --- | --- |
| natural_hazard_requested | False | 578 (38.79%) | 77 (38.50%) | 152 (38.97%) | 349 (38.78%) |
| natural_hazard_requested | True | 912 (61.21%) | 123 (61.50%) | 238 (61.03%) | 551 (61.22%) |

### owner_occupied

| Field | Category | Full source | Reward calibration | SFT pool | GRPO pool |
| --- | --- | --- | --- | --- | --- |
| owner_occupied | False | 436 (29.26%) | 58 (29.00%) | 114 (29.23%) | 264 (29.33%) |
| owner_occupied | True | 1054 (70.74%) | 142 (71.00%) | 276 (70.77%) | 636 (70.67%) |

## Final split distributions

The SFT and GRPO pools were independently stratified a second time.
Each cell is `count (share within split)`.

### hazard_tier

| Field | Category | Reward calibration | SFT train | SFT validation | GRPO train | GRPO validation |
| --- | --- | --- | --- | --- | --- | --- |
| hazard_tier | elevated | 62 (31.00%) | 91 (30.33%) | 28 (31.11%) | 250 (30.86%) | 27 (30.00%) |
| hazard_tier | high | 61 (30.50%) | 91 (30.33%) | 28 (31.11%) | 246 (30.37%) | 28 (31.11%) |
| hazard_tier | moderate | 77 (38.50%) | 118 (39.33%) | 34 (37.78%) | 314 (38.77%) | 35 (38.89%) |

### age_stratum

| Field | Category | Reward calibration | SFT train | SFT validation | GRPO train | GRPO validation |
| --- | --- | --- | --- | --- | --- | --- |
| age_stratum | middle | 67 (33.50%) | 102 (34.00%) | 29 (32.22%) | 274 (33.83%) | 30 (33.33%) |
| age_stratum | modern | 67 (33.50%) | 99 (33.00%) | 30 (33.33%) | 271 (33.46%) | 30 (33.33%) |
| age_stratum | old | 66 (33.00%) | 99 (33.00%) | 31 (34.44%) | 265 (32.72%) | 30 (33.33%) |

### building_type

| Field | Category | Reward calibration | SFT train | SFT validation | GRPO train | GRPO validation |
| --- | --- | --- | --- | --- | --- | --- |
| building_type | multi-family | 34 (17.00%) | 50 (16.67%) | 16 (17.78%) | 136 (16.79%) | 15 (16.67%) |
| building_type | semi-detached | 38 (19.00%) | 58 (19.33%) | 17 (18.89%) | 155 (19.14%) | 18 (20.00%) |
| building_type | single-family | 45 (22.50%) | 70 (23.33%) | 20 (22.22%) | 191 (23.58%) | 20 (22.22%) |
| building_type | terraced | 43 (21.50%) | 63 (21.00%) | 19 (21.11%) | 169 (20.86%) | 19 (21.11%) |
| building_type | two-family | 40 (20.00%) | 59 (19.67%) | 18 (20.00%) | 159 (19.63%) | 18 (20.00%) |

### construction_class

| Field | Category | Reward calibration | SFT train | SFT validation | GRPO train | GRPO validation |
| --- | --- | --- | --- | --- | --- | --- |
| construction_class | BAK1 | 52 (26.00%) | 80 (26.67%) | 23 (25.56%) | 210 (25.93%) | 24 (26.67%) |
| construction_class | BAK2 | 52 (26.00%) | 78 (26.00%) | 24 (26.67%) | 212 (26.17%) | 22 (24.44%) |
| construction_class | FHG1 | 49 (24.50%) | 73 (24.33%) | 21 (23.33%) | 197 (24.32%) | 22 (24.44%) |
| construction_class | FHG2 | 47 (23.50%) | 69 (23.00%) | 22 (24.44%) | 191 (23.58%) | 22 (24.44%) |

### natural_hazard_requested

| Field | Category | Reward calibration | SFT train | SFT validation | GRPO train | GRPO validation |
| --- | --- | --- | --- | --- | --- | --- |
| natural_hazard_requested | False | 77 (38.50%) | 118 (39.33%) | 34 (37.78%) | 314 (38.77%) | 35 (38.89%) |
| natural_hazard_requested | True | 123 (61.50%) | 182 (60.67%) | 56 (62.22%) | 496 (61.23%) | 55 (61.11%) |

### owner_occupied

| Field | Category | Reward calibration | SFT train | SFT validation | GRPO train | GRPO validation |
| --- | --- | --- | --- | --- | --- | --- |
| owner_occupied | False | 58 (29.00%) | 87 (29.00%) | 27 (30.00%) | 238 (29.38%) | 26 (28.89%) |
| owner_occupied | True | 142 (71.00%) | 213 (71.00%) | 63 (70.00%) | 572 (70.62%) | 64 (71.11%) |

## Integrity checks

- All 1,490 source scenarios are assigned exactly once.
- The five final splits are mutually disjoint and exhaustive.
- Quote and teacher-annotation IDs match one-to-one.
- No synthetic risk exactly overlaps a supplied golden risk.
- GRPO training records omit teacher annotations; GRPO validation retains them only for offline metrics.
