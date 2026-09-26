# Quoted Dataset Distribution

Generated: `2026-09-25T20:17:10.992851+00:00`

## Dataset overview

| Scenarios | Quotes | Insurers | Insurer-products | Tariff IDs |
|---|---|---|---|---|
| 1490 | 9632 | 5 | 9 | 9 |

## Collection funnel

| Attempts | Successes | Success rate | No quotes | Insufficient | Errors |
|---|---|---|---|---|---|
| 2000 | 1490 | 74.50% | 178 | 332 | 0 |

## Hazard tier

| Hazard tier | Count | Share |
|---|---|---|
| moderate | 578 | 38.79% |
| elevated | 458 | 30.74% |
| high | 454 | 30.47% |

## Age stratum

| Age stratum | Count | Share |
|---|---|---|
| middle | 502 | 33.69% |
| modern | 497 | 33.36% |
| old | 491 | 32.95% |

## Building type

| Building type | Count | Share |
|---|---|---|
| single-family | 346 | 23.22% |
| terraced | 313 | 21.01% |
| two-family | 294 | 19.73% |
| semi-detached | 286 | 19.19% |
| multi-family | 251 | 16.85% |

## Construction class

| Construction class | Count | Share |
|---|---|---|
| BAK1 | 389 | 26.11% |
| BAK2 | 388 | 26.04% |
| FHG1 | 362 | 24.30% |
| FHG2 | 351 | 23.56% |

## Quote and reward statistics

| Metric | Min | Median | Mean | Max |
|---|---|---|---|---|
| Quotes per scenario | 3.0 | 6.0 | 6.4644 | 9.0 |
| Annual price - all quotes | 42.17 | 1009.895 | 1131.2825 | 7161.93 |
| Coverage score - all quotes | 25.0 | 56.0 | 56.3054 | 79.0 |
| Top-1 vs top-2 reward margin | 0.0001 | 0.0833 | 0.0815 | 0.2491 |

## Reward-signal variation within scenarios

| Field | Varying scenarios | Flat scenarios | Varying share |
|---|---|---|---|
| price_annual | 1490 | 0 | 100.00% |
| insured_amount | 1220 | 270 | 81.88% |
| deductible | 0 | 1490 | 0.00% |
| num_coverages | 0 | 1490 | 0.00% |
| coverage_score | 1490 | 0 | 100.00% |

## Reward-recommended insurers

| Insurer | Count | Share |
|---|---|---|
| Baloise | 601 | 40.34% |
| Gothaer | 458 | 30.74% |
| NV-Versicherung | 334 | 22.42% |
| VdVA | 97 | 6.51% |

## Interpretation notes

- The quoted JSONL contains successful scenarios only; the collection funnel comes from cache records.
- Flat within-scenario fields cannot distinguish tariffs under a simple ranking reward.
- Subgroup success rates based on small counts are descriptive, not statistically conclusive.
- The hazard context is a supplied heuristic, not authoritative ZURS/GDV hazard data.
