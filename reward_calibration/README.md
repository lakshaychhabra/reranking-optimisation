# Reward calibration

This module selects fixed `RewardWeights` using only the 200-scenario reward
calibration split. It does not call an LLM, train a model, or inspect the
supplied golden benchmark.

Run from the repository root:

```bash
python reward_calibration/calibrate.py
```

The search is deterministic under seed `20260926`. It evaluates 20,000 weight
configurations and uses five stratified folds to estimate how well the
selection procedure generalizes. The final weights are selected on all 200
calibration scenarios only after the fold-level stability check.

Generated evidence is written to `artifacts/reward_calibration/`:

- `best_weights.json`: frozen weights and headline metrics;
- `calibration_summary.json`: full machine-readable experiment summary;
- `candidate_results.jsonl`: every evaluated candidate, best first; and
- `calibration_report.md`: report-ready methodology and result tables.

The calibration objective is lexicographic: teacher Top-1 agreement first,
teacher Top-3 recall second, and teacher pairwise-ranking accuracy third.
Distance from the supplied default is used only as a final tie-breaker.

`num_coverages` and `deductible` are constant within every calibration quote
panel. Their weights therefore cannot be identified separately. The search
uses a combined static-protection weight and maps it back to `w_cov:w_ded` in
the supplied default ratio of `4:3`.

## Completed run

The seed-`20260926` run evaluated 20,000 candidates and selected candidate
`15957`. Teacher Top-1 agreement increased from 49% under the supplied default
reward to 94%. Nested five-fold selection also achieved 94% aggregated held-out
Top-1; every four-fold training subset independently selected the same
candidate. See `artifacts/reward_calibration/calibration_report.md` for the full
results and caveats.

## Frozen validation

After freezing, run the falsification and transfer checks with:

```bash
python reward_calibration/validate_frozen.py
```

This repeats the nested search against randomly permuted labels, evaluates the
frozen reward by subgroup on the 90-example GRPO validation set, and performs
the locked deterministic baseline evaluation on the 50 supplied Fable
scenarios. It never writes to or reselects `best_weights.json`. Evidence is
stored under `artifacts/reward_calibration/validation/`.
