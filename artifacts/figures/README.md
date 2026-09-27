# Story figures

Report-ready figures generated from frozen repository artifacts:

1. `01-dataset-funnel-and-splits` — collection funnel and leakage-safe split design.
2. `02-dataset-distribution` — portfolio coverage across six risk dimensions.
3. `03-reward-calibration` — supplied versus calibrated scorer on the calibration split.
4. `04-sft-training` — SFT train/validation loss and learning-rate schedule.
5. `05-grpo-training` — GRPO reward, KL, and output-stability diagnostics.
6. `06-model-progress` — base → SFT → GRPO golden metrics, with calibration kept separate.
7. `07-cost-quality` — price versus three quality metrics on a linear cost axis, using measured 2B serving costs.
8. `08-output-efficiency` — completion length and strict-JSON compliance.
9. `09-serving-efficiency` — concurrency versus throughput, latency, GPU utilization, and cost.

Regenerate both PNG and editable SVG versions with:

```bash
MPLCONFIGDIR=/tmp/wg-matplotlib python3 scripts/create_story_figures.py --seed 42
```

`resolved_config.json` records source hashes and the complete serving-cost calculation. The 2B
figures come from a reproducible RunPod RTX 4090 benchmark at $0.74/hour, including raw
request-level results and GPU telemetry under `artifacts/serving_benchmark/`. The recommended
planning assumption is 25% fleet utilization; saturated and low-traffic scenarios are retained
instead of collapsing utilization into a single context-free price.
