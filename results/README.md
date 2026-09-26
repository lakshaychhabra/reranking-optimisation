# Evaluation results

`summary.json` is the comparison index for all evaluated models. The original
frontier and Qwen3.5-9B files use the supplied evaluator schema. The three
Qwen3.5-2B files are byte-identical copies of the completed local evaluator
outputs imported from the RunPod archive:

- `qwen35_2b_base.jsonl`: unchanged `Qwen/Qwen3.5-2B`.
- `qwen35_2b_sft.jsonl`: selected SFT LoRA adapter.
- `qwen35_2b_grpo.jsonl`: optional ten-step GRPO continuation.

Each contains exactly 50 golden-scenario predictions. These are final
evaluation records and must never be used for training, checkpoint selection,
prompt tuning, or hyperparameter tuning. Their SHA-256 hashes are recorded in
`summary.json`.

The Qwen3.5-2B local-evaluator rows include `raw_output`, strict-format and
validity fields, timing, token counts, ranked model and teacher IDs, and the
already-computed per-scenario metrics. Aggregate values are copied from each
run's frozen `summary.json`, not recomputed to make training decisions.
