# WG recommendation release

This directory is the tracked, portable release for the take-home submission.
It contains the selected SFT LoRA adapter, the optional ten-step GRPO adapter,
prepared non-golden training data, one development evaluation split, aggregate
evaluation summaries, and exact run configurations.

The base model is not duplicated. Both adapters use `Qwen/Qwen3.5-2B` at
revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; the repository's
`scripts/reproduce_model.py` downloads and verifies that base checkpoint.

Start with `HOW_TO_RUN.md` in the repository root. `MANIFEST.json` records the
source and SHA-256 of every released file, while `SHA256SUMS` supports an
independent integrity check.

No golden rows, per-scenario golden predictions, live quote cache, provider raw
responses, credentials, base-model weights, or optimizer checkpoints are in
this release. Aggregate golden summaries are included as final report evidence.
