# Imported GPU training artifacts

Source archive: `tmp/wg-recommendation-full-20260926.tar.gz`

- Archive SHA-256: `76da6f8a013fc0280e6a3a12b07c16f34652cbac0959e84a858c2fc26fa87c1e`
- Imported SFT run: `runs/sft-qwen35-2b-seed3407/` (328 MB)
- Imported GRPO run: `runs/grpo-qwen35-2b-10step/` (184 MB)
- Per-file inventory: `important_checkpoints.sha256` (59 files)
- Full remote journal: `../../documentation/runpod_training_journal.md`

## SFT lineage

- `checkpoints/checkpoint-14/`: resumable epoch checkpoint
- `checkpoints/checkpoint-21/`: resumable best/final epoch checkpoint
- `final_adapter/`: portable BF16 LoRA used for reported SFT evaluation
- Final-adapter directory SHA-256: `f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48`

## GRPO lineage

- `checkpoints/checkpoint-10/`: resumable ten-step checkpoint
- `final_adapter/`: portable LoRA produced by the optional GRPO continuation
- Final-adapter directory SHA-256: `474543cab81b246c90b2b7c16bf79f714e7e4cdbedbba5c5826269106be25954`

The archive's 4.3 GB copy of the Hugging Face base model was intentionally not
duplicated. `scripts/reproduce_model.py` downloads immutable revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` and verifies the base weight SHA-256
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.

Smoke-test and batch-sizing checkpoints remain in the original archive but are
not promoted into the important-checkpoint handoff.
