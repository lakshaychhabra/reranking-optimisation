# GRPO Stage

This directory is the copyable GRPO code unit. It contains:

- `prepare_data.py`: converts the raw 810/90 GRPO splits into self-contained
  prompts with per-tariff scores from the frozen calibrated reward.
- `rewards.py`: four dependency-light rollout reward components.
- `train.py`: the GPU trainer, beginning from an existing SFT LoRA adapter.

On the GPU VM it additionally needs the shared `training/common.py` file and
the prepared directory:

```text
artifacts/training/grpo_data/
├── manifest.json
├── train.jsonl
└── validation.jsonl
```

The data is already prepared locally, so the raw splits, `src/`, `tools/`, and
frozen-weight file are not needed on the VM for training. They are needed only
if `prepare_data.py` is rerun.

After SFT completes, validate the upload and run the smoke gate:

```bash
python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --validate-only

python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-smoke \
  --smoke-test
```

Do not run full GRPO until the one-step and ten-step gates pass.

The conservative default is 50 optimizer steps, saving every 10 steps and
evaluating every 50 steps. Runs over 100 steps require `--allow-long-run`.
After confirming nonzero rollout reward variance in the one-step smoke run,
run the ten-step gate on the prepared training data:

```bash
python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-10step \
  --max-steps 10 \
  --save-steps 10 \
  --eval-steps 50
```

The trainer performs final validation after these 10 steps. Compare the saved
adapter on `artifacts/splits/grpo_validation.jsonl` before considering a longer
run. Keep the golden dataset out of this gate.
