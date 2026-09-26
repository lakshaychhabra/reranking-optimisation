# How to run the WG recommendation model

This repository is self-contained for evaluation and training except for the
public Qwen base weights. The selected model is the SFT LoRA in
`release/model/sft-final/`. `release/model/grpo-10step/` is an optional ablation;
it did not improve the final headline result.

## 1. Clone the repository and fetch model files

The portable adapters are tracked with Git LFS:

```bash
git lfs install
git clone <PRIVATE_GITHUB_REPOSITORY_URL>
cd ml-engineer-take-home-wg-recommendation
git lfs pull
```

Verify the tracked release before loading a model:

```bash
python3 scripts/create_release_bundle.py --verify-only
python3 scripts/reproduce_model.py --action verify-imported
```

The second command must report these adapter directory hashes:

```text
sft:  f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48
grpo: 474543cab81b246c90b2b7c16bf79f714e7e4cdbedbba5c5826269106be25954
```

## 2. Create the GPU environment

The recorded environment used Linux, Python 3.12, an NVIDIA RTX A6000, CUDA
12.8, `torch==2.8.0+cu128`, and `torchvision==0.23.0+cu128`. Install the matching
CUDA PyTorch wheels first, then install the exact successful user-space stack:

```bash
python -m pip install -r training/requirements-runpod-lock.txt
python -m pip check
nvidia-smi
```

`training/requirements-runpod-lock.txt` includes a Linux/Python 3.12/PyTorch
2.8 causal-conv1d wheel. For another platform, use `training/requirements.txt`
and install a matching optional kernel wheel.

## 3. Evaluate the selected SFT model

`training/run_eval.py` downloads the public base model when it is not already in
the Hugging Face cache, loads the SFT LoRA, builds prompts through the canonical
`src/prompts.py`, performs deterministic generation with thinking disabled, and
writes `predictions.jsonl` plus `summary.json`.

Run the final held-out evaluation once:

```bash
python training/run_eval.py \
  --stage sft \
  --base-model Qwen/Qwen3.5-2B \
  --model-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --adapter release/model/sft-final \
  --dataset data/golden_wg_recommendations.jsonl \
  --limit 50 \
  --max-seq-length 2048 \
  --max-new-tokens 700 \
  --output-root artifacts/reproduction/evaluations \
  --run-name golden-sft-reproduction \
  --confirm-golden-eval
```

Expected headline results on the supplied 50-scenario golden file:

| Metric | SFT result |
|---|---:|
| Top-1 accuracy | 0.84 (42/50) |
| Top-3 recall | 0.866667 |
| Strict JSON rate | 1.0 |
| Valid three-tariff rate | 1.0 |
| Supplied reward ratio | 0.811724 |

The saved reference summary is `release/evaluation/sft-golden-summary.json`.
Do not use the golden result to select a checkpoint or tune a setting.

The exact frozen 50-row outputs used for the final comparison are also tracked
under `results/`:

```text
results/qwen35_2b_base.jsonl
results/qwen35_2b_sft.jsonl
results/qwen35_2b_grpo.jsonl
results/summary.json
```

Use `results/qwen35_2b_sft.jsonl` as the selected-model reference when checking
a reproduced run. `results/summary.json` records the expected file hash and
aggregate metrics for all three stages.

## 4. Development evaluation without touching golden

For a routine installation check, use the released non-golden development
split:

```bash
python training/run_eval.py \
  --stage sft \
  --base-model Qwen/Qwen3.5-2B \
  --model-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --adapter release/model/sft-final \
  --dataset release/data/eval/grpo_validation.jsonl \
  --limit 90 \
  --max-seq-length 2048 \
  --max-new-tokens 700 \
  --output-root artifacts/reproduction/evaluations \
  --run-name sft-development-reproduction
```

Or use the orchestration wrapper, which evaluates both released adapters on the
same development split:

```bash
python scripts/reproduce_model.py \
  --action evaluate-imported \
  --seed 3407 \
  --execute
```

## 5. Evaluate the optional GRPO adapter

```bash
python training/run_eval.py \
  --stage grpo \
  --base-model Qwen/Qwen3.5-2B \
  --model-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --adapter release/model/grpo-10step \
  --dataset release/data/eval/grpo_validation.jsonl \
  --limit 90 \
  --max-seq-length 2048 \
  --max-new-tokens 700 \
  --output-root artifacts/reproduction/evaluations \
  --run-name grpo-development-reproduction
```

On golden, the recorded GRPO adapter also reached 0.84 Top-1 and 0.811724
reward ratio, but Top-3 recall decreased to 0.86. It is therefore preserved for
reproducibility but is not the selected delivery model.

## 6. Reproduce training from the released data

Preview every command without downloading weights or starting a GPU job:

```bash
python scripts/reproduce_model.py \
  --action sft-grpo \
  --seed 3407
```

Reproduce SFT only:

```bash
python scripts/reproduce_model.py \
  --action sft \
  --seed 3407 \
  --execute
```

Reproduce SFT followed by the optional ten-step GRPO continuation:

```bash
python scripts/reproduce_model.py \
  --action sft-grpo \
  --seed 3407 \
  --execute
```

The wrapper downloads and verifies the pinned base weights, trains into
`artifacts/reproduction/runs/`, and evaluates into
`artifacts/reproduction/evaluations/`. It uses only `release/data/` for
training and development evaluation. Golden evaluation is disabled unless
`--confirm-golden-eval` is explicitly supplied.

The recorded SFT run used three epochs, BF16, completion-only loss, LoRA rank
and alpha 16, learning rate `2e-4`, per-device batch 48, and seed 3407. The
optional GRPO continuation used ten optimizer steps, four generations per
prompt, learning rate `5e-6`, and KL coefficient `0.001`.

## 7. Validate data and commands without a GPU

```bash
python training/train_sft.py \
  --train-data release/data/sft/train.jsonl \
  --validation-data release/data/sft/validation.jsonl \
  --validate-only

python training/grpo/train.py \
  --sft-adapter release/model/sft-final \
  --train-data release/data/grpo/train.jsonl \
  --validation-data release/data/grpo/validation.jsonl \
  --manifest release/data/grpo/manifest.json \
  --validate-only

python training/run_eval.py \
  --stage sft \
  --adapter release/model/sft-final \
  --dataset release/data/eval/grpo_validation.jsonl \
  --limit 90 \
  --run-name preflight-only \
  --validate-only
```

## 8. Apple Silicon evaluation

The same evaluator has a portable Transformers backend. The selected SFT model
was verified on an Apple M4: all 50 exact raw outputs and substantive
per-scenario fields matched the frozen CUDA result. Only generation timing
differed. Full evidence is in `documentation/macos-verification.md`.

Create or reuse a Conda environment and install the pinned inference-only
stack:

```bash
conda create -n work python=3.11
conda run -n work python -m pip install -r training/requirements-macos-eval.txt
```

If the base model is not already local, the reproduction wrapper downloads and
verifies the pinned public snapshot before evaluating both released adapters:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n work \
  python scripts/reproduce_model.py \
  --action evaluate-imported \
  --seed 3407 \
  --execute
```

For a direct final SFT evaluation, use the command in section 3 and add
`--backend transformers`. On CUDA, the default `--backend auto` continues to
select the as-run Unsloth backend; on Apple Silicon it selects Transformers.

## Output format

The evaluator writes one prediction per scenario. Model text is expected to be
strict JSON with three distinct offered tariffs:

```json
{"top_3":[{"rank":1,"tariff_id":"49"},{"rank":2,"tariff_id":"48"},{"rank":3,"tariff_id":"162"}]}
```

The canonical serializer and parser live in `src/prompts.py` and
`training/common.py`. Training and evaluation do not construct a second prompt
format.
