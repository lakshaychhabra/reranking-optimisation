# Qwen3.5-2B Training and Evaluation

For a fresh-clone evaluator workflow, start with `HOW_TO_RUN.md`; it uses the
tracked adapters and data under `release/`. This document additionally records
the original local/RunPod experiment workflow and its `artifacts/` paths.

This directory contains the model-training and experiment code added for the
take-home. Generated datasets, adapters, predictions, and TensorBoard logs live
under `artifacts/training/`; training code and documentation stay here.

The supplied `scripts/run_eval.py` remains intact. `training/run_eval.py` is the
local-checkpoint evaluator used to compare the same Qwen3.5-2B model at three
fixed stages: unchanged base, SFT adapter, and GRPO adapter.

## Files

- `prepare_sft_data.py`: validates the 300/90 SFT split and produces TRL
  conversational prompt-completion JSONL.
- `train_sft.py`: Unsloth BF16/FP16 LoRA SFT with validation loss, TensorBoard,
  environment capture, token-length audit, and a portable final adapter.
- `run_eval.py`: deterministic local inference and identical metrics for base,
  SFT, and GRPO checkpoints.
- `plot_run.py`: converts a saved run record into local loss and learning-rate
  plots.
- `common.py`: shared schema, hashing, JSONL, target, and parsing helpers.
- `requirements.txt`: portable GPU compatibility range.
- `requirements-runpod-lock.txt`: exact successful RTX A6000 user-space stack.
- `requirements-macos-eval.txt`: pinned Apple-Silicon inference-only stack.

## 1. Install on the GPU machine

Use a fresh Python environment with a CUDA-enabled PyTorch runtime, then:

```bash
python -m pip install -r training/requirements.txt
```

For the exact successful RTX A6000 environment, first install matching
`torch==2.8.0+cu128` and `torchvision==0.23.0+cu128`, then install
`training/requirements-runpod-lock.txt`. The lock captures Transformers 5.5.0,
TRL 0.24.0, PEFT 0.21.0, Datasets 4.3.0, and the other versions recorded by the
completed runs. Its causal-conv1d wheel is platform-specific; use the portable
requirements file when the CUDA/Python/ABI combination differs.

The intended machine has one 24 GB NVIDIA GPU. The scripts refuse to start the
actual model run without CUDA.

An RTX A6000 has 48 GB, so it is more than sufficient for this BF16-LoRA plan.

On this RunPod machine, the original `Qwen/Qwen3.5-2B` Hugging Face safetensors
checkpoint is available at `/root/wg-recommendation/artifacts/models/Qwen3.5-2B`.
It is the post-trained source model used for text-only LoRA SFT; it is not a
GGUF or inference-only quantization. A second copy is in the persistent
`/workspace/.cache/huggingface` cache. Read weights from the `/root` copy when
training because reads from that `/workspace` cache were very slow on this pod.
The exact Hub revision and weight checksum are recorded in `gpt.md`. The same
revision is cached locally under `/root/.cache/huggingface`. Set both cache
variables before using the default model identifier in the commands below:

```bash
export HF_HOME=/root/.cache/huggingface
export HF_HUB_CACHE=/root/.cache/huggingface/hub
```

### Qwen3.5 fast kernels on this pod

The RTX A6000 pod runs Python 3.12, PyTorch 2.8.0+cu128, and CUDA 12.8. The
`training/requirements.txt` includes Flash Linear Attention and causal-conv1d
without replacing PyTorch. The causal-conv1d wheel is built for this exact
Python/PyTorch/CUDA/Linux x86_64 combination. The equivalent manual commands
are:

```bash
python -m pip install 'flash-linear-attention[cuda]==0.5.2'
python -m pip install --no-deps \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
python -m pip install ninja==1.13.2
```

Start a new Python process after installing; one already running retains its
previously imported fallback kernels. Check the host's Python, PyTorch, CUDA,
and ABI before using the causal-conv1d wheel on another machine.

## Upload a clean bundle to RunPod

Create the allowlisted archive locally:

```bash
python training/create_runpod_bundle.py
```

This writes:

```text
artifacts/runpod/wg-training-runpod.tar.gz
artifacts/runpod/wg-training-runpod.tar.gz.sha256
artifacts/runpod/wg-training-runpod.tar.gz.manifest.json
```

The bundle includes code, documentation, prepared SFT data, SFT/GRPO splits,
the frozen reward, and the golden file needed for evaluation. It explicitly
excludes `.env`, credentials, live quote caches, teacher API requests/raw
responses, pilots, old datasets, checkpoints, and previous training runs.

Upload the archive into the pod's persistent `/workspace` directory using the
RunPod file browser or the SSH/SCP connection command shown by the pod. Then, in
the pod terminal:

```bash
cd /workspace
sha256sum -c wg-training-runpod.tar.gz.sha256
tar -xzf wg-training-runpod.tar.gz
cd ml-engineer-take-home-wg-recommendation
python -m pip install -r training/requirements.txt
nvidia-smi
```

Keep the repository and model outputs under `/workspace`; RunPod documents this
as the persistent-volume mount path. Do not upload the local `.env` file.

## 2. Prepare SFT data

```bash
python training/prepare_sft_data.py
```

Outputs:

```text
artifacts/training/sft_data/train.jsonl
artifacts/training/sft_data/validation.jsonl
artifacts/training/sft_data/manifest.json
```

The completion is deliberately ranking-only JSON, matching the benchmark
prompt. Teacher rationales remain in the source split but are not trained as
output tokens. The preparation step uses no golden records.

An offline validation-only run is also available:

```bash
python training/prepare_sft_data.py --validate-only
python training/train_sft.py --validate-only
```

## 3. Evaluate the unchanged base model

Run this once, before training. This is a locked reporting baseline, not a
source of hyperparameter feedback:

```bash
python training/run_eval.py \
  --stage base \
  --run-name golden-base-qwen35-2b \
  --confirm-golden-eval
```

Optionally add `--gpu-hourly-cost <USD_PER_HOUR>` to estimate self-hosted cost
per 1,000 recommendations from measured generation time.

## 4. Run the one-step SFT smoke test

This pod already has a verified smoke run at
`artifacts/training/runs/sft-qwen35-2b-smoke-v2/`. The earlier
`sft-qwen35-2b-smoke/` run used a zero learning rate and is preserved as an
incomplete parameter-update check. Choose a fresh run name if repeating the
smoke test. The raw SFT validation split needed by the `run_eval.py` reload
command below is currently absent on this pod.

```bash
python training/train_sft.py \
  --run-name sft-qwen35-2b-smoke-next \
  --smoke-test \
  --train-batch-size 1 \
  --eval-batch-size 1 \
  --gradient-accumulation-steps 1
```

Do not start the full run unless this completes, saves the adapter, and reports
finite train/evaluation losses. Verify that the saved adapter reloads by running
one validation example:

```bash
python training/run_eval.py \
  --stage sft \
  --adapter artifacts/training/runs/sft-qwen35-2b-smoke-next/final_adapter \
  --dataset artifacts/splits/sft_validation.jsonl \
  --limit 1 \
  --run-name sft-smoke-reload
```

## 5. Run full SFT

```bash
cd /root/wg-recommendation
python training/train_sft.py \
  --run-name sft-qwen35-2b-seed3407 \
  --train-batch-size 48 \
  --gradient-accumulation-steps 1
```

The completed configuration is three epochs, learning rate `2e-4`, LoRA rank
16, per-device and effective batch size 48, completion-only loss, and a
2,048-token ceiling. On 300 training rows it completed 21 optimizer steps.
Validation and
checkpoint saving occur after each epoch; at most two checkpoints are retained,
and the best validation checkpoint is loaded before writing `final_adapter`.
Before training, the tokenizer audit refuses to continue if any target would
be truncated.

Important outputs:

```text
artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter/
artifacts/training/runs/sft-qwen35-2b-seed3407/run_record.json
artifacts/training/runs/sft-qwen35-2b-seed3407/tensorboard/
```

Inspect live local metrics with TensorBoard:

```bash
tensorboard --logdir artifacts/training/runs/sft-qwen35-2b-seed3407/tensorboard
```

Create the static report figure:

```bash
python training/plot_run.py \
  artifacts/training/runs/sft-qwen35-2b-seed3407/run_record.json
```

## 6. Evaluate SFT validation before golden

Use the 90-example SFT validation split to diagnose the adapter:

```bash
python training/run_eval.py \
  --stage sft \
  --adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --dataset artifacts/splits/sft_validation.jsonl \
  --limit 90 \
  --run-name sft-validation-qwen35-2b
```

This reports Top-1, Top-3 recall, supplied/default reward ratio, strict-JSON
rate, valid-tariff rates, tokens, and generation speed.

## 7. Evaluate the fixed SFT adapter on golden

After the adapter is fixed using validation only:

```bash
python training/run_eval.py \
  --stage sft \
  --adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name golden-sft-qwen35-2b \
  --confirm-golden-eval
```

## 8. Optional ten-step GRPO continuation

The completed conservative GRPO run starts from the frozen SFT adapter:

```bash
python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-10step \
  --max-steps 10 \
  --save-steps 10 \
  --eval-steps 50 \
  --seed 3407
```

The same evaluator accepts the GRPO adapter, ensuring that base, SFT, and GRPO
use identical prompts, decoding, parsing, and metrics:

```bash
python training/run_eval.py \
  --stage grpo \
  --adapter artifacts/training/runs/grpo-qwen35-2b-10step/final_adapter \
  --run-name golden-grpo-qwen35-2b-10step-700tok \
  --confirm-golden-eval
```

The preserved GRPO checkpoint matched SFT Top-1 and supplied reward ratio on
golden, while Top-3 recall decreased from 0.866667 to 0.860000. SFT therefore
remains the selected delivery model; GRPO is retained as a reproducible
optional ablation.

## Evaluation integrity

- Prompts contain risk, risk context, and the allowed quote facts only.
- The prompt never contains reward, reward components, or teacher/Fable labels.
- Reward ratio uses the supplied per-quote `reward` stored in the evaluation
  dataset. Calibrated reward weights are not applied to headline evaluation.
- Invalid or missing predictions count as zero for all-scenario headline
  metrics; a valid-picks-only reward ratio is reported separately.
- Golden evaluation requires `--confirm-golden-eval` to reduce accidental
  repeated inspection. Never select epochs, learning rates, or checkpoints from
  golden results.
