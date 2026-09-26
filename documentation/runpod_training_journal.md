# Experiment journal and decision log

## 2026-09-26 — Initial audit (completed)

### Objective and scope

Prepare the existing Qwen3.5-2B LoRA SFT pipeline for an RTX A6000 48 GB RunPod machine. The task currently covers repository, data, environment, and dependency auditing. Full SFT is pending explicit approval. No prepared dataset was regenerated or changed.

### Repository and data

Present: `training/{README.md,requirements.txt,common.py,create_runpod_bundle.py,plot_run.py,prepare_sft_data.py,run_eval.py,train_sft.py,__init__.py}` and `artifacts/training/sft_data/{manifest.json,train.jsonl,validation.jsonl}`. The directory is not a Git checkout on this machine.

The prepared files contain 300 training and 90 validation records. Each record has `scenario_id`, `prompt`, and `completion`. `prompt` is a two-message list (`system`, `user`), and `completion` is a one-message list (`assistant`); all message contents are strings. The target is JSON with exactly three ranked tariff IDs under `top_3`. All 390 sampled-by-parser completions parse as JSON, and both local file SHA-256 hashes match the manifest. The manifest's source and output paths refer to a different machine; local hashes identify the copied prepared files. The prompt and target are already formatted, so SFT training does not need `src/prompts.py` or original scenarios.

### Script and dependency decisions

`train_sft.py` uses prepared JSONL, `datasets`, Unsloth, TRL `SFTTrainer`/`SFTConfig`, PyTorch, and TensorBoard. It defaults to 16-bit LoRA (`load_in_4bit=False`), sequence length 2048, rank 16, seven projection targets, 3 epochs, batch size 2 with accumulation 8, and completion-only loss. `run_eval.py` imports `src.prompts` and defaults to `data/golden_wg_recommendations.jsonl`; those are separate evaluation inputs and currently missing. `prepare_sft_data.py` also imports `src.prompts`, but it is unnecessary because the prepared data exists. `plot_run.py` requires Matplotlib only when invoked.

The current `requirements.txt` pins `unsloth==2026.9.11` and lists TensorBoard, Matplotlib, Pillow, and torchvision. Its suitability and transitive dependencies remain under audit; no package changes or installation have been made.

### Commands and measured results

Read the requested source files and manifest; inspected Python imports; parsed both prepared JSONL files; checked local SHA-256 hashes against manifest. Counts: 300/90. Hash checks: both passed. JSON parsing: 300/300 training and 90/90 validation targets passed. No model, optimizer, or evaluation run has been started.

### Planned work

Audit OS, Python, GPU, CUDA, installed package versions, and `pip check`; then finish the requirements audit and record any missing dependencies or conflicts. Subsequent model access and smoke tests follow only after the initial audit.

## 2026-09-26 — Environment and requirements audit (completed)

### Measured environment

| Item | Result |
| --- | --- |
| OS | Ubuntu 24.04.3 LTS |
| Python | `/usr/local/bin/python`, 3.12.3 |
| pip | 25.2 |
| GPU | NVIDIA RTX A6000, 49,140 MiB reported by `nvidia-smi` (48 GiB class) |
| Driver | 595.91.07 |
| `nvidia-smi` CUDA | 13.2 (driver capability) |
| PyTorch | 2.8.0+cu128; CUDA build 12.8 |
| CUDA available | true |
| Compute capability | 8.6 |
| BF16 supported | true |
| torchvision | 0.23.0+cu128 |
| Pillow | 11.0.0 |

Transformers, Datasets, Accelerate, PEFT, TRL, bitsandbytes, Unsloth, Flash Attention, TensorBoard, Matplotlib, Hugging Face Hub, safetensors, and tokenizers were not installed at audit time. `python -m pip check` returned `No broken requirements found` for the *current* environment; it does not validate the uninstalled SFT stack. pip warned that `/workspace/.cache/pip` is not writable and disabled its cache. No package was installed.

### Requirements decision

`training/requirements.txt` now lists Unsloth 2026.9.11, Transformers 5.2–5.5, Datasets 3.4.1–<4.4, Accelerate, PEFT, TRL <=0.24, bitsandbytes, xformers 0.0.32.post2, TensorBoard, and Matplotlib. Unsloth supplies the model loader and LoRA wrapper; Transformers supplies Qwen3.5 architecture and chat templating; Datasets builds the SFT datasets; Accelerate/PEFT/TRL support adapter training and `SFTTrainer`; bitsandbytes and xformers are Unsloth runtime dependencies (the current script uses 16-bit LoRA, not 4-bit quantization); TensorBoard stores local trainer logs; Matplotlib renders local plots. Unsloth's dependency metadata also pulls tokenizer and model file libraries. Pillow and torchvision were removed as explicit top-level requirements because the text-only scripts do not import them; Unsloth pulls torchvision transitively and the pod already has a matching CUDA build. PyTorch is deliberately not pinned in this generic file.

A read-only pip dry run with `--report /tmp/wg-recommendation-pip-plan.json` confirmed `unsloth==2026.9.11` is available, but it began downloading wheels and was interrupted before dependency resolution finished. During that partial resolution, pip considered xformers 0.0.35 and newer CUDA/PyTorch dependencies. xformers 0.0.32.post2 was then pinned because its release is built for PyTorch 2.8. A complete post-change resolver check and import verification remain pending. Preserve torch 2.8.0+cu128 and torchvision 0.23.0+cu128 during installation; inspect the resolver plan before accepting any replacement.

### Commands executed

- `cat /etc/os-release`; `command -v python`; `python --version`; `python -m pip --version`; `nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader`; `nvidia-smi`.
- Python `importlib.metadata` / `find_spec` audit and `torch.cuda` query; `python -m pip check`.
- `python -m pip install --dry-run --report /tmp/wg-recommendation-pip-plan.json -r training/requirements.txt` (interrupted during downloads; no packages installed).

### Failed or incomplete checks

The package resolver has not completed. Imports for the SFT dependencies cannot pass until those packages are installed. No smoke test or training run has started; no training/evaluation configuration or measured loss exists yet. The planned smoke outputs will be kept separate from later full SFT outputs.

### Result locations and next steps

Prepared data: `artifacts/training/sft_data/`. Planned local run root: `artifacts/training/runs/`; actual runs have not been created. Next: finish dependency resolution with the existing PyTorch and torchvision preserved, install only after showing the proposed command, verify imports, confirm Hugging Face access without exposing `Hugging Face token environment variable`, then run minimal tokenizer/model and one-step adapter smoke tests. Full SFT needs explicit user approval.

### Missing local inputs found during import audit

- `/root/wg-recommendation/src/prompts.py` is absent. `training/run_eval.py` imports it at startup to build canonical evaluation prompts; `training/prepare_sft_data.py` also imports it. The existing prepared JSONL makes data preparation unnecessary for SFT.
- `/root/wg-recommendation/data/golden_wg_recommendations.jsonl` is absent. It is the default input for `training/run_eval.py`. Evaluation cannot be called golden evaluation without that file.
- `/root/wg-recommendation/artifacts/splits/sft_train.jsonl` and `/root/wg-recommendation/artifacts/splits/sft_validation.jsonl` are absent. They are default *source* inputs for `training/prepare_sft_data.py`; do not recreate them or overwrite the prepared JSONL.
- `training/create_runpod_bundle.py` expects additional `src`, `scripts`, `tools`, `reward_calibration`, documentation, golden, split, and calibration files. It is a packaging script for the source machine and is not needed to train on the already prepared JSONL.

None of these missing paths is required by `training/train_sft.py` when using its default prepared-data paths. No missing file was fabricated.

## 2026-09-26 — Hugging Face model setup (in progress)

### Completed

- The SFT stack is now installed: torch 2.8.0+cu128, torchvision 0.23.0+cu128, transformers 5.5.0, datasets 4.3.0, accelerate 1.15.0, peft 0.21.0, TRL 0.24.0, bitsandbytes 0.50.2, Unsloth 2026.9.11, xformers 0.0.32.post2, TensorBoard 2.21.0, Matplotlib 3.11.2, and huggingface_hub 1.33.0. `python -m pip check` passes. Installation occurred before this turn; this turn did not install packages.
- Confirmed `Qwen/Qwen3.5-2B` is public and ungated at revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Downloaded its complete 13-file snapshot to `/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc`. All 13 filenames and sizes match Hub metadata. The weights are the original 4,548,221,488-byte safetensors checkpoint, with tokenizer, chat template, config, index, and processor files. This is the 16-bit-capable source checkpoint used by the current LoRA SFT script, not GGUF or an inference quantization.
- The model is public, so download succeeded unauthenticated. `Hugging Face token environment variable` is not present in this process environment. No credential was printed, passed in a command, or written to the repository. Authenticated identity remains unverified.
- Found a token audit bug with Transformers 5.5: `apply_chat_template(tokenize=True)` returns `BatchEncoding` with two fields by default. `training/train_sft.py` now passes `return_dict=False`, so the audit counts token IDs. Using the downloaded local tokenizer, train token counts are min 729, p50 1021, p95 1304, max 1329; validation counts are min 842, p50 1023, p95 1302, max 1319. No record exceeds the configured 2048-token ceiling. All 390 prepared targets parse as JSON.
- `data/golden_wg_recommendations.jsonl` has appeared since the first audit. `src/prompts.py` is still absent; canonical evaluation still cannot start.

### Incomplete checks and observations

- A full SHA-256 pass over the 4.55 GB weight file was stopped because reads from `/workspace` were slow; file sizes and Hub snapshot download completion were verified. Local Unsloth 16-bit model load is in progress.
- Unsloth reports that optional flash linear attention and causal convolution fast kernels are absent and is falling back to PyTorch. Kernel performance is not yet measured. No optimizer step or full SFT run has started.

### Commands executed

- Hugging Face `HfApi.model_info('Qwen/Qwen3.5-2B', files_metadata=True)` and `snapshot_download` pinned to the revision above, with `token=False`.
- `python -m pip check`, local `AutoTokenizer.from_pretrained`, and `training.train_sft._token_audit` over the existing 300/90 prepared records.

## 2026-09-26 — Local SFT checkpoint validation (completed)

### Storage decision and download verification

Reads from the `/workspace` Hugging Face cache were unexpectedly slow: a 1 MiB read at a distant offset took roughly 16–19 seconds, and Unsloth loading made no progress past 0/617 tensors for more than five minutes. That read-only load was interrupted. The 13-file persistent snapshot remains in `/workspace`.

Downloaded a second copy directly to `/root/wg-recommendation/artifacts/models/Qwen3.5-2B` with `HF_HOME=/root/.cache/huggingface` and `HF_XET_HIGH_PERFORMANCE=1`. This completed in 39.1 seconds. All 13 file sizes match the pinned Hub revision, the weight index references the present safetensors file, and the 4,548,221,488-byte weight file's local SHA-256 matches the Hub's published LFS SHA-256: `aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`. This is the original Hugging Face safetensors checkpoint, suitable as the source for BF16 LoRA SFT. The `/root` copy is fast to read but may not persist across pod replacement; the `/workspace` snapshot is the persistent copy.

The safetensors header lists 632 tensors: 596 BF16 and 36 F32. The file is a trainable source checkpoint, not an inference-only quantization.

### Compatibility findings and fixes

Unsloth 2026.9.11 loads the local model with `load_in_4bit=False`, `load_in_16bit=True`, and `dtype=torch.bfloat16` in 12.88 seconds, returning `Qwen3_5ForConditionalGeneration` with 2,213,241,664 parameters and approximately 4.133 GiB GPU allocation after load. Optional fast attention/convolution kernels are missing; Unsloth falls back to PyTorch.

For this multimodal model, Unsloth returns a `Qwen3VLProcessor` as its second value. Its chat template rejects the prepared text-only message strings. `training/train_sft.py` and `training/run_eval.py` now unwrap `.tokenizer` for their text-only datasets. The full 300/90 token audit passes with the unwrapped tokenizer. A direct `AutoProcessor` import before Unsloth also exposed an installed `torchao`/PyTorch 2.8 import mismatch; the existing scripts import Unsloth before invoking the processor path, and that route worked. No PyTorch replacement was made.

A two-record SFT trainer initialization succeeded using the local checkpoint, BF16, sequence length 2048, completion-only loss, and rank-16 LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`. It produced 10,911,744 trainable parameters, 1,043 input tokens and 42 supervised completion tokens on the first record, with about 4.173 GiB GPU allocation. The check did not run an optimizer step or save a checkpoint. An earlier verification script expected a `labels` column and exited with `KeyError`; the prepared trainer dataset instead exposes `completion_mask`, so the verification was corrected and rerun successfully. This was a check-script assumption, not a trainer failure.

`python training/train_sft.py --validate-only`, `python training/train_sft.py --help`, and `python -m compileall -q training` passed. The JSONL split hashes remain unchanged. Hugging Face authenticated identity remains unverified because `Hugging Face token environment variable` is absent; the public model download required no credential. The key supplied in chat was not copied to commands or files.

### Next step

For the next SFT smoke run, pass `--model /root/wg-recommendation/artifacts/models/Qwen3.5-2B` so model weights are read from local disk. Keep a unique smoke output directory and run at most one optimizer step before proposing full SFT. Full SFT still requires user approval.

Updated `training/README.md` smoke and full SFT command examples to point at this verified local safetensors checkpoint. No training command from that document was executed.

## 2026-09-26 — Base evaluation and SFT smoke stage (in progress)

### Completed environment and input preflight

Verified the required training and evaluation scripts, prepared 300/90 SFT JSONL and manifest, golden JSONL, and `src/{__init__.py,prompts.py}` are present. `artifacts/splits/sft_validation.jsonl` is absent, so the requested one-example adapter reload evaluation must wait for that raw split. Syntax compilation, imports, and `--help` passed for `train_sft.py`, `run_eval.py`, and `plot_run.py`.

Python is `/usr/local/bin/python` 3.12.3; pip 25.2; GPU is RTX A6000 (49,140 MiB in `nvidia-smi`, 47.4 GiB in PyTorch), driver 595.91.07, driver CUDA 13.2, PyTorch 2.8.0+cu128 with CUDA 12.8, CUDA available and BF16 supported. Installed: Transformers 5.5.0, TRL 0.24.0, PEFT 0.21.0, Datasets 4.3.0, Accelerate 1.15.0, Unsloth 2026.9.11, unsloth_zoo 2026.9.7, bitsandbytes 0.50.2. `python -m pip check` passes. No package was installed or upgraded this turn.

`HF_TOKEN` is not present in the execution environment, and no stored Hugging Face credential is present. Anonymous metadata access to public, ungated `Qwen/Qwen3.5-2B` succeeds; current Hub `main` and the local cache resolve to revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Authenticated identity cannot be verified until a token is supplied through the environment. No token was printed or persisted.

Commands: `python training/train_sft.py --validate-only` and `python training/run_eval.py --stage base --run-name base-preflight-only --validate-only` both passed. SFT counts are 300 train and 90 validation, unique and disjoint, with 300/300 and 90/90 valid ranked three-ID JSON targets. Both SFT file SHA-256 hashes match the manifest. Golden evaluation input contains 50 unique valid scenarios, and `--validate-only` reports the supplied per-quote reward definition. Training default data paths are under `artifacts/training/sft_data`, outside held-out `data/`.

### Existing unchanged-base evaluation (verified; preserved)

The `artifacts/training/evaluations/golden-base-qwen35-2b/` directory already contained `predictions.jsonl` and `summary.json`, completed at 2026-09-26 11:27:06 UTC. This run was not overwritten or repeated. Verified 50 predictions match the current 50 golden scenario IDs in order, the current golden SHA-256 equals the run's stored dataset hash, every raw output re-parses to its stored prediction, and the stored Top-1 count is reproducible. It used `Qwen/Qwen3.5-2B`, revision request `main`, BF16 on the RTX A6000, greedy decoding, thinking disabled, max 128 new tokens, and the supplied per-quote reward only. The run did not record a resolved commit hash; the current Hub `main` and cache ref are the pinned revision above, but the exact weights loaded in that prior process cannot be proven from its summary alone.

Measured base results: Top-1 11/50 (0.22), Top-3 recall 0.273333, strict JSON 1/50 (0.02), valid Top-1 tariff 50/50 (1.0), valid three unique tariffs 1/50 (0.02), supplied reward ratio 0.844364, mean prompt tokens 995.0, mean completion tokens 126.3, model load 167.29 s, generation 396.445 s, throughput 0.126121 recommendations/s. All 50 outputs yielded a leniently parsed Top-1 tariff; 49/50 failed strict JSON, and 49/50 reached the 128-token cap. This is a baseline observation, not grounds to adjust SFT hyperparameters from golden results.

### Smoke preparation

Changed `training/train_sft.py` so `--smoke-test` audits all 300/90 records and their token lengths, then gives the trainer only the first two records from each split for one optimizer step and tiny evaluation. The resolved configuration now records the two subset counts and trainable parameter count, and training refuses zero trainable parameters. This changes smoke execution only; full SFT hyperparameters and data remain unchanged.

Populated `/root/.cache/huggingface` with the verified Qwen model revision and resolved its `main` ref, so the exact default model identifier loads from fast local disk when `HF_HOME=/root/.cache/huggingface` is set. This avoids the slow `/workspace` cache while preserving the standard `--model` default and supplied smoke command. The Hugging Face download was anonymous and did not change model weights.

### Planned next actions

Run the one-step smoke test in the unused `sft-qwen35-2b-smoke` directory; inspect finite train/eval losses, checkpoint, adapter, tokenizer, TensorBoard, and peak VRAM; make the smoke plot. Adapter reload generation waits for the missing raw SFT validation split. Do not launch full SFT.

### First smoke attempt — completed pipeline, no parameter update

Executed `HF_HOME=/root/.cache/huggingface python training/train_sft.py --run-name sft-qwen35-2b-smoke --smoke-test --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 1`. It audited all 300/90 records (max 1329/1319 tokens), then used two train and two validation rows for one optimizer step. BF16 and all seven intended LoRA target types loaded; 10,911,744 parameters were trainable. The script saved a checkpoint and final adapter, and its finite losses were train 0.1147468686 and final validation 0.1206426769. Measured train/evaluation/save interval was 70.314 s, peak allocated GPU memory 5.679 GiB. TensorBoard event files were produced under `checkpoints/runs/`, not the `tensorboard/` path recorded in the first run record; the installed Transformers version ignored the deprecated `logging_dir` argument. Checkpoint and tokenizer files exist.

The optimizer checkpoint has step 1 and nonempty Adam state, but the logged step learning rate was 0. Comparing all 192 saved LoRA tensors against a fresh same-seed initialization showed **zero changed tensors**. This first run is therefore an incomplete test of actual parameter updating even though `run_record.json` says `status: complete`. The directory is preserved. The SFT smoke-only schedule was changed to constant learning rate with zero warmup, keeping the full-run cosine schedule, 10% warmup, and all requested full SFT hyperparameters unchanged. The script now uses `TENSORBOARD_LOGGING_DIR` so TensorBoard events should land in the documented run directory. A new run name will be used for verification.

## 2026-09-26 — Corrected SFT smoke and adapter reload (completed)

### Command and configuration

Executed `HF_HOME=/root/.cache/huggingface python training/train_sft.py --run-name sft-qwen35-2b-smoke-v2 --smoke-test --train-batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 1`. This used the original `Qwen/Qwen3.5-2B` model at current cached `main` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, BF16, no 4-bit quantization, max length 2048, rank 16 and alpha 16, seven projection targets, effective batch size 1, seed 3407, AdamW, and completion-only loss. The smoke-only scheduler was constant at learning rate `2e-4`, with no warmup. All 300/90 records were validated and token-audited; the trainer used two records per split for one optimizer step and tiny evaluation.

### Measured results and artifact verification

The corrected run has 10,911,744 trainable parameters. Train loss was 0.1147468686; the post-training validation loss was 0.0721997321; both are finite. The logged step learning rate was 0.0002, gradient norm 2.664, and optimizer state records step 1. Compared with the first same-seed zero-update smoke checkpoint, all 192 adapter tensors changed, with maximum absolute tensor difference about 0.0002. The train/evaluation/save interval in `run_record.json` was 11.292 seconds, with 5.679 GiB peak allocated GPU memory; the first run took 70.314 seconds in the same interval because it included first-use compilation/setup. These one-step measurements are pipeline checks, not model-quality evidence.

Verified `artifacts/training/runs/sft-qwen35-2b-smoke-v2/` contains `resolved_config.json`, `run_record.json`, `command.txt`, `status.json`, `environment.json`, `metrics.json`, `final_adapter/{adapter_config.json,adapter_model.safetensors,tokenizer.json,tokenizer_config.json,chat_template.jinja}`, `checkpoints/checkpoint-1/{adapter_model.safetensors,optimizer.pt,scheduler.pt,trainer_state.json}`, `checkpoints/trainer_state.json`, two TensorBoard event files under `tensorboard/`, and `training_curves.png` (77,329 bytes). The plot was made with `python training/plot_run.py artifacts/training/runs/sft-qwen35-2b-smoke-v2/run_record.json`; its single training point is only a plotting check. The original `sft-qwen35-2b-smoke/` directory remains intact and has `status.json` marking its no-update limitation despite the earlier `run_record.json` complete flag.

### Adapter reload and evaluation limit

`artifacts/splits/sft_validation.jsonl` is still absent, so the requested canonical `training/run_eval.py --stage sft --dataset artifacts/splits/sft_validation.jsonl --limit 1` was not run. No golden example was substituted. A separate technical reload used the first already prepared `artifacts/training/sft_data/validation.jsonl` prompt: Unsloth loaded `sft-qwen35-2b-smoke-v2/final_adapter`, generated 45 tokens from an 803-token prompt, and produced strict JSON with three parsed tariff IDs. The raw response and metadata are saved to `sft-qwen35-2b-smoke-v2/reload_check.json` with `canonical_evaluation: false`. This is not a quality metric.

### Warnings and decisions

Unsloth used its PyTorch fallback because optional flash linear attention and causal-convolution kernels are absent. `warmup_ratio` and some PyTorch backend properties emit deprecation warnings. The tokenizer EOS token was aligned to model generation config during trainer construction. `training/README.md` now documents the fast local Hub cache and unique names for any repeat smoke checks; `training/run_eval.py` will record a resolved model commit hash in future summaries when the loader exposes it. The prior base evaluation result was preserved and never overwritten. `HF_TOKEN` remains unavailable to this process, so authenticated identity remains unverified; anonymous access to the public model works. No package or CUDA/PyTorch version was changed.

### Proposed full SFT and rough time estimate (planned, not executed)

Use `HF_HOME=/root/.cache/huggingface` in the shell so the default model ID reads the fast local cache, then run the unchanged command:

```bash
python training/train_sft.py \
  --run-name sft-qwen35-2b-seed3407
```

The default full configuration remains three epochs, learning rate `2e-4`, BF16 LoRA rank 16/alpha 16, sequence length 2048, batch size 2 with accumulation 8 (effective batch 16), cosine schedule with 10% warmup, and seed 3407. For 300 rows and this effective batch, expect roughly 57 optimizer steps across three epochs, plus full 90-record validation each epoch and checkpoint writes. The measured one-step intervals were 11.292 seconds with cached compilation and 70.314 seconds on the first run, each at effective batch 1 and two-row validation. Scaling for 16 examples per full optimizer step and separate full evaluations suggests roughly **1–3 hours**, with large uncertainty from sequence lengths, batch-2 kernels, first-use compilation, and storage/checkpoint overhead. Full SFT has not started and awaits approval.

## Base Qwen3.5-2B Golden Evaluation

**Date:** 2026-09-26 UTC. **Purpose:** unchanged pre-training baseline, before SFT. The preflight used `python training/run_eval.py --stage base --run-name golden-base-qwen35-2b --validate-only` and confirmed the base model, null adapter, 50 golden scenarios, the dataset hash, and supplied per-quote reward without calibrated weights. The output directory did not exist before this run.

**Exact evaluation command:**

```bash
python training/run_eval.py --stage base --run-name golden-base-qwen35-2b --confirm-golden-eval
```

| Input or configuration | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` |
| Requested revision | `main` |
| Resolved revision (local Hugging Face `refs/main`) | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Golden dataset | `/root/wg-recommendation/data/golden_wg_recommendations.jsonl` |
| Golden dataset SHA-256 | `864123d2f1c2741f94cfef94ebdd09ecc6eb6d1b3fe898212441ad99b54eaa33` |
| Scenarios | 50 |
| Generation | Greedy (`do_sample=False`), thinking disabled, `max_new_tokens=128`, `max_seq_length=2048` |
| Precision | BF16, 16-bit model load; no 4-bit quantization |

| Environment | Value |
| --- | --- |
| Python | 3.12.3 |
| GPU | NVIDIA RTX A6000, 49,140 MiB total; 43,960 MiB free before evaluation |
| NVIDIA driver | 595.91.07 |
| CUDA | 13.2 driver capability (`nvidia-smi`); 12.8 PyTorch runtime |
| PyTorch | 2.8.0+cu128 |
| Transformers | 5.5.0 |
| Unsloth | 2026.9.11 |
| BF16 supported | Yes |
| Hugging Face token present in process environment | No |

| Metric | Result |
| --- | ---: |
| Top-1 accuracy | 0.22 (11/50 correct) |
| Top-3 recall | 0.273333 |
| Strict JSON rate | 0.02 (1/50) |
| Valid Top-1 tariff rate | 1.0 (50/50) |
| Valid three-unique-tariffs rate | 0.02 (1/50) |
| Supplied reward ratio | 0.844364 |
| Reward ratio, valid picks only | 0.844364 |
| Mean prompt tokens | 995.0 |
| Mean completion tokens | 126.3 |
| Generation time | 396.445 seconds |
| Recommendations per second | 0.126121 |
| Model loading time | 167.29 seconds |
| Unrecoverable parsing failures (no extracted tariff ID) | 0 |
| Strict JSON failures | 49 |
| Invalid Top-1 tariff selections | 0 |
| Invalid three-unique-tariff rankings | 49 |

**Outputs:** `/root/wg-recommendation/artifacts/training/evaluations/golden-base-qwen35-2b/predictions.jsonl` and `/root/wg-recommendation/artifacts/training/evaluations/golden-base-qwen35-2b/summary.json`. Both files parsed successfully. The predictions contain exactly 50 records, with unique scenario IDs in the same order as the golden dataset. The summary reports 50 scenarios, and its metrics match recomputation from all prediction rows. No scenario disappeared after parsing. Rows with no valid Top-1 tariff would remain in the denominator with zero reward under the current evaluator; this run had no such rows.

**Warnings and limitation:** Unsloth reported that optional fast attention/convolution libraries were absent and fell back to PyTorch. It also emitted PyTorch backend deprecation warnings. The evaluator's lenient parser recovered a valid first tariff from 49 outputs that failed strict JSON and did not contain a valid three-unique-tariff ranking. Those 49 rows received Top-1 and reward credit despite their invalid full output format. Thus the requirement that *all* invalid outputs receive zero credit is **not met** by the current evaluator; the recorded metrics are its unchanged, as-run metrics. No generation or model failure interrupted the run.

No adapter was loaded. No training had occurred. The canonical prompt was used. The supplied per-quote reward was used unchanged. The calibrated GRPO reward was not used. The golden dataset was not used to tune any setting. These results are frozen as the pre-SFT baseline. SFT must use its existing predetermined configuration rather than reacting to this result. No SFT smoke test or full training was started.

## 2026-09-26 — GPU batch sizing for SFT

The user requested a larger batch and then explicitly selected per-device batch size 24. Added `--smoke-steps` and a per-optimizer-step timer to `training/train_sft.py` so batch candidates could be compared on real prepared data without starting the full run. The tests kept BF16, LoRA rank 16, sequence length 2048, seed 3407, learning rate `2e-4`, completion-only loss, and eval batch size 2. The first four tests used effective batch 16; the requested batch 24 test used accumulation 1 and effective batch 24. Each test audited all 300/90 records and ran two optimizer updates on its first 32 or 48 training records, followed by two-row validation. First-step times include compilation and vary with cache state, so they are unsuitable for throughput comparison.

| Per-device batch | Accumulation | Effective batch | Second step, seconds | Peak allocated VRAM, GiB | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 8 | 16 | 3.532 | 7.121 | complete, finite losses |
| 4 | 4 | 16 | 3.520 | 7.141 | complete, finite losses |
| 8 | 2 | 16 | 3.309 | 7.180 | complete, finite losses |
| 16 | 1 | 16 | 2.905 | 7.259 | complete, finite losses |
| 24 | 1 | 24 | 4.470 | 8.377 | complete, finite losses |

The batch 24 run is `artifacts/training/runs/sft-batch-b24-a1-fast-bench/`; its measured train loss was 0.106914, final two-row validation loss 0.043118, and step times were 41.037 and 4.470 seconds. These short runs establish that the batch fits and training updates execute, but they do not establish final model quality or full-run speed. The first 48 rows were used for batch 24; the full token audit still covered the longest 1329-token train example. `training/train_sft.py` defaults are now train batch 24 and gradient accumulation 1; all other full-run training settings remain as before. For 300 training rows and three epochs, expect about 39 optimizer steps, validation and saving after each epoch, and at most two retained checkpoints before the final adapter is saved.

The requested `/workspace/huggingface-cache` path initially did not exist. It was linked to the existing persistent `/workspace/.cache/huggingface` cache, which has the same Qwen revision. Loading model weights from `/workspace` proved very slow, so that attempted batch 24 load was stopped before training. The successful batch 24 check used `HF_HOME=/root/.cache/huggingface` and `HF_HUB_CACHE=/root/.cache/huggingface/hub`; these are the recommended values for this pod. The user will start the full SFT script. Full SFT has not been launched by the assistant.

## 2026-09-26 — Full SFT completed by user

The user reported a completed SFT run at `artifacts/training/runs/sft-qwen35-2b-seed3407/`. I checked its on-disk `run_record.json`, `resolved_config.json`, trainer state, checkpoint directories, final adapter, and TensorBoard directory. The run record confirms `status: complete`, from 12:22:27 to 12:29:20 UTC. Its measured train interval was 378.169 seconds (6 minutes 18 seconds), peak allocated GPU memory 12.612 GiB, train loss 0.04926585, and final 90-row validation loss 0.02673021. The trainer reported 376.5792 seconds training runtime and 2.39 training samples per second. All 300 training and 90 validation records were used, disjoint by recorded dataset audit, with no examples above the 2048-token ceiling. The Qwen revision recorded was `15852e8c16360a2fea060d615a32b45270f8a8fc`; precision was BF16, with 10,911,744 trainable LoRA parameters.

**Actual batch setting:** This run's recorded per-device train batch size is **48**, gradient accumulation **1**, and effective batch **48**. That differs from the batch-24 command and repository default previously given to the user. The exact invocation was not saved, so do not describe this as a batch-24 run. Other recorded settings were three epochs, learning rate `2e-4`, cosine schedule with 10% warmup, LoRA rank/alpha 16, eval batch 2, max length 2048, and seed 3407. The trainer completed **21 optimizer steps** (7 per epoch). The prior estimate of 39 steps applied to batch 24 and does not apply to this run.

Validation loss at epoch checkpoints was 0.03753975 (step 7), 0.02811436 (step 14), and 0.02673652 (step 21). Trainer state identifies `checkpoint-21` as the best checkpoint, with best metric 0.02673652. A separate final evaluation after training reported 0.02673021. The two retained checkpoints are `checkpoint-14` and `checkpoint-21`; the `final_adapter/` directory contains adapter safetensors, config, tokenizer, and chat template. Its saved adapter tensors are BF16 while checkpoint tensors are FP32, so byte hashes differ. TensorBoard events are present. These are training and validation-loss results; no post-SFT ranking or golden evaluation was performed as part of this check.

## 2026-09-26 — Optional Qwen3.5 fast kernels installed

At the user's request, installed `flash-linear-attention[cuda]==0.5.2` (which brought in `fla-core==0.5.2` and `einops==0.8.2`), `causal-conv1d==1.7.0`, and its required `ninja==1.13.2` in the pod's existing Python environment. The causal-conv1d package came from the project's prebuilt `cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64` GitHub release wheel, matching Python 3.12, PyTorch 2.8.0+cu128, CUDA 12.8, x86_64, and PyTorch CXX11 ABI true. The GitHub API lists its release asset SHA-256 as `a1fcd5d022b67ddfc9e1c291f890004c1c081a03d66b532f6eebe31219168e0e`. No PyTorch, Transformers, or Unsloth version was changed by the install, and `pip check` found no broken requirements. The exact reinstall commands are in `training/README.md`.

In a fresh process after importing Unsloth, Transformers reports both `is_causal_conv1d_available()` and `is_flash_linear_attention_available()` as true, and all Qwen3.5 fast-path symbols import (`causal_conv1d_fn`, `causal_conv1d_update`, `FusedRMSNormGated`, `chunk_gated_delta_rule`, and `fused_recurrent_gated_delta_rule`). An SFT golden evaluation with a 700-token output ceiling was already running before the install; that process retains its previously imported PyTorch fallback. No new evaluation was launched for this install.

The user subsequently requested repeatable installation. Added `flash-linear-attention[cuda]==0.5.2`, `ninja==1.13.2`, and the exact causal-conv1d release wheel URL with its SHA-256 to `training/requirements.txt`. `pip install --dry-run -r training/requirements.txt` resolved successfully on this pod without planning to replace its PyTorch, Transformers, or Unsloth packages. The wheel pin is specific to this pod's Python/PyTorch/CUDA/ABI/platform combination; another GPU environment needs a matching wheel.

## Corrected Base Qwen3.5-2B Golden Evaluation — 700-token allowance

**Date:** 2026-09-26 UTC. The previous 128-token run was a preliminary truncated diagnostic: its mean completion length was 126.3 tokens, near the ceiling. Its 50-scenario metrics were Top-1 accuracy 0.22 (11 correct), Top-3 recall 0.273333, strict JSON rate 0.02, valid Top-1 tariff rate 1.0, valid three-unique-tariffs rate 0.02, supplied reward ratio 0.844364, and valid-pick reward ratio 0.844364. It took 396.445 seconds to generate at 0.126121 recommendations/second and 167.29 seconds to load. Its artifacts at `artifacts/training/evaluations/golden-base-qwen35-2b/` were preserved unchanged; the prediction and summary SHA-256 hashes remain `f8e6705cd62353a9ebc27ba56eda09738656da08e124a8be8c7f38b700c2ebee` and `9e466b894c329b80210f874c949c5971ae6a232dedc305a33610de03ef5aec64`.

**Code change:** In `training/run_eval.py`, set the default `--max-new-tokens` to 700. The live file had default 2048 immediately before this edit, although the preliminary run had used 128. A comment explains that the supplied Qwen evaluator allows 700, the expected JSON is shorter, this avoids truncating an untuned model's verbose output, and all base/SFT/GRPO evaluations must share the ceiling. Added a pre-generation check that fails with the scenario ID, prompt token count, requested output allowance, configured maximum sequence length, and minimum required length if prompt tokens plus output allowance exceed `--max-seq-length`. Prompt construction, parsing, metrics, reward calculation, and the default 2048 sequence length were unchanged. `python -m py_compile training/run_eval.py` passed.

**Preflight and capacity audit:** The requested `--validate-only` command confirmed `Qwen/Qwen3.5-2B`, 50 golden scenarios, no adapter, the dataset hash, and supplied per-quote reward with no calibrated weights. The local model tokenizer was applied to all 50 exact canonical system/user messages with the same chat-template options as the evaluator, including `enable_thinking=False`. Prompt tokens: minimum 802, median 985.5, nearest-rank p95 1259, maximum 1261 (scenarios `wg-013` and `wg-025`). Maximum prompt plus 700 is **1961**, within 2048 by 87 tokens.

**Exact corrected command:**

```bash
python training/run_eval.py --stage base --run-name golden-base-qwen35-2b-700tok --max-seq-length 2048 --max-new-tokens 700 --confirm-golden-eval
```

| Input or generation configuration | Value |
| --- | --- |
| Model | `Qwen/Qwen3.5-2B` |
| Requested revision | `main` |
| Resolved revision in summary | `15852e8c16360a2fea060d615a32b45270f8a8fc` |
| Golden dataset | `/root/wg-recommendation/data/golden_wg_recommendations.jsonl` |
| Golden dataset SHA-256 | `864123d2f1c2741f94cfef94ebdd09ecc6eb6d1b3fe898212441ad99b54eaa33` |
| Scenarios | 50 |
| Generation | Deterministic (`do_sample=False`), thinking disabled, 700 new tokens maximum, 2048 sequence length |
| GPU and precision | NVIDIA RTX A6000, BF16 |

| Corrected metric | Result |
| --- | ---: |
| Top-1 accuracy | 0.22 (11/50 correct) |
| Top-3 recall | 0.666667 |
| Strict JSON rate | 0.02 (1/50) |
| Valid Top-1 tariff rate | 1.0 (50/50) |
| Valid three-unique-tariffs rate | 1.0 (50/50) |
| Supplied reward ratio | 0.844364 |
| Reward ratio, valid picks only | 0.844364 |
| Mean prompt tokens | 995.0 |
| Mean completion tokens | 493.82 |
| Completion p50 / nearest-rank p95 / maximum | 500 / 617 / 628 tokens |
| Completions reaching exactly 700 tokens | 0/50 (0%) |
| Unrecoverable parsing failures | 0 |
| Invalid Top-1 tariff selections | 0 |
| Generation runtime | 979.76 seconds |
| Throughput | 0.051033 recommendations/second |
| Model loading time | 194.146 seconds |
| Peak GPU memory | Unavailable; no peak measurement was recorded for this evaluation process |

**Outputs:** `/root/wg-recommendation/artifacts/training/evaluations/golden-base-qwen35-2b-700tok/predictions.jsonl` and `/root/wg-recommendation/artifacts/training/evaluations/golden-base-qwen35-2b-700tok/summary.json`. Both parse successfully. The predictions contain exactly 50 unique scenario IDs, matching the complete golden dataset in order; no scenario is missing. The summary reports 50, records deterministic 700-token generation with thinking disabled, and its metrics match recomputation from the prediction rows.

**Warnings and timing limitation:** Unsloth reported its optional fast-kernel fallback to PyTorch and emitted backend deprecation warnings. Strict JSON remained at 1/50 despite all 50 rows yielding three unique tariff IDs under the unchanged lenient parser; malformed full outputs can still receive selection and reward credit. No response reached the new ceiling, so this run does not show a 700-token stopping/repetition problem. A separate user-started Python process shared the GPU during part of generation (about 14 GiB used when observed). Generation runtime and throughput are therefore affected by contention and should not be compared directly with the preliminary run's timing. A point-in-time GPU sample before the second process appeared showed about 4,760 MiB used; it is not a peak measurement.

This remains an unchanged, pre-training **base-model** run: the model weights were not modified, and no adapter was loaded. The canonical prompt was unchanged. The supplied golden reward was unchanged. The calibrated GRPO reward was not used. The only intended benchmark configuration change from the preliminary run was the Qwen output ceiling from 128 to 700 tokens. No SFT or GRPO adapter was involved in this evaluation; the user did run SFT concurrently later, so it would be inaccurate to claim that no SFT had occurred by the time this run finished. All later SFT and GRPO evaluations must use the same final 2048 sequence and 700 output-token limits. The golden result must not be used for hyperparameter tuning. The assistant did not start SFT in this task.

## SFT Qwen3.5-2B Golden Evaluation — 700-token allowance

**Date:** 2026-09-26 UTC. Evaluated the completed SFT LoRA at `artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter` against all 50 golden scenarios. The adapter directory hash recorded by the evaluator is `f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48`. Base model: `Qwen/Qwen3.5-2B`, resolved revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. Golden dataset: `/root/wg-recommendation/data/golden_wg_recommendations.jsonl`, SHA-256 `864123d2f1c2741f94cfef94ebdd09ecc6eb6d1b3fe898212441ad99b54eaa33`.

**Exact command:**

```bash
python training/run_eval.py --stage sft --adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter --run-name golden-sft-qwen35-2b-seed3407-700tok --max-seq-length 2048 --max-new-tokens 700 --confirm-golden-eval
```

Preflight confirmed the adapter, 50 golden scenarios, and unchanged supplied per-quote reward. An audit with the saved adapter tokenizer found prompt lengths from 802 to 1261 tokens; the largest prompt plus the 700-token allowance is 1961, within the 2048-token sequence limit. Generation was deterministic (`do_sample=False`), with thinking disabled, BF16 precision, and the canonical broker prompt. The 700 tokens are a **maximum allowance**, not a forced completion length.

| Metric | SFT result | Corrected base result |
| --- | ---: | ---: |
| Top-1 accuracy | 0.84 (42/50) | 0.22 (11/50) |
| Top-3 recall | 0.866667 | 0.666667 |
| Strict JSON rate | 1.0 (50/50) | 0.02 (1/50) |
| Valid Top-1 tariff rate | 1.0 | 1.0 |
| Valid three-unique-tariffs rate | 1.0 | 1.0 |
| Supplied reward ratio | 0.811724 | 0.844364 |
| Reward ratio, valid picks only | 0.811724 | 0.844364 |
| Mean prompt tokens | 995.0 | 995.0 |
| Completion tokens, mean / median / nearest-rank p95 / max | 43.3 / 43 / 45 / 45 | 493.82 / 500 / 617 / 628 |
| Completions reaching 700 tokens | 0/50 | 0/50 |
| Unrecoverable parsing failures | 0 | 0 |
| Invalid Top-1 tariff selections | 0 | 0 |
| Generation runtime | 222.164 seconds | 979.76 seconds (GPU shared during part of base run) |
| Throughput | 0.225059 recommendations/second | 0.051033 recommendations/second |
| Model loading time | 256.815 seconds | 194.146 seconds |

**Artifacts:** `artifacts/training/evaluations/golden-sft-qwen35-2b-seed3407-700tok/predictions.jsonl` and `artifacts/training/evaluations/golden-sft-qwen35-2b-seed3407-700tok/summary.json`. Both parse correctly. Predictions contain 50 unique scenario IDs in golden-dataset order with none missing, and the summary metrics match recomputation from all rows. Its generation configuration records `max_new_tokens=700`, `max_seq_length=2048`, no sampling, and thinking disabled. The corrected base artifacts remain unchanged.

**Interpretation and warnings:** The SFT run improved agreement with the teacher ranking and output format. Its supplied per-quote reward ratio is lower than the base run's; teacher Top-1 agreement and quote reward optimize different targets, so neither number should be substituted for the other. No completion approached the 700-token limit. Unsloth reported the optional fast-kernel fallback to PyTorch and emitted backend deprecation warnings. The model load was slow; no model, prompt, parsing, metric, or reward setting was changed in response. No additional training was started for this evaluation, and the golden result was not used to tune settings.

## SFT Baseline on GRPO Validation

**Date:** 2026-09-26 UTC. Evaluated the completed SFT LoRA on the complete 90-example GRPO validation split before GRPO training. The split has 90 unique scenario IDs and no overlap with the SFT train, SFT validation, or GRPO train splits. The dataset SHA-256 is `56e96d73a877eff594282a154db9458a0916973f1116d8e09170faddb59b459e`; the adapter directory hash is `f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48`. Base model `Qwen/Qwen3.5-2B` resolved to revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.

```bash
python training/run_eval.py --stage sft --adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter --dataset artifacts/splits/grpo_validation.jsonl --limit 90 --run-name grpo-validation-sft-baseline --max-seq-length 2048 --max-new-tokens 700
```

Preflight passed. The adapter tokenizer audit found the longest prompt at 1289 tokens, leaving room for 700 output tokens within the 2048-token sequence length. Generation was deterministic, with thinking disabled. This is a validation-split evaluation using the stored supplied per-quote reward; it is not the calibrated GRPO training reward and it is not a golden-dataset run.

| Metric | Result |
| --- | ---: |
| Top-1 accuracy | 0.922222 (83/90 correct) |
| Top-3 recall | 0.87037 |
| Strict JSON rate | 1.0 (90/90) |
| Valid Top-1 tariff rate | 1.0 (90/90) |
| Valid three-unique-tariffs rate | 1.0 (90/90) |
| Supplied reward ratio | 0.856095 |
| Reward ratio, valid picks only | 0.856095 |
| Mean prompt tokens | 1023.578 |
| Completion tokens, mean / median / nearest-rank p95 / maximum | 43.267 / 43 / 45 / 45 |
| Completions at the 700-token ceiling | 0/90 |
| Unrecoverable parsing failures / invalid Top-1 selections | 0 / 0 |
| Generation runtime | 200.865 seconds |
| Throughput | 0.448063 recommendations/second |
| Model loading time | 149.099 seconds |

**Artifacts:** `artifacts/training/evaluations/grpo-validation-sft-baseline/predictions.jsonl` and `artifacts/training/evaluations/grpo-validation-sft-baseline/summary.json`. Both parse correctly; all 90 scenario IDs are unique and match the validation split in order, and every summary metric matches recomputation from the prediction rows. The summary records the SFT adapter, `max_new_tokens=700`, `max_seq_length=2048`, no sampling, and thinking disabled. Unsloth reported optional fast-kernel fallback and backend deprecation warnings. No GRPO training was started in this evaluation task.

## Evaluation Results for Final Report — Base Model and SFT Adapter

**Recorded 2026-09-26 UTC.** This section is the concise source for reporting the completed evaluations. The detailed run sections above contain exact commands, environment, hashes, and validation checks. The SFT checkpoint is `sft-qwen35-2b-seed3407/final_adapter`; its directory hash is `f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48`. All runs used `Qwen/Qwen3.5-2B`, the canonical broker prompt, deterministic generation, thinking disabled, a 2048-token sequence limit, and the supplied per-quote reward in the evaluated dataset. The resolved base-model revision for the corrected and SFT runs is `15852e8c16360a2fea060d615a32b45270f8a8fc`. The calibrated GRPO training reward was not used for these evaluation metrics.

| Named run | Dataset | Output allowance | Top-1 correct / accuracy | Top-3 recall | Strict JSON | Valid three-tariff ranking | Supplied reward ratio | Mean completion tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Preliminary truncated base diagnostic (`golden-base-qwen35-2b`) | Golden, 50 | 128 | 11/50; 0.22 | 0.273333 | 1/50; 0.02 | 1/50; 0.02 | 0.844364 | 126.3 |
| Corrected pre-SFT base baseline (`golden-base-qwen35-2b-700tok`) | Golden, 50 | 700 | 11/50; 0.22 | 0.666667 | 1/50; 0.02 | 50/50; 1.0 | 0.844364 | 493.82 |
| Post-SFT golden evaluation (`golden-sft-qwen35-2b-seed3407-700tok`) | Golden, 50 | 700 | 42/50; 0.84 | 0.866667 | 50/50; 1.0 | 50/50; 1.0 | 0.811724 | 43.3 |
| Pre-GRPO SFT baseline (`grpo-validation-sft-baseline`) | GRPO validation, 90 | 700 | 83/90; 0.922222 | 0.87037 | 90/90; 1.0 | 90/90; 1.0 | 0.856095 | 43.267 |

**What changed:** The preliminary 128-token base run was capped on most examples (126.3 mean completion tokens). Raising the allowance to 700 let the base model produce three valid unique tariff IDs on all 50 golden scenarios, but strict JSON was still only **1/50**, not zero. After SFT, strict JSON was **50/50** on the same golden set and **90/90** on the separate GRPO validation set. On the golden set with the same 700-token allowance, Top-1 agreement rose from **11/50 to 42/50** and Top-3 recall from **0.666667 to 0.866667**. These are measured output differences from the saved base and SFT checkpoints; do not describe the 128-token diagnostic as the final base comparison.

**Metric interpretation:** The golden SFT run's supplied reward ratio (0.811724) is lower than the corrected base run's (0.844364), even though teacher-ranking agreement improved. Teacher-ranking accuracy and supplied quote reward are separate measurements. The GRPO validation figures use a different 90-scenario split, so they should not be presented as a direct before/after comparison with the 50-scenario golden figures. That split has no scenario-ID overlap with SFT train, SFT validation, or GRPO train. All four evaluations had 100% valid Top-1 tariff selections and zero unrecoverable parsing failures; the base runs' non-strict outputs could still earn credit through the unchanged lenient parser. The 700-token runs had **zero completions at the ceiling**; the SFT completion maxima were 45 tokens on both datasets.

**Runtime and provenance:** Corrected base generation took 979.76 seconds (0.051033 recommendations/second) with 194.146 seconds model loading; the GPU was shared with a user-started training process during part of generation, so its speed is not a clean model comparison. Golden SFT generation took 222.164 seconds (0.225059 recommendations/second) with 256.815 seconds loading. GRPO-validation SFT generation took 200.865 seconds (0.448063 recommendations/second) with 149.099 seconds loading. Golden dataset SHA-256: `864123d2f1c2741f94cfef94ebdd09ecc6eb6d1b3fe898212441ad99b54eaa33`; GRPO validation SHA-256: `56e96d73a877eff594282a154db9458a0916973f1116d8e09170faddb59b459e`. Each named run has `predictions.jsonl` and `summary.json` under `artifacts/training/evaluations/<run name>/`; the preliminary files remain preserved. The golden set was not used to choose settings. No GRPO training result is included here.

## Conservative GRPO Smoke and 10-Step Gate

**Date:** 2026-09-26 UTC. This is a controlled ablation from an already strong SFT model. The source was `artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter`, whose evaluator-style **directory** SHA-256 was `f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48` before and after GRPO. The original adapter and its evaluation results were preserved. The base was `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`.

**Preflight and safeguards.** All required scripts, prepared data, manifest, raw validation split, and SFT adapter existed. `python -m py_compile training/grpo/train.py training/grpo/rewards.py` passed. The following validation command passed with 810 train and 90 validation records, disjoint scenario IDs, matching manifest checksums, no teacher labels in GRPO train, and default `max_steps=50`:

```bash
python training/grpo/train.py --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter --validate-only
```

The trainer defaults are `--max-steps 50`, `--save-steps 10`, and `--eval-steps 50`; `--smoke-test` forces one step, and `--max-steps` above 100 is refused without the explicit boolean `--allow-long-run`. A validation-only probe at 101 steps confirmed that refusal. The trainer checks for nonzero trainable parameters after loading the SFT adapter. No golden path is used by the GRPO trainer. Prepared train SHA-256: `f93a7a9291aa3acf938e75bced912ee0d393853322ba7fb3fbbe707776e3d81e`; prepared validation SHA-256: `fb78e73af23420338b504741a9115be4abdb035f9cdf905ebafd8362d56f7d89`; manifest SHA-256: `e5281b29713398c9763091039302e317a9968b4deb5e09482c371deedebd6801`. The raw 90-example validation split SHA-256 was `56e96d73a877eff594282a154db9458a0916973f1116d8e09170faddb59b459e`. A separate reproduction of the preparation process matched both prepared JSONL files byte-for-byte; no data was regenerated in place.

**Frozen reward.** `artifacts/reward_calibration/best_weights.json` SHA-256 was `735f900d8bb043cc694c8f57a73ed01723389225611ba7a50d1e3db7ee6ac318`. The manifest records `w_cov=0.0928034111891159`, `w_ins=0.4102710203778362`, `w_ded=0.06960255839183692`, `w_score=0.42732301004121104`, and `w_price=1.793611631132337`. The four rollout components and fixed composition weights are `strict_json_reward` 0.10, `valid_tariffs_reward` 0.10, `calibrated_top1_reward` 0.40, and `calibrated_ranking_reward` 0.40. No reward weights were recalibrated or searched.

Offline checks on five prepared training records constructed ideal, reverse, unknown-ID, duplicate-ID, and prose-wrapped rankings using `training.common.ranking_json` and `training.grpo.rewards.reward_functions`. Every component was finite and within [0,1]. Ideal strict JSON rankings scored `[1,1,1,1]` in component order. Reverse rankings had lower business reward: Top-1 was 0 and ranking reward ranged 0.009573–0.059865 for the five sampled records. Unknown IDs received zero Top-1 and ranking rewards; duplicate IDs received ranking reward 0 and valid-tariffs reward 1/3. Only exact required JSON received strict-format reward 1.0. The unchanged lenient extractor can still award business reward to prose containing a valid JSON ranking; its strict-format component is 0. A syntactically strict ranking of unknown IDs can receive strict-format reward 1 while its validity and business rewards are 0. These components intentionally measure separate properties.

**Environment and configuration.** Python 3.12.3, NVIDIA RTX A6000 (one GPU), CUDA 12.8, BF16, PyTorch 2.8.0+cu128, Transformers 5.5.0, Unsloth 2026.9.11, TRL 0.24.0, PEFT 0.21.0, Accelerate 1.15.0, bitsandbytes 0.50.2, and Datasets 4.3.0. The loaded model had 2,224,153,408 total and **10,911,744 trainable LoRA parameters** (0.490602%); base parameters remained frozen. The ten-step settings were learning rate `5e-6`, beta `0.001`, four generations per prompt, temperature `0.8`, top-p `0.95`, 128 maximum completion tokens, 2048 sequence tokens, BF16, Dr. GRPO loss, reward scaling off, effective batch size four, and seed 3407. The smoke test used the same substantive settings with a constant `5e-6` learning rate and no warmup so its sole update was effective; the ten-step run used the configured cosine schedule and 10% warmup. All 810 train and 90 validation prompts passed the token-margin audit; maximum prompt lengths were 1280 and 1289 respectively, leaving room for 128 completion tokens.

**One-step smoke command** (an earlier smoke attempt was interrupted before training while correcting the token audit, so this completed run used a unique suffix):

```bash
HF_HOME=/root/.cache/huggingface HF_HUB_CACHE=/root/.cache/huggingface/hub python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-smoke-v2 --smoke-test
```

Exactly one optimizer update completed. Four sampled completions formed the prompt group. Training loss was finite at `4.30018e-05`; mean reward `0.993959`, reward standard deviation `0.009884`, and fraction of groups with zero reward standard deviation `0`. Strict JSON and valid-tariff means were both 1 with zero standard deviation. Calibrated Top-1 mean was 1 with zero standard deviation; calibrated ranking mean was `0.984896` with standard deviation `0.024709`. Thus at least two of the four rollouts differed in reward and could not have been identical outputs, although exact distinct-output count was not retained. KL was `0.043838`; clipping ratio was 0; mean/min/max completion lengths were `41.25/41/42`. Training runtime was `189.361` seconds, run-record elapsed training interval `191.211` seconds, and peak GPU memory `6.861 GiB`. `checkpoint-1` and `final_adapter` saved, with tokenizer files. The final adapter reloaded through the evaluator and produced one strict-JSON, valid three-tariff recommendation at `artifacts/training/evaluations/grpo-smoke-v2-reload/`. The smoke adapter's 192 LoRA tensors all differed from the original SFT tensors, consistent with an effective update.

**Full-data 10-step gate command:**

```bash
HF_HOME=/root/.cache/huggingface HF_HUB_CACHE=/root/.cache/huggingface/hub python training/grpo/train.py \
  --sft-adapter artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter \
  --run-name grpo-qwen35-2b-10step --max-steps 10 --save-steps 10 --eval-steps 50
```

The trainer reached exactly 10 optimizer steps on the full 810-record train data. Its logged training values are below; component cells give mean / standard deviation. No entropy key was emitted by this TRL/Unsloth version.

| Step | Loss | LR | Reward mean / std | Zero-variance groups | Strict JSON | Valid tariffs | Calibrated Top-1 | Calibrated ranking | KL | Mean completion | Clip ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.000001107 | 0 | 0.992569 / 0.008497 | 0 | 1 / 0 | 1 / 0 | 1 / 0 | 0.981422 / 0.021243 | 0.092026 | 41.5 | 0 |
| 5 | 0.000010661 | 0.00000375 | 0.983182 / 0.005383 | 0.25 | 1 / 0 | 1 / 0 | 1 / 0 | 0.957954 / 0.013458 | 0.070697 | 41.4375 | 0 |
| 10 | 0.000023085 | 0.0000001508 | 0.989264 / 0.011815 | 0 | 1 / 0 | 1 / 0 | 0.998464 / 0.003073 | 0.974697 / 0.026465 | 0.047199 | 41.7 | 0 |

All logged numeric values were finite. Training reward did not collapse, reward variance persisted, KL did not explode, mean completion length stayed near 41–42, completion clipping was 0, and the remaining clip-ratio keys (`low_mean`, `low_min`, `high_mean`, `high_max`, `region_mean`) were all 0 at logged training steps. The first held-out rollout evaluation at step 10 reported reward `0.984518`, reward standard deviation `0.015627`, zero-variance group fraction `0.188889`, strict/valid means `1/1`, calibrated Top-1 mean/std `0.988148/0.017136`, calibrated ranking mean/std `0.973147/0.022421`, KL `0.066947`, mean completion `41.256`, clip ratio 0, and eval loss `0.000021534`. A second final evaluation reported reward `0.989378`, reward standard deviation `0.010284`, zero-variance group fraction `0.222222`, strict/valid means `1/1`, calibrated Top-1 mean/std `0.996197/0.007606`, calibrated ranking mean/std `0.977247/0.018337`, mean completion `41.261`, clip ratio 0, and loss `0.000020833`. The two evaluations sample stochastic rollouts, so their reward values are not a deterministic before/after measurement.

Train loss averaged `0.000015918`; train runtime was `317.739` seconds, the run-record training interval `319.386` seconds, and peak GPU memory `6.927 GiB`. The process ran from `13:10:01` to `13:18:54` UTC, including two held-out evaluations (232.286 and 212.540 seconds). `checkpoints/checkpoint-10/` and `final_adapter/` were saved under `artifacts/training/runs/grpo-qwen35-2b-10step/`, alongside `run_record.json`, `resolved_config.json`, and TensorBoard events. The final adapter directory SHA-256 was `474543cab81b246c90b2b7c16bf79f714e7e4cdbedbba5c5826269106be25954`.

**Evaluation command:**

```bash
HF_HOME=/root/.cache/huggingface HF_HUB_CACHE=/root/.cache/huggingface/hub python training/run_eval.py \
  --stage grpo --adapter artifacts/training/runs/grpo-qwen35-2b-10step/final_adapter \
  --dataset artifacts/splits/grpo_validation.jsonl --limit 90 \
  --run-name grpo-validation-10step --max-seq-length 2048 --max-new-tokens 700
```

The evaluation used the same untouched 90-example split, deterministic generation, thinking disabled, and the stored **supplied per-quote reward**, which is distinct from the calibrated GRPO training reward. Its 90 prediction IDs are unique and match the SFT baseline and validation split in order.

| Metric | Frozen SFT | GRPO 10-step | Change |
| --- | ---: | ---: | ---: |
| Top-1 correct / accuracy | 83/90; 0.922222 | 84/90; 0.933333 | +1 scenario; +0.011111 |
| Top-3 recall | 0.870370 | 0.862963 | -0.007407 |
| Strict JSON | 90/90; 1.0 | 90/90; 1.0 | 0 |
| Valid three unique tariffs | 90/90; 1.0 | 90/90; 1.0 | 0 |
| Valid Top-1 tariff | 90/90; 1.0 | 90/90; 1.0 | 0 |
| Supplied reward ratio | 0.856095 | 0.859726 | +0.003631 |
| Mean completion tokens | 43.267 | 43.322 | +0.055 |

Only one scenario changed Top-1 correctness, from incorrect SFT pick `49` to correct GRPO pick `90`; ten scenarios changed their full Top-3 ordering. The small Top-3 decline is a regression signal to retain in any later decision, even though Top-1 and supplied reward rose. Generation took `168.987` seconds after a `14.590`-second model load, with throughput `0.532586` recommendations/second. Outputs: `artifacts/training/evaluations/grpo-validation-10step/predictions.jsonl` and `summary.json`; frozen baseline outputs remain at `artifacts/training/evaluations/grpo-validation-sft-baseline/`.

**Compatibility and decision.** Installed TRL 0.24.0 ignored `chat_template_kwargs` and `num_generations_eval`, so the trainer renders the chat template with thinking disabled before passing prompts to TRL, removes unsupported settings, and sets the prompt limit to 1920 so complete prompts fit alongside 128 output tokens. Unsloth is imported before TRL for its patching order; the tokenizer is unwrapped from the loaded processor. The installed trainer still evaluated at the final step despite `--eval-steps 50`, then the script performed a duplicate explicit final evaluation. The script was adjusted afterward to reuse a final-step evaluation if already present; that change did not alter this completed run's weights or metrics. Unsloth emitted backend deprecation warnings. A separate exploratory pilot was stopped at step 0 and produced no trained adapter. The one-step and ten-step gates support considering a 50-step experiment as a controlled ablation, but the gain is small and Top-3 recall fell slightly, so these results do not establish a broadly better model. **No run longer than 10 steps was started. No 50-step run or GRPO golden evaluation was launched. Golden data was not used. Teacher labels were absent from GRPO training. Reward weights were not recalibrated. The original SFT adapter was preserved.** Further training and golden evaluation await user approval.

## GRPO 10-Step Golden Evaluation

**Date:** 2026-09-26 UTC. After the conservative gate, the user explicitly requested the golden evaluation of the saved 10-step GRPO adapter. This updates the earlier gate's statement that no GRPO golden evaluation had yet been launched. The evaluated adapter was `artifacts/training/runs/grpo-qwen35-2b-10step/final_adapter`, directory SHA-256 `474543cab81b246c90b2b7c16bf79f714e7e4cdbedbba5c5826269106be25954`; the SFT source adapter remains untouched. Preflight verified the 50-example golden file, SHA-256 `864123d2f1c2741f94cfef94ebdd09ecc6eb6d1b3fe898212441ad99b54eaa33`, and an unused output name.

```bash
HF_HOME=/root/.cache/huggingface HF_HUB_CACHE=/root/.cache/huggingface/hub python training/run_eval.py \
  --stage grpo \
  --adapter artifacts/training/runs/grpo-qwen35-2b-10step/final_adapter \
  --run-name golden-grpo-qwen35-2b-10step-700tok \
  --max-seq-length 2048 --max-new-tokens 700 --confirm-golden-eval
```

Generation was deterministic with thinking disabled and BF16 on the NVIDIA RTX A6000, using `Qwen/Qwen3.5-2B` revision `15852e8c16360a2fea060d615a32b45270f8a8fc`. The settings matched the prior SFT golden evaluation. The supplied per-quote reward stored in the golden dataset was used for the reported reward ratio; no calibrated GRPO reward weights were applied during evaluation.

| Metric | Golden SFT | Golden GRPO 10-step | Change |
| --- | ---: | ---: | ---: |
| Top-1 correct / accuracy | 42/50; 0.84 | 42/50; 0.84 | 0 |
| Top-3 recall | 0.866667 | 0.860000 | -0.006667 |
| Strict JSON | 50/50; 1.0 | 50/50; 1.0 | 0 |
| Valid three unique tariffs | 50/50; 1.0 | 50/50; 1.0 | 0 |
| Valid Top-1 tariff | 50/50; 1.0 | 50/50; 1.0 | 0 |
| Supplied reward ratio | 0.811724 | 0.811724 | 0 |
| Mean completion tokens | 43.300 | 43.320 | +0.020 |

All 50 predictions have unique scenario IDs matching the golden file and earlier SFT evaluation in order. No Top-1 correctness changed. Only `wg-042` changed its full ranking: SFT predicted `[49,48,162]`, matching the teacher's Top-3, while GRPO predicted `[49,162,161]`; that row's Top-3 recall fell from 1 to 2/3 and accounts for the aggregate decline. Its first pick remained `49`. The GRPO run had generation time `96.762` seconds, model-load time `13.817` seconds, and throughput `0.516732` recommendations/second. The earlier SFT run's timing is not a controlled speed comparison. Unsloth emitted backend deprecation warnings, without affecting completion.

Artifacts: `artifacts/training/evaluations/golden-grpo-qwen35-2b-10step-700tok/predictions.jsonl` and `summary.json`. This completed golden comparison shows no measured Top-1 or supplied-reward improvement over SFT and a small Top-3 regression on one scenario. No additional training was run, and the golden result was not used to change weights or settings.
