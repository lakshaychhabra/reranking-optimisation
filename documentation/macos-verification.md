# Apple Silicon verification

The selected SFT adapter was independently reproduced on an Apple M4 MacBook
Air using the portable Transformers backend added to `training/run_eval.py`.
This is a portability check; the frozen NVIDIA RTX A6000 result under `results/`
remains the canonical reported evaluation.

## Environment

- macOS 26.6.2, Apple Silicon MPS
- Python 3.11.14 in Conda environment `work`
- PyTorch 2.8.0
- Transformers 5.5.0
- PEFT 0.21.0
- Accelerate 1.15.0
- BF16, greedy generation, thinking disabled

The base model was read from the already downloaded local
`Qwen/Qwen3.5-2B` snapshot. Its 4.55 GB weight file SHA-256 was verified as
`aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1`.
The released SFT adapter directory SHA-256 was verified as
`f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48`.

## Command

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 conda run -n work \
  python training/run_eval.py \
  --backend transformers \
  --stage sft \
  --base-model tmp/imported-runpod/artifacts/models/Qwen3.5-2B \
  --model-revision 15852e8c16360a2fea060d615a32b45270f8a8fc \
  --adapter release/model/sft-final \
  --dataset data/golden_wg_recommendations.jsonl \
  --limit 50 \
  --max-seq-length 2048 \
  --max-new-tokens 700 \
  --output-root artifacts/macos_verification \
  --run-name golden-sft-mps-reproduction \
  --confirm-golden-eval
```

## Result

| Metric | Frozen CUDA | Mac MPS | Match |
|---|---:|---:|---:|
| Top-1 correct | 42/50 | 42/50 | yes |
| Top-1 accuracy | 0.84 | 0.84 | yes |
| Top-3 recall | 0.866667 | 0.866667 | yes |
| Strict JSON rate | 1.0 | 1.0 | yes |
| Valid three-tariff rate | 1.0 | 1.0 | yes |
| Reward ratio | 0.811724 | 0.811724 | yes |
| Mean prompt tokens | 995.0 | 995.0 | yes |
| Mean completion tokens | 43.3 | 43.3 | yes |

All 50 rows matched on `scenario_id`, exact `raw_output`, ranked Top-3, first
pick, teacher Top-3, correctness, Top-3 recall, strict-JSON validity, offered-ID
validity, rewards, reward ratio, prompt tokens, and completion tokens. There
were zero substantive mismatches.

The complete JSONL SHA-256 differs because each row records backend-dependent
`generation_seconds`. Mac MPS generated the 50 recommendations in 323.184
seconds (0.154711 recommendations/second); timing is not used as a quality
metric and is not expected to match the CUDA run.
