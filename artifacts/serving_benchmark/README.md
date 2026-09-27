# Qwen3.5-2B SFT serving benchmark

Production-style inference was measured with the selected SFT LoRA adapter on one secure-cloud
RunPod RTX 4090 at `$0.74/hour`, using vLLM 0.29.0, BF16, a 2,048-token context ceiling, and the
frozen non-golden GRPO validation prompts. The model generated strict top-3 JSON with thinking
disabled. The golden set was not used.

## Planning result

- Recommended `price_assumption_eur_per_mtok`: **€0.010640** at **25% fleet utilization**.
- Equivalent recommendation cost: **€0.011327 per 1,000 recommendations**.
- Saturated concurrency-64 result: **63.66 recommendations/s**, **€0.002832/1k**, p50 **1.003 s**,
  p95 **1.105 s**, mean GPU utilization **90.7%**.
- Single-request/no-batching result: **3.34 recommendations/s**, **€0.053899/1k**, p50 **0.295 s**.
- All 3,600 sustained requests succeeded and produced valid top-3 JSON.

The workload averaged about 1,065 total tokens per recommendation. Effective token cost is
computed as:

```text
EUR / Mtok = (EUR / 1k recommendations) * 1,000 / tokens per recommendation
```

For self-hosted serving this is an *effective infrastructure rate*, not a vendor token tariff.
It depends on batching and fleet utilization:

| Fleet utilization | EUR / 1k recommendations | Effective EUR / Mtok |
|---:|---:|---:|
| 100% | 0.002832 | 0.002660 |
| 50% | 0.005663 | 0.005320 |
| 25% (planning) | 0.011327 | 0.010640 |
| 10% | 0.028317 | 0.026601 |
| One request at a time | 0.053899 | 0.050632 |

These costs include GPU runtime only. They exclude startup/download time, storage, network,
taxes, monitoring, failover headroom, and engineering/operations.

## GPU memory and replicas

vLLM measured 3.91 GiB for weights plus non-PyTorch allocations, 0.40 GiB peak activation, and
0.14 GiB CUDA-graph memory. On 23.52 GiB usable VRAM this gives a hard ceiling of five engine
footprints before any KV cache, or an estimated four independent instances if each receives at
least 1 GiB of KV cache. That is a capacity estimate, not a tested four-server deployment.

Multiple copies are not the cost-saving topology here. A single continuously batched vLLM engine
already demonstrated 64 concurrent requests while using the remaining 16.85 GiB as KV cache.
Extra copies duplicate weights and reduce cache capacity; use replicas for isolation or high
availability, and use one batching engine per GPU for throughput economics.

## Artifacts

- `qwen35-2b-sft-vllm-rtx4090-20260926/`: concurrency 1, 2, 4, 8, 16, 32, and 64 sweep.
- `qwen35-2b-sft-vllm-rtx4090-sustained-20260926/`: 1,200 requests at concurrency 16, 32, and 64.
- `environment/`: server log, exact dependency lock snapshot, hardware metadata, model checksum,
  Prometheus metrics, and the benchmark client used on the pod.

Each run contains `resolved_config.json`, `summary.json`, request-level JSONL, and sampled GPU
telemetry CSV.
