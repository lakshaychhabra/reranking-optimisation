#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible vLLM server on frozen non-golden WG prompts."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import random
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUT = "release/data/grpo/validation.jsonl"
DEFAULT_OUTPUT_ROOT = "artifacts/serving_benchmark"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def rounded(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("prompt"), list):
                raise ValueError(f"record {line_number} in {path} has no conversational prompt")
            records.append(record)
    if not records:
        raise ValueError(f"no records in {path}")
    return records


def parse_top3(text: str, valid_ids: set[str]) -> tuple[bool, bool, list[str]]:
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return False, False, []
    rows = payload.get("top_3") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return True, False, []
    ids = [str(row.get("tariff_id")) for row in rows if isinstance(row, dict) and row.get("tariff_id") is not None]
    valid = len(ids) == 3 and len(set(ids)) == 3 and set(ids) <= valid_ids
    return True, valid, ids


@dataclass
class TelemetrySample:
    elapsed_seconds: float
    gpu_utilization_pct: float
    memory_used_mib: float
    memory_total_mib: float
    power_draw_w: float
    sm_clock_mhz: float
    temperature_c: float


class NvidiaSampler:
    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.samples: list[TelemetrySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = 0.0

    def start(self) -> None:
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[TelemetrySample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_seconds * 3)
        return self.samples

    def _run(self) -> None:
        query = (
            "utilization.gpu,memory.used,memory.total,power.draw,"
            "clocks.sm,temperature.gpu"
        )
        while not self._stop.is_set():
            try:
                completed = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--query-gpu={query}",
                        "--format=csv,noheader,nounits",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                values = [float(value.strip()) for value in completed.stdout.strip().split(",")]
                if len(values) == 6:
                    self.samples.append(
                        TelemetrySample(
                            elapsed_seconds=time.monotonic() - self._started,
                            gpu_utilization_pct=values[0],
                            memory_used_mib=values[1],
                            memory_total_mib=values[2],
                            power_draw_w=values[3],
                            sm_clock_mhz=values[4],
                            temperature_c=values[5],
                        )
                    )
            except (FileNotFoundError, subprocess.SubprocessError, ValueError):
                pass
            self._stop.wait(self.interval_seconds)


def telemetry_summary(samples: list[TelemetrySample], duration_seconds: float) -> dict[str, Any]:
    def summarize(attribute: str) -> dict[str, float | None]:
        values = [float(getattr(sample, attribute)) for sample in samples]
        return {
            "mean": rounded(mean(values), 3),
            "p50": rounded(percentile(values, 0.50), 3),
            "p95": rounded(percentile(values, 0.95), 3),
            "max": rounded(max(values), 3) if values else None,
        }

    power = [sample.power_draw_w for sample in samples]
    average_power = mean(power)
    return {
        "samples": len(samples),
        "gpu_utilization_pct": summarize("gpu_utilization_pct"),
        "memory_used_mib": summarize("memory_used_mib"),
        "memory_total_mib": rounded(samples[0].memory_total_mib, 3) if samples else None,
        "power_draw_w": summarize("power_draw_w"),
        "sm_clock_mhz": summarize("sm_clock_mhz"),
        "temperature_c": summarize("temperature_c"),
        "estimated_energy_wh": rounded(average_power * duration_seconds / 3600, 4) if average_power else None,
    }


def request_chat(
    endpoint: str,
    model: str,
    record: dict[str, Any],
    max_tokens: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    scenario_id = str(record.get("scenario_id", "unknown"))
    payload = {
        "model": model,
        "messages": record["prompt"],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        latency = time.monotonic() - started
        content = str(body["choices"][0]["message"]["content"]).strip()
        usage = body.get("usage") or {}
        valid_ids = {str(value) for value in record.get("reward_by_tariff_id", {})}
        strict_json, valid_schema, top3 = parse_top3(content, valid_ids)
        return {
            "scenario_id": scenario_id,
            "ok": True,
            "latency_seconds": latency,
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "strict_json": strict_json,
            "valid_top3": valid_schema,
            "top3": top3,
            "content": content,
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "error": None,
        }
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
        return {
            "scenario_id": scenario_id,
            "ok": False,
            "latency_seconds": time.monotonic() - started,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "strict_json": False,
            "valid_top3": False,
            "top3": [],
            "content": None,
            "content_sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def select_workload(records: list[dict[str, Any]], requests: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    while len(selected) < requests:
        block = records.copy()
        rng.shuffle(block)
        selected.extend(block)
    return selected[:requests]


def benchmark_level(
    *,
    endpoint: str,
    model: str,
    records: list[dict[str, Any]],
    concurrency: int,
    max_tokens: int,
    timeout_seconds: float,
    gpu_hourly_usd: float,
    usd_per_eur: float,
    telemetry_interval_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[TelemetrySample]]:
    sampler = NvidiaSampler(telemetry_interval_seconds)
    sampler.start()
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(request_chat, endpoint, model, record, max_tokens, timeout_seconds)
            for record in records
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    duration = time.monotonic() - started
    samples = sampler.stop()
    successful = [result for result in results if result["ok"]]
    latencies = [float(result["latency_seconds"]) for result in successful]
    prompt_tokens = sum(int(result["prompt_tokens"]) for result in successful)
    completion_tokens = sum(int(result["completion_tokens"]) for result in successful)
    requests_per_second = len(successful) / duration if duration else 0.0
    gpu_eur_per_hour = gpu_hourly_usd / usd_per_eur
    eur_per_1k = gpu_eur_per_hour * 1000 / (3600 * requests_per_second) if requests_per_second else None
    summary = {
        "concurrency": concurrency,
        "requests": len(results),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
        "duration_seconds": rounded(duration, 4),
        "requests_per_second": rounded(requests_per_second, 6),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "prompt_tokens_per_second": rounded(prompt_tokens / duration if duration else None, 3),
        "completion_tokens_per_second": rounded(completion_tokens / duration if duration else None, 3),
        "total_tokens_per_second": rounded((prompt_tokens + completion_tokens) / duration if duration else None, 3),
        "mean_prompt_tokens": rounded(prompt_tokens / len(successful) if successful else None, 3),
        "mean_completion_tokens": rounded(completion_tokens / len(successful) if successful else None, 3),
        "latency_seconds": {
            "mean": rounded(mean(latencies), 4),
            "p50": rounded(percentile(latencies, 0.50), 4),
            "p95": rounded(percentile(latencies, 0.95), 4),
            "p99": rounded(percentile(latencies, 0.99), 4),
            "max": rounded(max(latencies), 4) if latencies else None,
        },
        "strict_json_rate": rounded(sum(result["strict_json"] for result in successful) / len(successful) if successful else None),
        "valid_top3_rate": rounded(sum(result["valid_top3"] for result in successful) / len(successful) if successful else None),
        "gpu_hourly_usd": gpu_hourly_usd,
        "usd_per_eur": usd_per_eur,
        "cost_usd_for_measured_window": rounded(gpu_hourly_usd * duration / 3600, 6),
        "saturated_cost_eur_per_1k": rounded(eur_per_1k, 6),
        "cost_eur_per_1k_at_utilization": {
            "100_pct": rounded(eur_per_1k, 6),
            "50_pct": rounded(eur_per_1k / 0.5 if eur_per_1k else None, 6),
            "25_pct": rounded(eur_per_1k / 0.25 if eur_per_1k else None, 6),
            "10_pct": rounded(eur_per_1k / 0.10 if eur_per_1k else None, 6),
        },
        "telemetry": telemetry_summary(samples, duration),
    }
    return summary, results, samples


def get_server_models(endpoint: str, timeout_seconds: float) -> dict[str, Any]:
    with urllib.request.urlopen(endpoint.rstrip("/") + "/v1/models", timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="wg-sft")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--concurrency", default="1,2,4,8,16,32")
    parser.add_argument("--requests-per-level", type=int, default=90)
    parser.add_argument("--warmup-requests", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    parser.add_argument("--telemetry-interval-seconds", type=float, default=0.5)
    parser.add_argument("--gpu-hourly-usd", type=float, required=True)
    parser.add_argument("--usd-per-eur", type=float, default=1.1403)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    input_path = (repo / args.input).resolve() if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = (repo / args.output_root / args.run_name).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.requests_per_level <= 0 or args.warmup_requests < 0:
        raise SystemExit("request counts must be positive (warmups may be zero)")
    concurrency_levels = [int(value) for value in args.concurrency.split(",")]
    if not concurrency_levels or any(value <= 0 for value in concurrency_levels):
        raise SystemExit("concurrency must be a comma-separated list of positive integers")

    records = load_jsonl(input_path)
    server_models = get_server_models(args.endpoint, args.timeout_seconds)
    warmups = select_workload(records, args.warmup_requests, args.seed + 10_000)
    warmup_results = [
        request_chat(args.endpoint, args.model, record, args.max_tokens, args.timeout_seconds)
        for record in warmups
    ]
    if warmup_results and not all(result["ok"] for result in warmup_results):
        errors = [result["error"] for result in warmup_results if not result["ok"]]
        raise SystemExit(f"warm-up failed: {errors[:3]}")

    summaries: list[dict[str, Any]] = []
    all_results: list[dict[str, Any]] = []
    telemetry_rows: list[dict[str, Any]] = []
    for index, concurrency in enumerate(concurrency_levels):
        workload = select_workload(records, args.requests_per_level, args.seed + index)
        print(f"benchmarking concurrency={concurrency} requests={len(workload)}", flush=True)
        summary, results, samples = benchmark_level(
            endpoint=args.endpoint,
            model=args.model,
            records=workload,
            concurrency=concurrency,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            gpu_hourly_usd=args.gpu_hourly_usd,
            usd_per_eur=args.usd_per_eur,
            telemetry_interval_seconds=args.telemetry_interval_seconds,
        )
        summaries.append(summary)
        all_results.extend({"concurrency": concurrency, **result} for result in results)
        telemetry_rows.extend({"concurrency": concurrency, **sample.__dict__} for sample in samples)
        print(json.dumps(summary, sort_keys=True), flush=True)

    best = max(summaries, key=lambda row: float(row["requests_per_second"]))
    resolved_config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "endpoint": args.endpoint,
        "model": args.model,
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "input_records": len(records),
        "concurrency": concurrency_levels,
        "requests_per_level": args.requests_per_level,
        "warmup_requests": args.warmup_requests,
        "max_tokens": args.max_tokens,
        "timeout_seconds": args.timeout_seconds,
        "telemetry_interval_seconds": args.telemetry_interval_seconds,
        "gpu_hourly_usd": args.gpu_hourly_usd,
        "usd_per_eur": args.usd_per_eur,
        "server_models": server_models,
    }
    summary_payload = {
        "run_name": args.run_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "best_throughput": best,
        "levels": summaries,
        "cost_scope": (
            "GPU runtime at the configured hourly price only; excludes model download, startup, storage, "
            "idle capacity, network, taxes, and operational overhead. Utilization scenarios divide saturated cost by utilization."
        ),
    }
    (output_dir / "resolved_config.json").write_text(json.dumps(resolved_config, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as handle:
        for result in all_results:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
    if telemetry_rows:
        with (output_dir / "gpu_telemetry.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(telemetry_rows[0]))
            writer.writeheader()
            writer.writerows(telemetry_rows)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
