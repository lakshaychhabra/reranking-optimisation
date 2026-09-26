#!/usr/bin/env python3
"""Evaluate base, SFT, or GRPO Qwen checkpoints on one fixed WG dataset."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BROKER_SYSTEM, build_broker_prompt  # noqa: E402
from training.common import (  # noqa: E402
    DEFAULT_MODEL,
    GOLDEN_PATH,
    annotation_top3,
    load_jsonl,
    parse_top3,
    record_id,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "training" / "evaluations"


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for package in ("torch", "unsloth", "unsloth_zoo", "transformers", "peft"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(file for file in path.rglob("*") if file.is_file())
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _validate_dataset(records: list[dict[str, Any]], path: Path) -> None:
    seen: set[str] = set()
    for record in records:
        scenario_id = record_id(record)
        if scenario_id in seen:
            raise SystemExit(f"duplicate scenario ID in {path}: {scenario_id}")
        seen.add(scenario_id)
        if len(annotation_top3(record)) != 3:
            raise SystemExit(f"scenario {scenario_id} has no valid teacher/Fable top_3")
        quotes = record.get("quotes")
        if not isinstance(quotes, list) or len(quotes) < 3:
            raise SystemExit(f"scenario {scenario_id} has fewer than three quotes")
        for quote in quotes:
            if quote.get("tariff_id") is None or not isinstance(quote.get("reward"), (int, float)):
                raise SystemExit(f"scenario {scenario_id} has a quote without tariff_id/default reward")


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("base", "sft", "grpo"), required=True)
    parser.add_argument(
        "--backend",
        choices=("auto", "unsloth", "transformers"),
        default="auto",
        help="auto uses Unsloth on CUDA when installed, otherwise portable Transformers inference.",
    )
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--adapter", help="LoRA adapter directory; required for sft/grpo stages.")
    parser.add_argument("--dataset", default=str(GOLDEN_PATH))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # The supplied Qwen evaluator allows 700 output tokens. Expected JSON is much
    # shorter, but this avoids truncating an untuned model's verbose response.
    # Base, SFT, and GRPO evaluations must share this ceiling.
    parser.add_argument("--max-new-tokens", type=int, default=700)
    parser.add_argument("--gpu-hourly-cost", type=float, help="Optional USD/hour used for an inference cost estimate.")
    parser.add_argument("--confirm-golden-eval", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate dataset and arguments without loading a model.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    dataset_path = Path(args.dataset).resolve()
    golden = dataset_path == GOLDEN_PATH.resolve()
    if golden and not args.confirm_golden_eval and not args.validate_only:
        raise SystemExit(
            "this is the locked golden dataset; rerun with --confirm-golden-eval after fixing the checkpoint. "
            "Do not use golden results for hyperparameter or checkpoint selection."
        )
    if args.stage in {"sft", "grpo"} and not args.adapter:
        raise SystemExit(f"--adapter is required for --stage {args.stage}")
    if args.stage == "base" and args.adapter:
        raise SystemExit("--adapter is not allowed for --stage base")

    records = load_jsonl(dataset_path)
    _validate_dataset(records, dataset_path)
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    records = records[: args.limit]
    adapter_path = Path(args.adapter).resolve() if args.adapter else None
    if adapter_path is not None and not adapter_path.exists():
        raise SystemExit(f"adapter not found: {adapter_path}")

    preflight = {
        "stage": args.stage,
        "base_model": args.base_model,
        "model_revision": args.model_revision,
        "adapter": str(adapter_path) if adapter_path else None,
        "dataset": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "scenarios": len(records),
        "golden": golden,
        "reward_definition": "supplied per-quote reward already stored in the evaluation dataset; no calibrated weights applied",
        "backend_requested": args.backend,
    }
    if args.validate_only:
        print(json.dumps(preflight, indent=2))
        return 0

    output_dir = Path(args.output_root).resolve() / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        raise SystemExit(f"evaluation directory is not empty: {output_dir}; choose a new --run-name")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(f"PyTorch is required for evaluation: {exc}") from exc

    unsloth_available = importlib.util.find_spec("unsloth") is not None
    if args.backend == "auto":
        backend = "unsloth" if torch.cuda.is_available() and unsloth_available else "transformers"
    else:
        backend = args.backend

    load_started = time.monotonic()
    if backend == "unsloth":
        if not torch.cuda.is_available():
            raise SystemExit("the Unsloth backend requires an NVIDIA CUDA GPU")
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise SystemExit(f"Unsloth backend requested but unavailable: {exc}") from exc
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        device = torch.device("cuda")
        load_target = str(adapter_path) if adapter_path else args.base_model
        load_kwargs: dict[str, Any] = {
            "model_name": load_target,
            "max_seq_length": args.max_seq_length,
            "dtype": dtype,
            "load_in_4bit": False,
            "load_in_16bit": True,
        }
        if adapter_path is None:
            load_kwargs["revision"] = args.model_revision
        model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        FastLanguageModel.for_inference(model)
    else:
        try:
            from peft import PeftModel
            from transformers import AutoModelForMultimodalLM, AutoTokenizer
        except ImportError as exc:
            raise SystemExit(
                "portable evaluation dependencies are missing; install "
                f"training/requirements-macos-eval.txt. Original error: {exc}"
            ) from exc
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.bfloat16
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32
        tokenizer_target = str(adapter_path) if adapter_path else args.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_target,
            revision=args.model_revision if adapter_path is None else None,
        )
        model = AutoModelForMultimodalLM.from_pretrained(
            args.base_model,
            revision=args.model_revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        )
        if adapter_path is not None:
            model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
        model.to(device)
        model.eval()
    model_load_seconds = time.monotonic() - load_started

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    predictions: list[dict[str, Any]] = []
    total_generation_seconds = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for index, record in enumerate(records, 1):
        scenario_id = record_id(record)
        prompt_text = build_broker_prompt(record, include_rationales=False)
        messages = [
            {"role": "system", "content": BROKER_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        if prompt_tokens > args.max_seq_length:
            raise SystemExit(
                f"scenario {scenario_id} has {prompt_tokens} prompt tokens, exceeding "
                f"--max-seq-length={args.max_seq_length}"
            )
        minimum_required_length = prompt_tokens + args.max_new_tokens
        if minimum_required_length > args.max_seq_length:
            raise SystemExit(
                f"scenario {scenario_id}: {prompt_tokens} prompt tokens + "
                f"{args.max_new_tokens} requested output tokens require a minimum sequence "
                f"length of {minimum_required_length}, exceeding configured "
                f"--max-seq-length={args.max_seq_length}"
            )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        synchronize()
        started = time.monotonic()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        synchronize()
        generation_seconds = time.monotonic() - started
        generated_ids = outputs[0][prompt_tokens:]
        completion_tokens = int(generated_ids.shape[-1])
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        predicted, strict_json = parse_top3(text)

        quote_ids = {str(quote["tariff_id"]) for quote in record["quotes"]}
        reward_by_id = {str(quote["tariff_id"]): float(quote["reward"]) for quote in record["quotes"]}
        teacher_ids = annotation_top3(record)
        pick = predicted[0] if predicted else None
        valid_pick = pick in quote_ids if pick is not None else False
        valid_ranking = len(predicted) == 3 and len(set(predicted)) == 3 and set(predicted) <= quote_ids
        argmax_reward = max(reward_by_id.values())
        pick_reward = reward_by_id.get(pick) if valid_pick else None
        reward_ratio = pick_reward / argmax_reward if pick_reward is not None and argmax_reward else 0.0

        predictions.append({
            "scenario_id": scenario_id,
            "stage": args.stage,
            "model_top3": predicted,
            "model_pick": pick,
            "teacher_top3": teacher_ids,
            "top1_hit": bool(pick == teacher_ids[0]),
            "top3_recall": len(set(predicted) & set(teacher_ids)) / 3.0,
            "strict_json": strict_json,
            "valid_top1_tariff": valid_pick,
            "valid_three_unique_tariffs": valid_ranking,
            "pick_reward": pick_reward,
            "argmax_reward": argmax_reward,
            "reward_ratio": reward_ratio,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "generation_seconds": round(generation_seconds, 6),
            "raw_output": text,
        })
        total_generation_seconds += generation_seconds
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        print(f"[{index:02d}/{len(records):02d}] {scenario_id[:10]} pick={pick} strict={strict_json}")

    scenario_count = len(predictions)
    valid_reward_ratios = [row["reward_ratio"] for row in predictions if row["valid_top1_tariff"]]
    summary: dict[str, Any] = {
        **preflight,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "backend": backend,
        "run_name": args.run_name,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "adapter_sha256": _directory_hash(adapter_path) if adapter_path else None,
        "model_load_seconds": round(model_load_seconds, 3),
        "generation_seconds": round(total_generation_seconds, 3),
        "metrics": {
            "top1_accuracy": round(sum(row["top1_hit"] for row in predictions) / scenario_count, 6),
            "top1_correct": sum(row["top1_hit"] for row in predictions),
            "top3_recall": _mean([row["top3_recall"] for row in predictions]),
            "strict_json_rate": round(sum(row["strict_json"] for row in predictions) / scenario_count, 6),
            "valid_top1_tariff_rate": round(sum(row["valid_top1_tariff"] for row in predictions) / scenario_count, 6),
            "valid_three_unique_tariffs_rate": round(
                sum(row["valid_three_unique_tariffs"] for row in predictions) / scenario_count, 6
            ),
            "reward_ratio": _mean([row["reward_ratio"] for row in predictions]),
            "reward_ratio_valid_picks_only": _mean(valid_reward_ratios),
            "mean_prompt_tokens": round(total_prompt_tokens / scenario_count, 3),
            "mean_completion_tokens": round(total_completion_tokens / scenario_count, 3),
            "recommendations_per_second": round(scenario_count / total_generation_seconds, 6),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "cuda": torch.version.cuda,
            "accelerator": device.type,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
            "precision": (
                "bf16" if dtype == torch.bfloat16 else "fp16" if dtype == torch.float16 else "fp32"
            ),
        },
        "generation": {
            "do_sample": False,
            "thinking": False,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_length": args.max_seq_length,
        },
    }
    if args.gpu_hourly_cost is not None:
        inference_cost = total_generation_seconds / 3600 * args.gpu_hourly_cost
        summary["cost"] = {
            "gpu_hourly_cost_usd": args.gpu_hourly_cost,
            "measured_generation_cost_usd": round(inference_cost, 6),
            "cost_per_1000_recommendations_usd": round(inference_cost / scenario_count * 1000, 6),
            "model_load_time_excluded": True,
        }

    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(output_dir), **summary["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
