#!/usr/bin/env python3
"""Align an SFT Qwen3.5-2B adapter with the frozen WG reward using GRPO."""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common import load_jsonl, percentile, sha256_file, write_json  # noqa: E402
from training.grpo.rewards import reward_functions  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "artifacts" / "training" / "grpo_data"
DEFAULT_RUNS_DIR = ROOT / "artifacts" / "training" / "runs"
DEFAULT_REWARD_WEIGHTS = (0.10, 0.10, 0.40, 0.40)


def _versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "unsloth",
        "unsloth_zoo",
        "transformers",
        "trl",
        "peft",
        "datasets",
        "accelerate",
        "bitsandbytes",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _validate_rows(path: Path, training: bool) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    seen: set[str] = set()
    for line_number, row in enumerate(rows, 1):
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id or scenario_id in seen:
            raise SystemExit(f"{path}:{line_number}: missing or duplicate scenario_id")
        seen.add(scenario_id)
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or [item.get("role") for item in prompt] != ["system", "user"]:
            raise SystemExit(f"{path}:{line_number}: expected a system/user conversational prompt")
        scores = row.get("reward_by_tariff_id")
        if (
            not isinstance(scores, dict)
            or len(scores) < 3
            or any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in scores.values())
        ):
            raise SystemExit(f"{path}:{line_number}: invalid reward_by_tariff_id")
        teacher = row.get("teacher_top3")
        if training and teacher is not None:
            raise SystemExit(f"{path}:{line_number}: teacher labels are forbidden in GRPO training data")
        if not training and teacher is not None:
            if not isinstance(teacher, list) or len(teacher) != 3 or len(set(teacher)) != 3:
                raise SystemExit(f"{path}:{line_number}: invalid validation teacher_top3")
    return rows


def _token_audit(tokenizer: Any, rows: list[dict[str, Any]], max_length: int, max_completion: int) -> dict[str, Any]:
    lengths: list[int] = []
    for row in rows:
        token_ids = tokenizer.apply_chat_template(
            row["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
            enable_thinking=False,
        )
        lengths.append(len(token_ids))
    over = sum(length + max_completion > max_length for length in lengths)
    result = {
        "records": len(rows),
        "prompt_tokens": {
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "max_sequence_length": max_length,
        "max_completion_length": max_completion,
        "prompts_without_completion_margin": over,
    }
    if over:
        raise SystemExit(
            f"{over} prompts plus --max-completion-length exceed --max-sequence-length={max_length}; "
            "increase the sequence ceiling rather than silently truncating prompts"
        )
    return result


def _render_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render chat prompts once so TRL cannot change the thinking setting."""
    return [
        {
            **row,
            "prompt": tokenizer.apply_chat_template(
                row["prompt"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            ),
        }
        for row in rows
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-adapter", required=True, help="Path to the completed SFT LoRA adapter.")
    parser.add_argument("--train-data", default=str(DEFAULT_DATA_DIR / "train.jsonl"))
    parser.add_argument("--validation-data", default=str(DEFAULT_DATA_DIR / "validation.jsonl"))
    parser.add_argument("--manifest", default=str(DEFAULT_DATA_DIR / "manifest.json"))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--run-name", default="grpo-qwen35-2b-seed3407")
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument("--max-completion-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--beta", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument(
        "--component-weights",
        type=float,
        nargs=4,
        default=DEFAULT_REWARD_WEIGHTS,
        metavar=("FORMAT", "VALIDITY", "TOP1", "RANKING"),
    )
    parser.add_argument("--smoke-test", action="store_true", help="Run a small subset for exactly one update.")
    parser.add_argument("--allow-long-run", action="store_true", help="Allow more than 100 optimizer updates.")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate files without GPU imports.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.max_steps <= 0 or args.num_generations < 2:
        raise SystemExit("--max-steps must be positive and --num-generations must be at least two")
    if args.max_steps > 100 and not args.smoke_test and not args.allow_long_run:
        raise SystemExit("more than 100 GRPO steps requires --allow-long-run")
    if args.max_completion_length <= 0 or args.max_sequence_length <= args.max_completion_length:
        raise SystemExit("--max-completion-length must be positive and shorter than --max-sequence-length")
    if args.train_batch_size <= 0 or args.eval_batch_size <= 0 or args.gradient_accumulation_steps <= 0:
        raise SystemExit("train/eval batch sizes and gradient accumulation must be positive")
    effective_batch = args.train_batch_size * args.gradient_accumulation_steps
    if effective_batch % args.num_generations:
        raise SystemExit(
            "train batch size * gradient accumulation must be divisible by --num-generations; "
            f"got {effective_batch} and {args.num_generations}"
        )
    if len(args.component_weights) != 4 or any(weight < 0 for weight in args.component_weights):
        raise SystemExit("--component-weights requires four non-negative values")

    train_path = Path(args.train_data).resolve()
    validation_path = Path(args.validation_data).resolve()
    manifest_path = Path(args.manifest).resolve()
    adapter_path = Path(args.sft_adapter).resolve()
    protected = (ROOT / "data").resolve()
    if any(path == protected or protected in path.parents for path in (train_path, validation_path)):
        raise SystemExit("GRPO must not train or validate on the locked data/ directory")
    if not adapter_path.exists():
        raise SystemExit(f"SFT adapter not found: {adapter_path}")
    if not manifest_path.exists():
        raise SystemExit(f"GRPO data manifest not found: {manifest_path}")

    train_rows = _validate_rows(train_path, training=True)
    validation_rows = _validate_rows(validation_path, training=False)
    overlap = {row["scenario_id"] for row in train_rows} & {row["scenario_id"] for row in validation_rows}
    if overlap:
        raise SystemExit(f"GRPO train/validation overlap: {len(overlap)} scenarios")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_train = manifest.get("splits", {}).get("train", {}).get("output", {}).get("sha256")
    expected_validation = manifest.get("splits", {}).get("validation", {}).get("output", {}).get("sha256")
    if expected_train and expected_train != sha256_file(train_path):
        raise SystemExit("GRPO train file does not match its manifest checksum")
    if expected_validation and expected_validation != sha256_file(validation_path):
        raise SystemExit("GRPO validation file does not match its manifest checksum")

    preflight = {
        "stage": "grpo",
        "sft_adapter": str(adapter_path),
        "data": {
            "train": str(train_path),
            "train_records": len(train_rows),
            "train_sha256": sha256_file(train_path),
            "validation": str(validation_path),
            "validation_records": len(validation_rows),
            "validation_sha256": sha256_file(validation_path),
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "disjoint": True,
            "teacher_labels_in_train": False,
        },
        "reward": {
            "frozen_business_reward": manifest.get("reward_source"),
            "component_names": [function.__name__ for function in reward_functions()],
            "component_weights": list(args.component_weights),
            "component_weight_policy": "fixed before GRPO; do not tune on GRPO validation or golden data",
        },
        "hyperparameters": {
            "max_steps": 1 if args.smoke_test else args.max_steps,
            "max_sequence_length": args.max_sequence_length,
            "max_completion_length": args.max_completion_length,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "constant" if args.smoke_test else "cosine",
            "warmup_ratio": 0.0 if args.smoke_test else 0.1,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": effective_batch,
            "num_generations": args.num_generations,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "beta": args.beta,
            "loss_type": "dr_grpo",
            "scale_rewards": False,
            "use_vllm": False,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "allow_long_run": args.allow_long_run,
            "prompt_format": "rendered chat template with thinking disabled",
            "log_completions": args.smoke_test,
        },
    }
    if args.validate_only:
        print(json.dumps(preflight, indent=2))
        return 0

    try:
        import torch
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise SystemExit(
            "GRPO dependencies are missing. Audit/install training/requirements.txt on the CUDA machine. "
            f"Original error: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected; refusing to start GRPO")

    full_train_rows, full_validation_rows = train_rows, validation_rows
    if args.smoke_test:
        train_rows = train_rows[: max(args.train_batch_size, args.num_generations)]
        validation_rows = validation_rows[: args.eval_batch_size]

    output_dir = Path(args.runs_dir).resolve() / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.overwrite_output or args.resume_from_checkpoint):
        raise SystemExit(
            f"run directory is not empty: {output_dir}; choose a new --run-name, resume, "
            "or explicitly pass --overwrite-output"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    bf16 = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else torch.float16
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_sequence_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    if trainable == 0:
        raise SystemExit("the loaded SFT adapter has zero trainable parameters; do not start GRPO")
    token_audit = {
        "train": _token_audit(tokenizer, full_train_rows, args.max_sequence_length, args.max_completion_length),
        "validation": _token_audit(
            tokenizer, full_validation_rows, args.max_sequence_length, args.max_completion_length
        ),
    }
    resolved = {
        **preflight,
        "run_name": args.run_name,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "token_audit": token_audit,
        "records_used": {"train": len(train_rows), "validation": len(validation_rows)},
        "model": {
            "resolved_base_model": getattr(model.config, "_name_or_path", None),
            "resolved_revision": getattr(model.config, "_commit_hash", None),
            "trainable_parameters": trainable,
            "total_parameters": total,
            "trainable_percentage": round(trainable / total * 100, 6),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _versions(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "precision": "bf16" if bf16 else "fp16",
        },
    }
    write_json(output_dir / "resolved_config.json", resolved)

    train_dataset = Dataset.from_list(_render_prompts(tokenizer, train_rows))
    validation_dataset = Dataset.from_list(_render_prompts(tokenizer, validation_rows))
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "tensorboard")
    config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=1 if args.smoke_test else args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant" if args.smoke_test else "cosine",
        warmup_ratio=0.0 if args.smoke_test else 0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        max_prompt_length=args.max_sequence_length - args.max_completion_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        reward_weights=list(args.component_weights),
        scale_rewards=False,
        loss_type="dr_grpo",
        mask_truncated_completions=True,
        use_vllm=False,
        eval_strategy="steps",
        eval_steps=1 if args.smoke_test else args.eval_steps,
        save_strategy="steps",
        save_steps=1 if args.smoke_test else args.save_steps,
        save_total_limit=5,
        logging_strategy="steps",
        logging_steps=1 if args.smoke_test else args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tensorboard"),
        run_name=args.run_name,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        log_completions=args.smoke_test,
        num_completions_to_print=4,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_functions(),
        args=config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    last_step_eval = next(
        (
            entry for entry in reversed(trainer.state.log_history)
            if entry.get("step") == trainer.state.global_step and "eval_loss" in entry
        ),
        None,
    )
    evaluation = last_step_eval if last_step_eval is not None else trainer.evaluate()
    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()

    result = {
        **resolved,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
        "log_history": trainer.state.log_history,
        "artifacts": {
            "adapter": str(adapter_dir),
            "checkpoints": str(output_dir / "checkpoints"),
            "tensorboard": str(output_dir / "tensorboard"),
        },
    }
    write_json(output_dir / "run_record.json", result)
    print(json.dumps({
        "status": "complete",
        "run": str(output_dir),
        "adapter": str(adapter_dir),
        "elapsed_seconds": result["elapsed_seconds"],
        "peak_gpu_memory_gb": result["peak_gpu_memory_gb"],
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
