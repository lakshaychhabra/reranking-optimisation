#!/usr/bin/env python3
"""Fine-tune Qwen3.5-2B on the prepared WG ranking data with BF16/FP16 LoRA."""
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


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.common import DEFAULT_MODEL, load_jsonl, percentile, sha256_file, write_json  # noqa: E402


DEFAULT_DATA_DIR = ROOT / "artifacts" / "training" / "sft_data"
DEFAULT_RUNS_DIR = ROOT / "artifacts" / "training" / "runs"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _package_versions() -> dict[str, str | None]:
    packages = ("torch", "unsloth", "unsloth_zoo", "transformers", "trl", "peft", "datasets", "accelerate")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result


def _validate_prepared(path: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    seen: set[str] = set()
    for index, row in enumerate(rows, 1):
        scenario_id = str(row.get("scenario_id", ""))
        if not scenario_id or scenario_id in seen:
            raise SystemExit(f"{path}:{index}: missing or duplicate scenario_id")
        seen.add(scenario_id)
        prompt, completion = row.get("prompt"), row.get("completion")
        if not isinstance(prompt, list) or [item.get("role") for item in prompt] != ["system", "user"]:
            raise SystemExit(f"{path}:{index}: expected system/user conversational prompt")
        if not isinstance(completion, list) or len(completion) != 1 or completion[0].get("role") != "assistant":
            raise SystemExit(f"{path}:{index}: expected one assistant completion")
        try:
            target = json.loads(completion[0]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"{path}:{index}: invalid JSON completion") from exc
        top = target.get("top_3") if isinstance(target, dict) else None
        ids = [item.get("tariff_id") for item in top] if isinstance(top, list) else []
        if (
            not isinstance(target, dict)
            or set(target) != {"top_3"}
            or not isinstance(top, list)
            or [item.get("rank") for item in top if isinstance(item, dict)] != [1, 2, 3]
            or any(not isinstance(tariff_id, str) for tariff_id in ids)
            or len(ids) != 3
            or len(set(ids)) != 3
        ):
            raise SystemExit(f"{path}:{index}: invalid top_3 target")
    return rows


def _token_audit(tokenizer: Any, rows: list[dict[str, Any]], max_length: int, split: str) -> dict[str, Any]:
    lengths: list[int] = []
    too_long: list[dict[str, Any]] = []
    for row in rows:
        conversation = row["prompt"] + row["completion"]
        token_ids = tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            enable_thinking=False,
        )
        length = len(token_ids)
        lengths.append(length)
        if length > max_length:
            too_long.append({"scenario_id": row["scenario_id"], "tokens": length})
    result = {
        "split": split,
        "records": len(rows),
        "tokens": {
            "min": min(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
        },
        "max_length": max_length,
        "over_limit": too_long,
    }
    if too_long:
        worst = sorted(too_long, key=lambda item: item["tokens"], reverse=True)[:5]
        raise SystemExit(
            f"{split}: {len(too_long)} examples exceed --max-length={max_length}; "
            f"increase it rather than truncating completion targets. Worst: {worst}"
        )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--train-data", default=str(DEFAULT_DATA_DIR / "train.jsonl"))
    parser.add_argument("--validation-data", default=str(DEFAULT_DATA_DIR / "validation.jsonl"))
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS_DIR))
    parser.add_argument("--run-name", default="sft-qwen35-2b-seed3407")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--train-batch-size", type=int, default=24)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--smoke-test", action="store_true", help="Run a small training subset (one optimizer step by default).")
    parser.add_argument("--smoke-steps", type=int, default=1, help="Optimizer steps in smoke mode (default: 1).")
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="Validate prepared JSONL without GPU imports.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.smoke_steps < 1 or (not args.smoke_test and args.smoke_steps != 1):
        raise SystemExit("--smoke-steps must be positive and is only configurable with --smoke-test")
    train_path = Path(args.train_data).resolve()
    validation_path = Path(args.validation_data).resolve()
    for path in (train_path, validation_path):
        if (ROOT / "data").resolve() == path or (ROOT / "data").resolve() in path.parents:
            raise SystemExit("training data must not come from the held-out data/ directory")
    train_rows = _validate_prepared(train_path)
    validation_rows = _validate_prepared(validation_path)
    overlap = {row["scenario_id"] for row in train_rows} & {row["scenario_id"] for row in validation_rows}
    if overlap:
        raise SystemExit(f"training and validation overlap: {len(overlap)} scenarios")

    offline_validation = {
        "train_records": len(train_rows),
        "validation_records": len(validation_rows),
        "train_sha256": sha256_file(train_path),
        "validation_sha256": sha256_file(validation_path),
        "disjoint": True,
    }
    if args.validate_only:
        print(json.dumps(offline_validation, indent=2))
        return 0

    try:
        import torch
        from datasets import Dataset
        from unsloth import FastLanguageModel
        from transformers import TrainerCallback
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(
            "GPU training dependencies are missing. Install training/requirements.txt on the CUDA machine. "
            f"Original error: {exc}"
        ) from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU not detected; refusing to start this training run")

    bf16 = bool(torch.cuda.is_bf16_supported())
    dtype = torch.bfloat16 if bf16 else torch.float16
    precision = "bf16" if bf16 else "fp16"
    output_dir = Path(args.runs_dir).resolve() / args.run_name
    if output_dir.exists() and any(output_dir.iterdir()) and not (args.overwrite_output or args.resume_from_checkpoint):
        raise SystemExit(
            f"run directory is not empty: {output_dir}; use a new --run-name, --resume-from-checkpoint, "
            "or explicitly pass --overwrite-output"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        revision=args.model_revision,
        max_seq_length=args.max_length,
        dtype=dtype,
        load_in_4bit=False,
        load_in_16bit=True,
        full_finetuning=False,
    )
    # Qwen3.5 may return a vision processor; this SFT dataset is text-only.
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=LORA_TARGETS,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        max_seq_length=args.max_length,
    )
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable_parameters == 0:
        raise SystemExit("LoRA attached no trainable parameters")

    audit = {
        "train": _token_audit(tokenizer, train_rows, args.max_length, "train"),
        "validation": _token_audit(tokenizer, validation_rows, args.max_length, "validation"),
    }
    smoke_train_records = min(
        len(train_rows),
        max(2, args.train_batch_size * args.gradient_accumulation_steps * args.smoke_steps),
    )
    resolved = {
        "stage": "sft",
        "run_name": args.run_name,
        "started_at": started_at,
        "model": args.model,
        "model_revision_requested": args.model_revision,
        "model_revision_resolved": getattr(model.config, "_commit_hash", None),
        "precision": precision,
        "dataset": offline_validation,
        "token_audit": audit,
        "trainable_parameters": trainable_parameters,
        "records_used": {
            "train": smoke_train_records if args.smoke_test else len(train_rows),
            "validation": min(2, len(validation_rows)) if args.smoke_test else len(validation_rows),
        },
        "hyperparameters": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "lr_scheduler_type": "constant" if args.smoke_test else "cosine",
            "warmup_ratio": 0.0 if args.smoke_test else 0.1,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.train_batch_size * args.gradient_accumulation_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0,
            "lora_targets": LORA_TARGETS,
            "seed": args.seed,
            "completion_only_loss": True,
            "smoke_test": args.smoke_test,
            "smoke_steps": args.smoke_steps if args.smoke_test else None,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
        },
    }
    write_json(output_dir / "resolved_config.json", resolved)

    train_dataset = Dataset.from_list(train_rows[:smoke_train_records] if args.smoke_test else train_rows).remove_columns("scenario_id")
    validation_dataset = Dataset.from_list(validation_rows[:2] if args.smoke_test else validation_rows).remove_columns("scenario_id")
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(output_dir / "tensorboard")
    training_args = SFTConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.smoke_steps if args.smoke_test else -1,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant" if args.smoke_test else "cosine",
        warmup_ratio=0.0 if args.smoke_test else 0.1,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=not bf16,
        max_length=args.max_length,
        completion_only_loss=True,
        packing=False,
        eval_packing=False,
        eval_strategy="steps" if args.smoke_test else "epoch",
        eval_steps=args.smoke_steps if args.smoke_test else None,
        save_strategy="steps" if args.smoke_test else "epoch",
        save_steps=args.smoke_steps if args.smoke_test else 500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=1 if args.smoke_test else args.logging_steps,
        logging_first_step=True,
        report_to=["tensorboard"],
        run_name=args.run_name,
        optim="adamw_torch",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=True,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
    )

    class StepTimer(TrainerCallback):
        def __init__(self) -> None:
            self.started: float | None = None
            self.seconds: list[float] = []

        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self.started = time.monotonic()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if self.started is not None:
                self.seconds.append(round(time.monotonic() - self.started, 3))
                self.started = None

    step_timer = StepTimer()
    trainer.add_callback(step_timer)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    elapsed = time.monotonic() - started
    evaluation = trainer.evaluate()

    adapter_dir = output_dir / "final_adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()

    summary = {
        **resolved,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 3),
        "step_seconds": step_timer.seconds,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / (1024**3), 3),
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
        "log_history": trainer.state.log_history,
        "artifacts": {
            "adapter": str(adapter_dir),
            "trainer_state": str(output_dir / "checkpoints" / "trainer_state.json"),
            "tensorboard": str(output_dir / "tensorboard"),
        },
    }
    write_json(output_dir / "run_record.json", summary)
    print(json.dumps({
        "status": "complete",
        "run": str(output_dir),
        "adapter": str(adapter_dir),
        "elapsed_seconds": summary["elapsed_seconds"],
        "peak_gpu_memory_gb": summary["peak_gpu_memory_gb"],
        "train_metrics": train_result.metrics,
        "validation_metrics": evaluation,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
