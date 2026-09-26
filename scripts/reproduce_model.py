#!/usr/bin/env python3
"""Plan or execute the pinned Qwen3.5-2B SFT -> optional GRPO workflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RELEASE = ROOT / "release"
MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MODEL_WEIGHT = "model.safetensors-00001-of-00001.safetensors"
MODEL_WEIGHT_SHA256 = "aa33250c4fc64891ddfaba3a314fd9542ea371843c387178b425fbcc5ed680b1"
IMPORTED_SFT = RELEASE / "model" / "sft-final"
IMPORTED_GRPO = RELEASE / "model" / "grpo-10step"
RELEASE_SFT_DATA = RELEASE / "data" / "sft"
RELEASE_GRPO_DATA = RELEASE / "data" / "grpo"
RELEASE_SFT_EVAL_DATA = RELEASE / "data" / "eval" / "sft_validation.jsonl"
RELEASE_EVAL_DATA = RELEASE / "data" / "eval" / "grpo_validation.jsonl"
IMPORTED_ADAPTER_HASHES = {
    "sft": "f3bbd9330f353c038ce3d9e2338c76b6f5cd4b036ee81cd23bc7ed510c66bf48",
    "grpo": "474543cab81b246c90b2b7c16bf79f714e7e4cdbedbba5c5826269106be25954",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(file.relative_to(path)).encode())
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _command_text(command: list[str]) -> str:
    return " ".join(command)


def _run(command: list[str], execute: bool) -> None:
    print(f"$ {_command_text(command)}", flush=True)
    if execute:
        subprocess.run(command, cwd=ROOT, check=True)


def _download_model(model_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required; install training/requirements.txt") from exc
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(model_dir),
    )
    weight = model_dir / MODEL_WEIGHT
    if not weight.is_file():
        raise SystemExit(f"downloaded snapshot is missing {weight}")
    actual = _sha256(weight)
    if actual != MODEL_WEIGHT_SHA256:
        raise SystemExit(f"base-model weight checksum mismatch: expected {MODEL_WEIGHT_SHA256}, got {actual}")


def _verify_imported_adapters() -> dict[str, str]:
    paths = {"sft": IMPORTED_SFT, "grpo": IMPORTED_GRPO}
    verified: dict[str, str] = {}
    for stage, path in paths.items():
        if not path.is_dir():
            raise SystemExit(f"imported {stage} adapter is missing: {path}")
        actual = _directory_hash(path)
        expected = IMPORTED_ADAPTER_HASHES[stage]
        if actual != expected:
            raise SystemExit(f"imported {stage} adapter checksum mismatch: expected {expected}, got {actual}")
        verified[stage] = actual
    return verified


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=("verify-imported", "download", "sft", "sft-grpo", "evaluate-imported"),
        default="verify-imported",
    )
    parser.add_argument("--seed", type=int, default=42, help="Use 3407 to reproduce the recorded GPU runs.")
    parser.add_argument("--model-dir", default=str(ARTIFACTS / "models" / "Qwen3.5-2B"))
    parser.add_argument("--output-root", default=str(ARTIFACTS / "reproduction"))
    parser.add_argument("--run-prefix", default="replica")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--confirm-golden-eval",
        action="store_true",
        help="Run the locked golden evaluation after training. Omit for normal reproduction/dev checks.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute downloads/GPU commands. Without this flag, print and save the resolved plan only.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model_dir = Path(args.model_dir).resolve()
    output_root = Path(args.output_root).resolve()
    runs_dir = output_root / "runs"
    evaluations_dir = output_root / "evaluations"
    sft_run_name = f"{args.run_prefix}-sft-seed{args.seed}"
    grpo_run_name = f"{args.run_prefix}-grpo-10step-seed{args.seed}"
    sft_adapter = runs_dir / sft_run_name / "final_adapter"
    grpo_adapter = runs_dir / grpo_run_name / "final_adapter"

    if args.action == "verify-imported":
        verified = _verify_imported_adapters()
        print(json.dumps({"status": "verified", "adapter_hashes": verified}, indent=2))
        return 0

    commands: list[list[str]] = []
    python = sys.executable
    if args.action in {"sft", "sft-grpo"}:
        sft_command = [
            python,
            "training/train_sft.py",
            "--model",
            str(model_dir),
            "--runs-dir",
            str(runs_dir),
            "--train-data",
            str(RELEASE_SFT_DATA / "train.jsonl"),
            "--validation-data",
            str(RELEASE_SFT_DATA / "validation.jsonl"),
            "--run-name",
            sft_run_name,
            "--train-batch-size",
            "48",
            "--eval-batch-size",
            "2",
            "--gradient-accumulation-steps",
            "1",
            "--seed",
            str(args.seed),
        ]
        if args.smoke_test:
            sft_command.append("--smoke-test")
        commands.append(sft_command)
        commands.append([
            python,
            "training/run_eval.py",
            "--stage",
            "sft",
            "--base-model",
            str(model_dir),
            "--adapter",
            str(sft_adapter),
            "--dataset",
            str(RELEASE_SFT_EVAL_DATA),
            "--limit",
            "90",
            "--output-root",
            str(evaluations_dir),
            "--run-name",
            f"{sft_run_name}-validation",
        ])

        if args.action == "sft-grpo":
            commands.append([
                python,
                "training/run_eval.py",
                "--stage",
                "sft",
                "--base-model",
                str(model_dir),
                "--adapter",
                str(sft_adapter),
                "--dataset",
                str(RELEASE_EVAL_DATA),
                "--limit",
                "90",
                "--output-root",
                str(evaluations_dir),
                "--run-name",
                f"{sft_run_name}-grpo-validation-baseline",
            ])

    if args.action == "sft-grpo":
        grpo_command = [
            python,
            "training/grpo/train.py",
            "--sft-adapter",
            str(sft_adapter),
            "--runs-dir",
            str(runs_dir),
            "--train-data",
            str(RELEASE_GRPO_DATA / "train.jsonl"),
            "--validation-data",
            str(RELEASE_GRPO_DATA / "validation.jsonl"),
            "--manifest",
            str(RELEASE_GRPO_DATA / "manifest.json"),
            "--run-name",
            grpo_run_name,
            "--max-steps",
            "10",
            "--save-steps",
            "10",
            "--eval-steps",
            "50",
            "--seed",
            str(args.seed),
        ]
        if args.smoke_test:
            grpo_command.append("--smoke-test")
        commands.append(grpo_command)
        commands.append([
            python,
            "training/run_eval.py",
            "--stage",
            "grpo",
            "--base-model",
            str(model_dir),
            "--adapter",
            str(grpo_adapter),
            "--dataset",
            str(RELEASE_EVAL_DATA),
            "--limit",
            "90",
            "--output-root",
            str(evaluations_dir),
            "--run-name",
            f"{grpo_run_name}-validation",
        ])

    if args.action == "evaluate-imported":
        _verify_imported_adapters()
        for stage, adapter, dataset in (
            ("sft", IMPORTED_SFT, RELEASE_EVAL_DATA),
            ("grpo", IMPORTED_GRPO, RELEASE_EVAL_DATA),
        ):
            commands.append([
                python,
                "training/run_eval.py",
                "--stage",
                stage,
                "--base-model",
                str(model_dir),
                "--adapter",
                str(adapter),
                "--dataset",
                str(dataset),
                "--limit",
                "90",
                "--output-root",
                str(evaluations_dir),
                "--run-name",
                f"imported-{stage}-validation",
            ])

    selected_adapter = grpo_adapter if args.action == "sft-grpo" else sft_adapter
    selected_stage = "grpo" if args.action == "sft-grpo" else "sft"
    if args.confirm_golden_eval and args.action in {"sft", "sft-grpo"}:
        commands.append([
            python,
            "training/run_eval.py",
            "--stage",
            selected_stage,
            "--base-model",
            str(model_dir),
            "--adapter",
            str(selected_adapter),
            "--output-root",
            str(evaluations_dir),
            "--run-name",
            f"{args.run_prefix}-golden-{selected_stage}-seed{args.seed}",
            "--confirm-golden-eval",
        ])

    plan = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "action": args.action,
        "execute": args.execute,
        "seed": args.seed,
        "exact_recorded_seed": 3407,
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "local_dir": str(model_dir),
            "weight_sha256": MODEL_WEIGHT_SHA256,
        },
        "output_root": str(output_root),
        "smoke_test": args.smoke_test,
        "golden_eval": args.confirm_golden_eval,
        "commands": commands,
    }
    _write_json(output_root / "resolved_config.json", plan)
    print(json.dumps(plan, indent=2))

    if args.execute and args.action in {"download", "sft", "sft-grpo", "evaluate-imported"} and not args.skip_download:
        _download_model(model_dir)
    elif args.execute and args.action != "download":
        weight = model_dir / MODEL_WEIGHT
        if not weight.is_file():
            raise SystemExit(f"base model is missing: {weight}; omit --skip-download")

    if args.action != "download":
        for command in commands:
            _run(command, args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
