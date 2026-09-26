#!/usr/bin/env python3
"""Build the tracked, minimal model/data release used by a fresh Git clone."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "release"

FILE_MAPPINGS = (
    ("artifacts/training/sft_data/train.jsonl", "data/sft/train.jsonl"),
    ("artifacts/training/sft_data/validation.jsonl", "data/sft/validation.jsonl"),
    ("artifacts/training/sft_data/manifest.json", "data/sft/manifest.json"),
    ("artifacts/training/grpo_data/train.jsonl", "data/grpo/train.jsonl"),
    ("artifacts/training/grpo_data/validation.jsonl", "data/grpo/validation.jsonl"),
    ("artifacts/training/grpo_data/manifest.json", "data/grpo/manifest.json"),
    ("artifacts/splits/sft_validation.jsonl", "data/eval/sft_validation.jsonl"),
    ("artifacts/splits/grpo_validation.jsonl", "data/eval/grpo_validation.jsonl"),
    ("artifacts/reward_calibration/best_weights.json", "configs/reward/best_weights.json"),
    ("artifacts/reward_calibration/calibration_summary.json", "configs/reward/calibration_summary.json"),
    (
        "artifacts/training/runs/sft-qwen35-2b-seed3407/resolved_config.json",
        "configs/sft/resolved_config.json",
    ),
    (
        "artifacts/training/runs/sft-qwen35-2b-seed3407/run_record.json",
        "configs/sft/run_record.json",
    ),
    (
        "artifacts/training/runs/grpo-qwen35-2b-10step/resolved_config.json",
        "configs/grpo/resolved_config.json",
    ),
    (
        "artifacts/training/runs/grpo-qwen35-2b-10step/run_record.json",
        "configs/grpo/run_record.json",
    ),
    (
        "artifacts/training/evaluations/golden-base-qwen35-2b-700tok/summary.json",
        "evaluation/base-golden-summary.json",
    ),
    (
        "artifacts/training/evaluations/golden-sft-qwen35-2b-seed3407-700tok/summary.json",
        "evaluation/sft-golden-summary.json",
    ),
    (
        "artifacts/training/evaluations/golden-grpo-qwen35-2b-10step-700tok/summary.json",
        "evaluation/grpo-golden-summary.json",
    ),
    (
        "artifacts/training/evaluations/grpo-validation-sft-baseline/summary.json",
        "evaluation/sft-grpo-validation-summary.json",
    ),
    (
        "artifacts/training/evaluations/grpo-validation-10step/summary.json",
        "evaluation/grpo-validation-summary.json",
    ),
)

DIRECTORY_MAPPINGS = (
    (
        "artifacts/training/runs/sft-qwen35-2b-seed3407/final_adapter",
        "model/sft-final",
    ),
    (
        "artifacts/training/runs/grpo-qwen35-2b-10step/final_adapter",
        "model/grpo-10step",
    ),
)

FORBIDDEN_PATH_PARTS = {
    ".env",
    "cache",
    "checkpoints",
    "raw",
    "requests",
    "runpod",
    "teacher",
    "tmp",
}
SECRET_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)authorization\s*[:=]\s*bearer\s+[A-Za-z0-9._-]{12,}"),
    re.compile(r"(?i)(?:anthropic_api_key|mm_id|mm_pa)\s*[:=]\s*['\"]?[^\s'\"]+"),
)
TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".txt", ".jinja"}

RELEASE_README = """# WG recommendation release

This directory is the tracked, portable release for the take-home submission.
It contains the selected SFT LoRA adapter, the optional ten-step GRPO adapter,
prepared non-golden training data, one development evaluation split, aggregate
evaluation summaries, and exact run configurations.

The base model is not duplicated. Both adapters use `Qwen/Qwen3.5-2B` at
revision `15852e8c16360a2fea060d615a32b45270f8a8fc`; the repository's
`scripts/reproduce_model.py` downloads and verifies that base checkpoint.

Start with `HOW_TO_RUN.md` in the repository root. `MANIFEST.json` records the
source and SHA-256 of every released file, while `SHA256SUMS` supports an
independent integrity check.

No golden rows, per-scenario golden predictions, live quote cache, provider raw
responses, credentials, base-model weights, or optimizer checkpoints are in
this release. Aggregate golden summaries are included as final report evidence.
"""

DATASET_README = """---
pretty_name: WG Recommendation Training and Development Data
language:
- de
license: other
task_categories:
- text-generation
tags:
- insurance
- recommendation
- sft
- grpo
- qwen
size_categories:
- 1K<n<10K
configs:
- config_name: sft
  data_files:
  - split: train
    path: sft/train.jsonl
  - split: validation
    path: sft/validation.jsonl
- config_name: grpo
  data_files:
  - split: train
    path: grpo/train.jsonl
  - split: validation
    path: grpo/validation.jsonl
- config_name: eval_sft
  data_files:
  - split: validation
    path: eval/sft_validation.jsonl
- config_name: eval_grpo
  data_files:
  - split: validation
    path: eval/grpo_validation.jsonl
---

# WG Recommendation Training and Development Data

This dataset supports reproducible fine-tuning of a small language model to rank
German Wohngebäude (residential-building insurance) tariffs. Given a building
risk, risk context, and a shuffled set of priced quotes, the model returns a
ranked top three in strict JSON.

It contains the prepared data used for supervised fine-tuning (SFT), the optional
GRPO continuation, and development-set evaluation. It does **not** contain the
held-out 50-scenario golden benchmark, golden labels, or per-scenario golden
predictions.

## Configurations and splits

| Configuration | Split | Rows | Purpose |
|---|---|---:|---|
| `sft` | `train` | 300 | Conversational prompt/completion examples for SFT |
| `sft` | `validation` | 90 | SFT checkpoint selection and development monitoring |
| `grpo` | `train` | 810 | Prompts with frozen per-tariff business rewards |
| `grpo` | `validation` | 90 | GRPO development monitoring; teacher ranking retained offline |
| `eval_sft` | `validation` | 90 | Raw development scenarios corresponding to SFT validation |
| `eval_grpo` | `validation` | 90 | Raw development scenarios corresponding to GRPO validation |

Load a configuration with `datasets`:

```python
from datasets import load_dataset

sft = load_dataset("lakshaychhabra/wg-recommendation-data", "sft")
grpo = load_dataset("lakshaychhabra/wg-recommendation-data", "grpo")
```

## Record formats

### SFT

- `scenario_id`: deterministic identifier for the normalized risk.
- `prompt`: TRL conversational messages created by the canonical prompt builder.
- `completion`: assistant message containing the ranked top-three target.

Teacher rationales are not included in the target.

### GRPO

- `scenario_id`: deterministic scenario identifier.
- `prompt`: TRL conversational messages using the same canonical prompt format.
- `reward_by_tariff_id`: frozen business reward for each valid tariff.
- `teacher_top3`: validation-only field for offline monitoring; absent from
  training and never passed to the GRPO reward function.

### Development evaluation

The `eval_*` configurations retain the structured risk, requested coverage,
quotes, risk context, and development teacher annotation needed to run the
repository evaluator. These are development records, not the golden benchmark.

## Data construction and leakage controls

- Scenarios were generated independently from the held-out benchmark, priced
  through the cached quote-engine client, and split deterministically by
  `scenario_id`.
- Training and validation are disjoint within each training stage.
- SFT targets contain three distinct tariff IDs that exist in their quote sets.
- GRPO training contains no teacher labels; rewards were precomputed from frozen
  reward weights.
- The manifests record row counts, prompt statistics, source hashes, reward
  configuration, and validation checks.
- The golden benchmark was used only for final aggregate evaluation and is not
  distributed here.

## Reproducibility

The companion project contains the canonical prompt builder, preparation
scripts, training entry points, exact resolved configurations, adapter weights,
and evaluation commands. Use the repository's `HOW_TO_RUN.md` as the primary
runbook. The selected result is a Qwen3.5-2B SFT LoRA with 0.84 top-1 agreement
on the held-out benchmark; the GRPO continuation is included as an optional
ablation and was not selected.

## Scope and license

This dataset is published for reproducing the associated take-home experiment.
It is not a general insurance-advice dataset and must not be used as a substitute
for qualified brokerage, underwriting, or legal review. The repository does not
grant additional rights over third-party tariff or quote-engine data; downstream
users are responsible for confirming that their use is permitted.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    yield from sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def _validate_destination(relative: Path) -> None:
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe release destination: {relative}")
    lowered = {part.lower() for part in relative.parts}
    if lowered & FORBIDDEN_PATH_PARTS:
        raise ValueError(f"forbidden release path: {relative}")
    if "golden" in relative.name.lower() and relative.parts[0] != "evaluation":
        raise ValueError(f"golden data is forbidden outside aggregate evaluation summaries: {relative}")


def audit_release(root: Path) -> None:
    for path in _iter_files(root):
        relative = path.relative_to(root)
        _validate_destination(relative)
        if path.is_symlink():
            raise ValueError(f"release contains a symlink: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if relative.parts[:2] == ("data", "eval") or relative.parts[:1] == ("data",):
            if "fable_annotation" in text:
                raise ValueError(f"release data contains a held-out Fable field: {relative}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                raise ValueError(f"release text matches a credential pattern: {relative}")


def verify_checksums(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    if not checksum_path.is_file():
        raise ValueError(f"release checksum file is missing: {checksum_path}")
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, relative_name = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid SHA256SUMS line {line_number}") from exc
        relative = Path(relative_name)
        _validate_destination(relative)
        path = root / relative
        if not path.is_file():
            raise ValueError(f"release file listed in SHA256SUMS is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"release checksum mismatch for {relative}: expected {expected}, got {actual}")


def _copy_sources(staging: Path) -> dict[str, str]:
    sources: dict[str, str] = {}
    for source_name, destination_name in FILE_MAPPINGS:
        source = ROOT / source_name
        destination_relative = Path(destination_name)
        _validate_destination(destination_relative)
        if not source.is_file():
            raise FileNotFoundError(f"required release source is missing: {source}")
        destination = staging / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sources[str(destination_relative)] = source_name
    for source_name, destination_name in DIRECTORY_MAPPINGS:
        source = ROOT / source_name
        destination_relative = Path(destination_name)
        _validate_destination(destination_relative)
        if not source.is_dir():
            raise FileNotFoundError(f"required release source is missing: {source}")
        for source_file in _iter_files(source):
            relative_file = destination_relative / source_file.relative_to(source)
            _validate_destination(relative_file)
            destination = staging / relative_file
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            sources[str(relative_file)] = str(source_file.relative_to(ROOT))
    return sources


def _write_metadata(staging: Path, sources: dict[str, str], seed: int) -> None:
    (staging / "README.md").write_text(RELEASE_README, encoding="utf-8")
    sources["README.md"] = "generated by scripts/create_release_bundle.py"
    data_readme = staging / "data" / "README.md"
    data_readme.parent.mkdir(parents=True, exist_ok=True)
    data_readme.write_text(DATASET_README, encoding="utf-8")
    sources["data/README.md"] = "generated by scripts/create_release_bundle.py"
    entries = []
    for path in _iter_files(staging):
        relative = str(path.relative_to(staging))
        entries.append({
            "path": relative,
            "source": sources[relative],
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest: dict[str, Any] = {
        "release_version": "2026-09-26",
        "builder": "scripts/create_release_bundle.py",
        "seed": seed,
        "base_model": "Qwen/Qwen3.5-2B",
        "base_model_revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
        "selected_model": "model/sft-final",
        "optional_model": "model/grpo-10step",
        "golden_rows_included": False,
        "entries": entries,
    }
    manifest_path = staging / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checksum_files = [*list(_iter_files(staging))]
    checksums = "".join(
        f"{sha256_file(path)}  {path.relative_to(staging)}\n"
        for path in checksum_files
        if path.name != "SHA256SUMS"
    )
    (staging / "SHA256SUMS").write_text(checksums, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Replace an existing release directory.")
    parser.add_argument("--verify-only", action="store_true", help="Audit an existing release without rebuilding it.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = Path(args.output).resolve()
    if output == ROOT or ROOT not in output.parents:
        raise SystemExit(f"release output must be a child of the repository: {output}")
    if args.verify_only:
        if not output.is_dir():
            raise SystemExit(f"release directory not found: {output}")
        audit_release(output)
        verify_checksums(output)
        print(json.dumps({"status": "verified", "release": str(output)}, indent=2))
        return 0
    if output.exists() and not args.force:
        raise SystemExit(f"release already exists: {output}; pass --force to rebuild it")

    temporary = Path(tempfile.mkdtemp(prefix=".release-build-", dir=ROOT))
    try:
        sources = _copy_sources(temporary)
        audit_release(temporary)
        _write_metadata(temporary, sources, args.seed)
        audit_release(temporary)
        if output.exists():
            shutil.rmtree(output)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps({
        "status": "created",
        "release": str(output),
        "files": sum(1 for _ in _iter_files(output)),
        "bytes": sum(path.stat().st_size for path in _iter_files(output)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
