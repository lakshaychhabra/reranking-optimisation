#!/usr/bin/env python3
"""Create a credential-free RunPod bundle for base evaluation, SFT, and GRPO."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "runpod" / "wg-training-runpod.tar.gz"

DIRECTORIES = (
    "training",
    "src",
    "scripts",
    "tools",
    "reward_calibration",
    "documentation",
)
ROOT_FILES = (
    ".gitignore",
    "README.md",
    "BENCHMARK.md",
    "PROBLEM_FRAMING.md",
    "DECISIONS.md",
    "replicate.md",
)
DATA_FILES = (
    "data/golden_wg_recommendations.jsonl",
    "artifacts/training/sft_data/train.jsonl",
    "artifacts/training/sft_data/validation.jsonl",
    "artifacts/training/sft_data/manifest.json",
    "artifacts/splits/sft_train.jsonl",
    "artifacts/splits/sft_validation.jsonl",
    "artifacts/splits/grpo_train.jsonl",
    "artifacts/splits/grpo_validation.jsonl",
    "artifacts/splits/reward_calibration.jsonl",
    "artifacts/splits/split_manifest.json",
    "artifacts/splits/split_distribution_report.md",
    "artifacts/reward_calibration/best_weights.json",
    "artifacts/reward_calibration/calibration_summary.json",
    "artifacts/reward_calibration/calibration_report.md",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_NAMES = {".env", ".env.local", ".DS_Store"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_directory(relative: str) -> Iterable[Path]:
    directory = ROOT / relative
    if not directory.exists():
        raise SystemExit(f"required directory not found: {directory}")
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if EXCLUDED_PARTS & set(path.relative_to(ROOT).parts):
            continue
        if path.name in FORBIDDEN_NAMES or path.suffix in {".pyc", ".pyo"}:
            continue
        yield path


def _selected_files() -> list[Path]:
    selected: list[Path] = []
    for directory in DIRECTORIES:
        selected.extend(_iter_directory(directory))
    for relative in (*ROOT_FILES, *DATA_FILES):
        path = ROOT / relative
        if not path.is_file():
            raise SystemExit(f"required bundle file not found: {path}")
        selected.append(path)
    unique = sorted(set(selected), key=lambda path: str(path.relative_to(ROOT)))
    for path in unique:
        relative = path.relative_to(ROOT)
        if path.name in FORBIDDEN_NAMES or ".env" in path.name or "cache" in relative.parts:
            raise RuntimeError(f"security check rejected bundle path: {relative}")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"bundle already exists: {output}; pass --force to replace this generated archive")
    output.parent.mkdir(parents=True, exist_ok=True)
    files = _selected_files()
    entries = [
        {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in files
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "bundle_root": "ml-engineer-take-home-wg-recommendation",
        "files": entries,
        "checks": {
            "credential_files_included": False,
            "cache_directories_included": False,
            "golden_included_for_evaluation_only": True,
            "prepared_sft_data_included": True,
            "grpo_splits_and_frozen_reward_included": True,
        },
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = output.with_suffix(output.suffix + f".tmp{os.getpid()}")
    archive_root = manifest["bundle_root"]
    with tarfile.open(temporary, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=f"{archive_root}/{path.relative_to(ROOT)}", recursive=False)
        info = tarfile.TarInfo(f"{archive_root}/RUNPOD_BUNDLE_MANIFEST.json")
        info.size = len(manifest_bytes)
        info.mtime = int(datetime.now().timestamp())
        archive.addfile(info, io.BytesIO(manifest_bytes))
    os.replace(temporary, output)

    checksum_path = output.with_suffix(output.suffix + ".sha256")
    checksum_path.write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
    external_manifest = output.with_suffix(output.suffix + ".manifest.json")
    external_manifest.write_bytes(manifest_bytes)
    print(json.dumps({
        "bundle": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "files": len(files),
        "checksum": str(checksum_path),
        "manifest": str(external_manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
