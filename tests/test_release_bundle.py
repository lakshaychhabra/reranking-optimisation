from __future__ import annotations

from pathlib import Path

import pytest

from scripts.create_release_bundle import _validate_destination, audit_release, verify_checksums


def test_release_destination_rejects_unsafe_or_sensitive_paths() -> None:
    for value in ("../escape.json", "cache/quote.json", "data/golden.jsonl", "runpod/archive.tar"):
        with pytest.raises(ValueError):
            _validate_destination(Path(value))


def test_release_audit_rejects_fable_labels_in_data(tmp_path: Path) -> None:
    path = tmp_path / "data" / "train.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"fable_annotation": {"top_3": []}}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="held-out Fable"):
        audit_release(tmp_path)


def test_release_audit_accepts_aggregate_golden_summary(tmp_path: Path) -> None:
    path = tmp_path / "evaluation" / "sft-golden-summary.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"golden": true, "metrics": {"top1_accuracy": 0.84}}\n', encoding="utf-8")
    audit_release(tmp_path)


def test_tracked_release_integrity() -> None:
    release = Path(__file__).resolve().parents[1] / "release"
    audit_release(release)
    verify_checksums(release)
