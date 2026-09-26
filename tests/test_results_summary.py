from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_qwen35_2b_summary_matches_frozen_prediction_files() -> None:
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    models = {entry["model"]: entry for entry in summary["models"]}
    for model_name in ("qwen35_2b_base", "qwen35_2b_sft", "qwen35_2b_grpo"):
        entry = models[model_name]
        path = RESULTS / entry["result_file"]
        rows = _rows(path)
        assert len(rows) == entry["scenarios"] == 50
        assert _sha256(path) == entry["result_sha256"]
        assert round(sum(float(row["top1_hit"]) for row in rows) / len(rows), 6) == entry["top1_accuracy"]
        assert round(sum(float(row["top3_recall"]) for row in rows) / len(rows), 6) == entry["top3_recall"]
        assert round(sum(float(row["reward_ratio"]) for row in rows) / len(rows), 6) == entry["reward_ratio"]
