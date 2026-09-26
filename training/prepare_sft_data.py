#!/usr/bin/env python3
"""Prepare validated conversational prompt-completion data for WG SFT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BROKER_SYSTEM, build_broker_prompt  # noqa: E402
from training.common import (  # noqa: E402
    annotation_top3,
    load_jsonl,
    percentile,
    ranking_json,
    record_id,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_INPUTS = {
    "train": ROOT / "artifacts" / "splits" / "sft_train.jsonl",
    "validation": ROOT / "artifacts" / "splits" / "sft_validation.jsonl",
}
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "training" / "sft_data"


def _validate_and_prepare(record: dict[str, Any], source: Path) -> dict[str, Any]:
    scenario_id = record_id(record)
    for field in ("risk", "risk_context", "quotes"):
        if field not in record:
            raise SystemExit(f"{source}: scenario {scenario_id} is missing {field}")
    quotes = record["quotes"]
    if not isinstance(quotes, list) or len(quotes) < 3:
        raise SystemExit(f"{source}: scenario {scenario_id} has fewer than three quotes")
    available = [str(quote.get("tariff_id")) for quote in quotes]
    if len(available) != len(set(available)) or "None" in available:
        raise SystemExit(f"{source}: scenario {scenario_id} has invalid or duplicate tariff IDs")

    teacher_ids = annotation_top3(record)
    if len(teacher_ids) != 3 or len(set(teacher_ids)) != 3:
        raise SystemExit(f"{source}: scenario {scenario_id} has an invalid teacher top_3")
    unknown = sorted(set(teacher_ids) - set(available))
    if unknown:
        raise SystemExit(f"{source}: scenario {scenario_id} teacher IDs are absent from quotes: {unknown}")

    return {
        "scenario_id": scenario_id,
        "prompt": [
            {"role": "system", "content": BROKER_SYSTEM},
            {"role": "user", "content": build_broker_prompt(record, include_rationales=False)},
        ],
        "completion": [
            {"role": "assistant", "content": ranking_json(teacher_ids)},
        ],
    }


def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_chars = [sum(len(message["content"]) for message in record["prompt"]) for record in records]
    completion_chars = [len(record["completion"][0]["content"]) for record in records]
    return {
        "records": len(records),
        "unique_scenario_ids": len({record["scenario_id"] for record in records}),
        "prompt_characters": {
            "min": min(prompt_chars),
            "p50": percentile(prompt_chars, 0.50),
            "p95": percentile(prompt_chars, 0.95),
            "max": max(prompt_chars),
        },
        "completion_characters": {
            "min": min(completion_chars),
            "p50": percentile(completion_chars, 0.50),
            "p95": percentile(completion_chars, 0.95),
            "max": max(completion_chars),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", default=str(DEFAULT_INPUTS["train"]))
    parser.add_argument("--validation-input", default=str(DEFAULT_INPUTS["validation"]))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    inputs = {
        "train": Path(args.train_input).resolve(),
        "validation": Path(args.validation_input).resolve(),
    }
    prepared: dict[str, list[dict[str, Any]]] = {}
    for split, path in inputs.items():
        rows = [_validate_and_prepare(record, path) for record in load_jsonl(path)]
        if len(rows) != len({row["scenario_id"] for row in rows}):
            raise SystemExit(f"{path}: duplicate scenario IDs")
        prepared[split] = rows

    overlap = {row["scenario_id"] for row in prepared["train"]} & {
        row["scenario_id"] for row in prepared["validation"]
    }
    if overlap:
        raise SystemExit(f"train/validation overlap: {len(overlap)} scenario IDs")

    summary = {
        "format": "TRL conversational prompt-completion",
        "prompt_source": "src/prompts.py: BROKER_SYSTEM + build_broker_prompt(include_rationales=False)",
        "target_schema": {"top_3": [{"rank": "1..3", "tariff_id": "string"}]},
        "teacher_rationales_in_target": False,
        "splits": {split: _stats(rows) for split, rows in prepared.items()},
        "checks": {
            "train_validation_disjoint": True,
            "all_teacher_ids_exist_in_quote_sets": True,
            "all_targets_have_three_unique_ids": True,
            "golden_records_used": 0,
        },
    }
    if args.validate_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir).resolve()
    protected = [(ROOT / "data").resolve(), (ROOT / "tools").resolve()]
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise SystemExit(f"refusing to write prepared training data under {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, rows in prepared.items():
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, rows)
        summary["splits"][split]["source"] = {
            "path": str(inputs[split]),
            "sha256": sha256_file(inputs[split]),
        }
        summary["splits"][split]["output"] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        }

    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, summary)
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path), **summary["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
