#!/usr/bin/env python3
"""Prepare teacher-free prompts and frozen quote rewards for WG GRPO."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BROKER_SYSTEM, build_broker_prompt  # noqa: E402
from tools.reward import RewardWeights, score_quotes  # noqa: E402
from training.common import (  # noqa: E402
    annotation_top3,
    load_jsonl,
    percentile,
    record_id,
    sha256_file,
    write_json,
    write_jsonl,
)


DEFAULT_INPUTS = {
    "train": ROOT / "artifacts" / "splits" / "grpo_train.jsonl",
    "validation": ROOT / "artifacts" / "splits" / "grpo_validation.jsonl",
}
DEFAULT_WEIGHTS = ROOT / "artifacts" / "reward_calibration" / "best_weights.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "training" / "grpo_data"
REWARD_FIELDS = ("w_cov", "w_ins", "w_ded", "w_price", "w_score")


def _load_frozen_weights(path: Path) -> RewardWeights:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise SystemExit(f"reward configuration is not frozen: {path}")
    raw = payload.get("weights")
    if not isinstance(raw, dict) or set(raw) != set(REWARD_FIELDS):
        raise SystemExit(f"reward configuration has unexpected fields: {path}")
    return RewardWeights(**{name: float(raw[name]) for name in REWARD_FIELDS})


def _prepare(record: dict[str, Any], source: Path, weights: RewardWeights, validation: bool) -> dict[str, Any]:
    scenario_id = record_id(record)
    for field in ("risk", "risk_context", "quotes"):
        if field not in record:
            raise SystemExit(f"{source}: scenario {scenario_id} is missing {field}")
    quotes = record["quotes"]
    if not isinstance(quotes, list) or len(quotes) < 3:
        raise SystemExit(f"{source}: scenario {scenario_id} has fewer than three quotes")
    scored = score_quotes(quotes, weights)
    reward_by_id = {str(quote["tariff_id"]): float(quote["reward"]) for quote in scored}
    if len(reward_by_id) != len(quotes):
        raise SystemExit(f"{source}: scenario {scenario_id} has missing or duplicate tariff IDs")

    output: dict[str, Any] = {
        "scenario_id": scenario_id,
        "prompt": [
            {"role": "system", "content": BROKER_SYSTEM},
            {"role": "user", "content": build_broker_prompt(record, include_rationales=False)},
        ],
        "reward_by_tariff_id": reward_by_id,
    }
    if validation:
        teacher_ids = annotation_top3(record)
        if len(teacher_ids) != 3 or len(set(teacher_ids)) != 3:
            raise SystemExit(f"{source}: validation scenario {scenario_id} has no valid teacher top_3")
        if not set(teacher_ids) <= set(reward_by_id):
            raise SystemExit(f"{source}: validation teacher ranking contains an unavailable tariff")
        output["teacher_top3"] = teacher_ids
    elif annotation_top3(record):
        raise SystemExit(f"{source}: GRPO training scenario {scenario_id} unexpectedly contains a teacher label")
    return output


def _stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_chars = [sum(len(message["content"]) for message in row["prompt"]) for row in rows]
    return {
        "records": len(rows),
        "unique_scenario_ids": len({row["scenario_id"] for row in rows}),
        "quote_counts": {
            "min": min(len(row["reward_by_tariff_id"]) for row in rows),
            "p50": percentile([len(row["reward_by_tariff_id"]) for row in rows], 0.50),
            "max": max(len(row["reward_by_tariff_id"]) for row in rows),
        },
        "prompt_characters": {
            "min": min(prompt_chars),
            "p50": percentile(prompt_chars, 0.50),
            "p95": percentile(prompt_chars, 0.95),
            "max": max(prompt_chars),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", default=str(DEFAULT_INPUTS["train"]))
    parser.add_argument("--validation-input", default=str(DEFAULT_INPUTS["validation"]))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    inputs = {"train": Path(args.train_input).resolve(), "validation": Path(args.validation_input).resolve()}
    weights_path = Path(args.weights).resolve()
    weights = _load_frozen_weights(weights_path)
    prepared = {
        split: [_prepare(record, path, weights, split == "validation") for record in load_jsonl(path)]
        for split, path in inputs.items()
    }
    for split, rows in prepared.items():
        if len(rows) != len({row["scenario_id"] for row in rows}):
            raise SystemExit(f"{inputs[split]} contains duplicate scenario IDs")
    overlap = {row["scenario_id"] for row in prepared["train"]} & {
        row["scenario_id"] for row in prepared["validation"]
    }
    if overlap:
        raise SystemExit(f"GRPO train/validation overlap: {len(overlap)} scenarios")

    manifest: dict[str, Any] = {
        "format": "TRL conversational prompts with frozen per-tariff business rewards",
        "prompt_source": "src/prompts.py: BROKER_SYSTEM + build_broker_prompt(include_rationales=False)",
        "reward_source": {
            "path": str(weights_path),
            "sha256": sha256_file(weights_path),
            "weights": {name: getattr(weights, name) for name in REWARD_FIELDS},
        },
        "reward_components": [
            "strict_json_reward",
            "valid_tariffs_reward",
            "calibrated_top1_reward",
            "calibrated_ranking_reward",
        ],
        "teacher_policy": {
            "train": "absent",
            "validation": "teacher_top3 retained for offline monitoring only; never passed into a reward function",
        },
        "splits": {split: _stats(rows) for split, rows in prepared.items()},
        "checks": {
            "train_validation_disjoint": True,
            "golden_records_used": 0,
            "train_teacher_labels_used": 0,
            "rewards_precomputed_from_frozen_weights": True,
        },
    }
    if args.validate_only:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    output_dir = Path(args.output_dir).resolve()
    protected = [(ROOT / "data").resolve(), (ROOT / "tools").resolve()]
    if any(output_dir == path or path in output_dir.parents for path in protected):
        raise SystemExit(f"refusing to write prepared GRPO data under {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in prepared.items():
        output_path = output_dir / f"{split}.jsonl"
        write_jsonl(output_path, rows)
        manifest["splits"][split]["source"] = {
            "path": str(inputs[split]),
            "sha256": sha256_file(inputs[split]),
        }
        manifest["splits"][split]["output"] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), "splits": manifest["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
