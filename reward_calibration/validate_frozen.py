#!/usr/bin/env python3
"""Falsification, subgroup, and locked-golden checks for frozen reward weights.

This script never updates best_weights.json. It performs three checks:
1. nested-CV selection against deterministically permuted teacher rankings;
2. subgroup evaluation on the held-out GRPO validation split; and
3. a locked deterministic baseline evaluation on the supplied Fable golden set.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward_calibration.calibrate import (  # noqa: E402
    DEFAULT_FOLDS,
    DEFAULT_SEED,
    DEFAULT_TRIALS,
    Candidate,
    Metrics,
    _candidate,
    _evaluate_candidate,
    _fold_assignments,
    _generate_candidates,
    _load_jsonl,
    _pct,
    _predicted_order,
    _prepare_scenarios,
    _selection_key,
    _sha256_file,
    _wilson_interval,
)
from scripts.create_training_splits import STRATIFY_FIELDS, _labels  # noqa: E402


DEFAULT_CALIBRATION = ROOT / "artifacts/splits/reward_calibration.jsonl"
DEFAULT_VALIDATION = ROOT / "artifacts/splits/grpo_validation.jsonl"
DEFAULT_GOLDEN = ROOT / "data/golden_wg_recommendations.jsonl"
DEFAULT_WEIGHTS = ROOT / "artifacts/reward_calibration/best_weights.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/reward_calibration/validation"
PERMUTATION_SEED_OFFSET = 91_733


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    _atomic_write(
        path,
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
    )


def _load_frozen_candidate(path: Path) -> tuple[Candidate, dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"frozen weights not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise SystemExit(f"weights file is not marked frozen: {path}")
    weights = payload.get("weights")
    required = {"w_cov", "w_ins", "w_ded", "w_price", "w_score"}
    if not isinstance(weights, dict) or set(weights) != required:
        raise SystemExit(f"invalid frozen weight schema: {path}")
    candidate = _candidate(
        int(payload.get("candidate_id", -1)),
        float(weights["w_cov"]) + float(weights["w_ded"]),
        float(weights["w_ins"]),
        float(weights["w_score"]),
        float(weights["w_price"]),
        "frozen",
    )
    reconstructed = {
        "w_cov": candidate.w_cov,
        "w_ins": candidate.w_ins,
        "w_ded": candidate.w_ded,
        "w_price": candidate.w_price,
        "w_score": candidate.w_score,
    }
    if any(not math.isclose(reconstructed[name], float(weights[name]), abs_tol=1e-12) for name in required):
        raise SystemExit("frozen weights do not match the declared 4:3 static-weight parameterization")
    return candidate, payload


def _default_candidate() -> Candidate:
    return _candidate(0, 0.7, 0.3, 0.0, 1.0, "supplied_default")


def _metrics_payload(metrics: Metrics) -> dict[str, Any]:
    payload = metrics.rates()
    payload["top1_wilson_95"] = list(_wilson_interval(metrics.top1_correct, metrics.scenarios))
    return payload


def _permuted_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    output = []
    for record in records:
        cloned = copy.deepcopy(record)
        quote_ids = [str(quote["tariff_id"]) for quote in cloned["quotes"]]
        rng.shuffle(quote_ids)
        selected = quote_ids[:3]
        cloned["teacher_annotation"] = {
            "id": cloned["scenario_id"],
            "top_3": [
                {"rank": rank, "tariff_id": tariff_id}
                for rank, tariff_id in enumerate(selected, 1)
            ],
            "overall_rationale": "permutation negative control",
            "annotator_model": "deterministic-permutation-control",
        }
        output.append(cloned)
    return output


def _run_permutation_control(
    records: list[dict[str, Any]], trials: int, folds: int, seed: int
) -> dict[str, Any]:
    permutation_seed = seed + PERMUTATION_SEED_OFFSET
    permuted = _permuted_records(records, permutation_seed)
    fold_by_id = _fold_assignments(permuted, folds, seed)
    scenarios, diagnostics = _prepare_scenarios(permuted, fold_by_id)
    candidates = _generate_candidates(trials, seed)
    results = []
    for index, candidate in enumerate(candidates, 1):
        results.append(_evaluate_candidate(scenarios, candidate, folds))
        if index % 2_000 == 0 or index == len(candidates):
            print(f"permutation control: evaluated {index}/{len(candidates)} candidates", file=sys.stderr)

    selected_all = max(
        results,
        key=lambda result: _selection_key(result.overall, result.candidate),
    )
    held_out = Metrics()
    fold_summaries = []
    for fold_index in range(folds):
        selected = max(
            results,
            key=lambda result: _selection_key(
                result.overall - result.folds[fold_index], result.candidate
            ),
        )
        training = selected.overall - selected.folds[fold_index]
        validation = selected.folds[fold_index]
        held_out = held_out + validation
        fold_summaries.append(
            {
                "fold": fold_index,
                "selected_candidate_id": selected.candidate.candidate_id,
                "training_metrics": _metrics_payload(training),
                "held_out_metrics": _metrics_payload(validation),
            }
        )

    expected_top1 = sum(1 / len(scenario.quote_ids) for scenario in scenarios) / len(scenarios)
    expected_top3 = sum(3 / len(scenario.quote_ids) for scenario in scenarios) / len(scenarios)
    held_out_rates = _metrics_payload(held_out)
    # Top-1 is the cleanest leakage diagnostic. Allow normal finite-sample noise
    # plus five percentage points beyond the theoretical random-label rate.
    passes = float(held_out_rates["top1_accuracy"]) <= expected_top1 + 0.05
    return {
        "method": "randomly reorder each scenario's valid quote IDs, select on four folds, evaluate on fifth",
        "seed": permutation_seed,
        "candidates": trials,
        "dataset": diagnostics,
        "theoretical_random_expectation": {
            "top1_accuracy": round(expected_top1, 6),
            "top3_recall": round(expected_top3, 6),
        },
        "best_in_sample": {
            "candidate_id": selected_all.candidate.candidate_id,
            "metrics": _metrics_payload(selected_all.overall),
        },
        "nested_held_out": held_out_rates,
        "folds": fold_summaries,
        "pass_rule": "nested held-out Top-1 <= theoretical random Top-1 + 0.05",
        "passed": passes,
    }


def _scenario_group_values(record: dict[str, Any]) -> dict[str, str]:
    values = {field: value for field, value in _labels(record)}
    quote_count = len(record["quotes"])
    if quote_count <= 5:
        quote_bucket = "4-5"
    elif quote_count <= 7:
        quote_bucket = "6-7"
    else:
        quote_bucket = "8-9"
    values["quote_count_bucket"] = quote_bucket
    return values


def _subgroup_evaluation(
    records: list[dict[str, Any]], frozen: Candidate, baseline: Candidate
) -> dict[str, Any]:
    fold_by_id = {str(record["scenario_id"]): 0 for record in records}
    scenarios, diagnostics = _prepare_scenarios(records, fold_by_id)
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for record in records:
        scenario = scenario_by_id[str(record["scenario_id"])]
        for field, value in _scenario_group_values(record).items():
            groups[(field, value)].append(scenario)

    subgroup_rows = []
    for (field, value), subset in sorted(groups.items()):
        baseline_metrics = _evaluate_candidate(subset, baseline, 1).overall
        frozen_metrics = _evaluate_candidate(subset, frozen, 1).overall
        baseline_rates = _metrics_payload(baseline_metrics)
        frozen_rates = _metrics_payload(frozen_metrics)
        subgroup_rows.append(
            {
                "field": field,
                "value": value,
                "scenarios": len(subset),
                "default": baseline_rates,
                "frozen": frozen_rates,
                "top1_delta": round(
                    float(frozen_rates["top1_accuracy"]) - float(baseline_rates["top1_accuracy"]), 6
                ),
                "top3_recall_delta": round(
                    float(frozen_rates["top3_recall"]) - float(baseline_rates["top3_recall"]), 6
                ),
            }
        )
    overall_default = _evaluate_candidate(scenarios, baseline, 1).overall
    overall_frozen = _evaluate_candidate(scenarios, frozen, 1).overall
    return {
        "dataset": diagnostics,
        "overall": {
            "default": _metrics_payload(overall_default),
            "frozen": _metrics_payload(overall_frozen),
        },
        "groups": subgroup_rows,
        "minimum_frozen_top1": min(
            (float(row["frozen"]["top1_accuracy"]) for row in subgroup_rows if row["scenarios"] >= 10),
            default=None,
        ),
        "all_groups_improve_or_match_default_top1": all(
            row["top1_delta"] >= 0 for row in subgroup_rows
        ),
    }


def _golden_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for record in records:
        annotation = record.get("fable_annotation")
        if not isinstance(annotation, dict):
            raise SystemExit(f"golden record {record.get('id')} lacks fable_annotation")
        output.append(
            {
                **record,
                "scenario_id": str(record["id"]),
                "teacher_annotation": {"id": str(record["id"]), **annotation},
            }
        )
    return output


def _golden_evaluation(
    raw_records: list[dict[str, Any]], frozen: Candidate, baseline: Candidate
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = _golden_records(raw_records)
    scenarios, diagnostics = _prepare_scenarios(
        records, {str(record["scenario_id"]): 0 for record in records}
    )
    record_by_id = {str(record["scenario_id"]): record for record in records}
    per_scenario = []
    reward_ratios = {"default": [], "frozen": []}
    for scenario in scenarios:
        record = record_by_id[scenario.scenario_id]
        default_order = _predicted_order(scenario, baseline)
        frozen_order = _predicted_order(scenario, frozen)
        supplied_reward = {str(quote["tariff_id"]): float(quote["reward"]) for quote in record["quotes"]}
        maximum_reward = max(supplied_reward.values())
        default_ratio = supplied_reward[default_order[0]] / maximum_reward
        frozen_ratio = supplied_reward[frozen_order[0]] / maximum_reward
        reward_ratios["default"].append(default_ratio)
        reward_ratios["frozen"].append(frozen_ratio)
        per_scenario.append(
            {
                "id": scenario.scenario_id,
                "fable_top3": list(scenario.teacher_top3),
                "default_top3": list(default_order[:3]),
                "frozen_top3": list(frozen_order[:3]),
                "default_top1_hit": default_order[0] == scenario.teacher_top3[0],
                "frozen_top1_hit": frozen_order[0] == scenario.teacher_top3[0],
                "default_top3_recall": len(set(default_order[:3]) & set(scenario.teacher_top3)) / 3,
                "frozen_top3_recall": len(set(frozen_order[:3]) & set(scenario.teacher_top3)) / 3,
                "default_reward_ratio": default_ratio,
                "frozen_reward_ratio": frozen_ratio,
            }
        )
    default_metrics = _evaluate_candidate(scenarios, baseline, 1).overall
    frozen_metrics = _evaluate_candidate(scenarios, frozen, 1).overall
    return (
        {
            "status": "locked baseline evaluation; weights must not be changed from this result",
            "dataset": diagnostics,
            "default": {
                **_metrics_payload(default_metrics),
                "mean_supplied_reward_ratio": round(
                    sum(reward_ratios["default"]) / len(reward_ratios["default"]), 6
                ),
            },
            "frozen": {
                **_metrics_payload(frozen_metrics),
                "mean_supplied_reward_ratio": round(
                    sum(reward_ratios["frozen"]) / len(reward_ratios["frozen"]), 6
                ),
            },
        },
        per_scenario,
    )


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _render_report(summary: dict[str, Any]) -> str:
    permutation = summary["permutation_negative_control"]
    subgroup = summary["subgroup_validation"]
    golden = summary["locked_golden_evaluation"]
    lines = [
        "# Frozen Reward Validation Report",
        "",
        "This report evaluates the already-frozen reward. No result in this report",
        "was used to change or reselect its weights.",
        "",
        "## 1. Permutation negative control",
        "",
        "Teacher rankings were replaced by deterministic random rankings drawn from",
        "each scenario's valid tariff IDs. The full 20,000-candidate, five-fold",
        "selection process was then repeated.",
        "",
        _markdown_table(
            [
                ["Theoretical random expectation", _pct(permutation["theoretical_random_expectation"]["top1_accuracy"]), _pct(permutation["theoretical_random_expectation"]["top3_recall"])],
                ["Best candidate on all shuffled labels (optimistic)", _pct(permutation["best_in_sample"]["metrics"]["top1_accuracy"]), _pct(permutation["best_in_sample"]["metrics"]["top3_recall"])],
                ["Nested held-out selection", _pct(permutation["nested_held_out"]["top1_accuracy"]), _pct(permutation["nested_held_out"]["top3_recall"])],
            ],
            ["Control", "Top-1", "Top-3 recall"],
        ),
        "",
        f"Control result: **{'PASS' if permutation['passed'] else 'FAIL'}**. {permutation['pass_rule']}.",
        "",
        "## 2. Held-out silver subgroup validation",
        "",
        _markdown_table(
            [
                ["Supplied default", _pct(subgroup["overall"]["default"]["top1_accuracy"]), _pct(subgroup["overall"]["default"]["top3_recall"]), _pct(subgroup["overall"]["default"]["pairwise_accuracy"])],
                ["Frozen calibrated reward", _pct(subgroup["overall"]["frozen"]["top1_accuracy"]), _pct(subgroup["overall"]["frozen"]["top3_recall"]), _pct(subgroup["overall"]["frozen"]["pairwise_accuracy"])],
            ],
            ["Reward", "Top-1", "Top-3 recall", "Pairwise"],
        ),
        "",
        "### Subgroups",
        "",
        _markdown_table(
            [
                [row["field"], row["value"], row["scenarios"], _pct(row["default"]["top1_accuracy"]), _pct(row["frozen"]["top1_accuracy"]), _pct(row["frozen"]["top3_recall"])]
                for row in subgroup["groups"]
            ],
            ["Dimension", "Category", "N", "Default Top-1", "Frozen Top-1", "Frozen Top-3 recall"],
        ),
        "",
        "## 3. Locked Fable golden evaluation",
        "",
        "The frozen weights were applied once to the 50 supplied golden scenarios.",
        "The calibration and validation procedure did not use these labels.",
        "",
        _markdown_table(
            [
                ["Supplied default", _pct(golden["default"]["top1_accuracy"]), _pct(golden["default"]["top3_recall"]), _pct(golden["default"]["pairwise_accuracy"]), _pct(golden["default"]["mean_supplied_reward_ratio"])],
                ["Frozen calibrated reward", _pct(golden["frozen"]["top1_accuracy"]), _pct(golden["frozen"]["top3_recall"]), _pct(golden["frozen"]["pairwise_accuracy"]), _pct(golden["frozen"]["mean_supplied_reward_ratio"])],
            ],
            ["Reward", "Fable Top-1", "Fable Top-3 recall", "Pairwise", "Supplied reward ratio"],
        ),
        "",
        "The golden result is a deterministic non-model baseline. It must not be used",
        "to revise the frozen weights; future SFT and GRPO models should be compared",
        "against it under the same `run_eval.py` definitions.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--validation", default=str(DEFAULT_VALIDATION))
    parser.add_argument("--golden", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    calibration_path = Path(args.calibration).resolve()
    validation_path = Path(args.validation).resolve()
    golden_path = Path(args.golden).resolve()
    weights_path = Path(args.weights).resolve()
    output_dir = Path(args.output_dir).resolve()
    if (ROOT / "data").resolve() in output_dir.parents or output_dir == (ROOT / "data").resolve():
        raise SystemExit("refusing to write validation outputs under data/")

    frozen, weights_payload = _load_frozen_candidate(weights_path)
    baseline = _default_candidate()
    calibration_records = _load_jsonl(calibration_path)
    validation_records = _load_jsonl(validation_path)
    golden_records = _load_jsonl(golden_path)

    permutation = _run_permutation_control(
        calibration_records, args.trials, args.folds, args.seed
    )
    subgroup = _subgroup_evaluation(validation_records, frozen, baseline)
    golden, golden_per_scenario = _golden_evaluation(golden_records, frozen, baseline)

    output_dir.mkdir(parents=True, exist_ok=True)
    golden_details_path = output_dir / "golden_per_scenario.jsonl"
    _write_jsonl(golden_details_path, golden_per_scenario)
    summary = {
        "experiment": "frozen_reward_validation",
        "seed": args.seed,
        "frozen_weights": {
            "path": str(weights_path),
            "sha256": _sha256_file(weights_path),
            "candidate_id": weights_payload["candidate_id"],
            "weights": weights_payload["weights"],
            "unchanged": True,
        },
        "inputs": {
            "calibration": {"path": str(calibration_path), "sha256": _sha256_file(calibration_path)},
            "validation": {"path": str(validation_path), "sha256": _sha256_file(validation_path)},
            "golden": {"path": str(golden_path), "sha256": _sha256_file(golden_path)},
        },
        "permutation_negative_control": permutation,
        "subgroup_validation": subgroup,
        "locked_golden_evaluation": golden,
        "golden_per_scenario": {
            "path": str(golden_details_path),
            "sha256": _sha256_file(golden_details_path),
            "records": len(golden_per_scenario),
        },
        "checks": {
            "weights_reselected": False,
            "permutation_control_passed": permutation["passed"],
            "golden_used_for_calibration": False,
            "llm_calls": 0,
        },
    }
    report_path = output_dir / "validation_report.md"
    _atomic_write(report_path, _render_report(summary))
    summary["report"] = {"path": str(report_path), "sha256": _sha256_file(report_path)}
    summary_path = output_dir / "validation_summary.json"
    _write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "permutation_control": {
                    "passed": permutation["passed"],
                    "expected_top1": permutation["theoretical_random_expectation"]["top1_accuracy"],
                    "nested_held_out": permutation["nested_held_out"],
                },
                "silver_validation": subgroup["overall"],
                "golden_evaluation": golden,
                "weights_unchanged": True,
                "summary": str(summary_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
