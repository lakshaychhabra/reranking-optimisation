#!/usr/bin/env python3
"""Create deterministic, hierarchical, stratified training-data splits.

Stage 1 partitions the complete synthetic dataset into three disjoint pools:
reward calibration, SFT, and GRPO. Stage 2 independently re-stratifies the SFT
and GRPO pools into train and validation splits.

No third-party packages are required. The splitter balances several categorical
dimensions simultaneously with a deterministic iterative-stratification
heuristic and writes distribution evidence for auditability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT / "data").resolve()
TOOLS_DIR = (ROOT / "tools").resolve()

DEFAULT_QUOTED = ROOT / "artifacts/quoted/final_quoted_scenarios.jsonl"
DEFAULT_ANNOTATIONS = ROOT / "artifacts/teacher/annotations.jsonl"
DEFAULT_GOLDEN = ROOT / "data/golden_wg_recommendations.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/splits"

STRATIFY_FIELDS = (
    "hazard_tier",
    "age_stratum",
    "building_type",
    "construction_class",
    "natural_hazard_requested",
    "owner_occupied",
)


def _assert_not_protected(path: Path) -> None:
    resolved = path.resolve()
    for protected in (DATA_DIR, TOOLS_DIR):
        if resolved == protected or protected in resolved.parents:
            raise SystemExit(
                f"refusing to write under protected path: {resolved} (under {protected})"
            )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"expected JSON object at {path}:{line_number}")
            records.append(record)
    if not records:
        raise SystemExit(f"no records found in {path}")
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(seed: int, namespace: str, scenario_id: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{scenario_id}".encode()).hexdigest()


def _canonical_risk(record: dict[str, Any]) -> str:
    risk = record.get("risk")
    if not isinstance(risk, dict):
        raise SystemExit(f"record {record.get('scenario_id', record.get('id'))} has no risk object")
    return json.dumps(risk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _age_stratum(year: int) -> str:
    if year <= 1959:
        return "old"
    if year <= 1994:
        return "middle"
    return "modern"


def _labels(record: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    risk = record.get("risk")
    risk_context = record.get("risk_context")
    requested = record.get("requested_coverage")
    if not isinstance(risk, dict) or not isinstance(risk_context, dict) or not isinstance(requested, dict):
        raise SystemExit(f"scenario {record.get('scenario_id')} lacks stratification fields")
    try:
        hazard = risk_context["hazard_exposure"]["flood_heavy_rain"]
        year = int(risk["construction_year"])
        values = {
            "hazard_tier": hazard,
            "age_stratum": _age_stratum(year),
            "building_type": risk["building_type"],
            "construction_class": risk["construction_class"],
            "natural_hazard_requested": requested["natural_hazard"],
            "owner_occupied": risk["owner_occupied"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(
            f"scenario {record.get('scenario_id')} has invalid stratification fields: {exc}"
        ) from exc
    return tuple((field, str(values[field]).lower()) for field in STRATIFY_FIELDS)


def _index_unique(
    records: Iterable[dict[str, Any]], key: str, source_name: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for record in records:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise SystemExit(f"{source_name} record has no non-empty {key!r}")
        if value in output:
            raise SystemExit(f"duplicate {key} {value!r} in {source_name}")
        output[value] = record
    return output


def _validate_sources(
    quoted: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    golden: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    quoted_by_id = _index_unique(quoted, "scenario_id", "quoted input")
    annotations_by_id = _index_unique(annotations, "id", "teacher annotations")
    quoted_ids = set(quoted_by_id)
    annotation_ids = set(annotations_by_id)
    if quoted_ids != annotation_ids:
        missing_labels = sorted(quoted_ids - annotation_ids)
        missing_quotes = sorted(annotation_ids - quoted_ids)
        raise SystemExit(
            "quoted/annotation ID mismatch: "
            f"missing_annotations={len(missing_labels)}, missing_quotes={len(missing_quotes)}"
        )

    golden_risks = {_canonical_risk(record) for record in golden}
    collisions = [sid for sid, record in quoted_by_id.items() if _canonical_risk(record) in golden_risks]
    if collisions:
        raise SystemExit(f"training input overlaps golden risks: {collisions[:5]}")

    for scenario_id, record in quoted_by_id.items():
        if record.get("status") != "success" or len(record.get("quotes", [])) < 3:
            raise SystemExit(f"scenario {scenario_id} is not a usable quoted episode")
        quote_ids = {str(quote.get("tariff_id")) for quote in record["quotes"]}
        annotation = annotations_by_id[scenario_id]
        top_3 = annotation.get("top_3")
        if not isinstance(top_3, list) or len(top_3) != 3:
            raise SystemExit(f"annotation {scenario_id} does not have exactly three rankings")
        ranked_ids = [str(item.get("tariff_id")) for item in top_3]
        ranks = [item.get("rank") for item in top_3]
        if ranks != [1, 2, 3] or len(set(ranked_ids)) != 3 or not set(ranked_ids) <= quote_ids:
            raise SystemExit(f"annotation {scenario_id} contains invalid rankings or tariff IDs")
        _labels(record)
    return quoted_by_id, annotations_by_id


def _iterative_stratified_partition(
    records: list[dict[str, Any]],
    group_sizes: dict[str, int],
    seed: int,
    namespace: str,
) -> dict[str, list[dict[str, Any]]]:
    """Partition records while approximately preserving every field marginal."""
    if sum(group_sizes.values()) != len(records):
        raise ValueError(
            f"group sizes sum to {sum(group_sizes.values())}, expected {len(records)}"
        )
    if any(size < 0 for size in group_sizes.values()):
        raise ValueError("group sizes cannot be negative")

    by_id = {str(record["scenario_id"]): record for record in records}
    label_by_id = {scenario_id: _labels(record) for scenario_id, record in by_id.items()}
    remaining_for_label: dict[tuple[str, str], set[str]] = defaultdict(set)
    total_by_label: Counter[tuple[str, str]] = Counter()
    for scenario_id, labels in label_by_id.items():
        for label in labels:
            remaining_for_label[label].add(scenario_id)
            total_by_label[label] += 1

    total = len(records)
    targets = {
        group: {
            label: count * size / total
            for label, count in total_by_label.items()
        }
        for group, size in group_sizes.items()
    }
    assigned_counts = {group: Counter() for group in group_sizes}
    remaining_capacity = dict(group_sizes)
    output: dict[str, list[dict[str, Any]]] = {group: [] for group in group_sizes}
    pending = set(by_id)

    while pending:
        active_labels = [
            (len(ids & pending), label)
            for label, ids in remaining_for_label.items()
            if ids & pending
        ]
        _, rarest_label = min(active_labels, key=lambda item: (item[0], item[1]))
        candidate_ids = sorted(
            remaining_for_label[rarest_label] & pending,
            key=lambda scenario_id: _stable_hash(seed, namespace + ":record", scenario_id),
        )

        for scenario_id in candidate_ids:
            if scenario_id not in pending:
                continue
            labels = label_by_id[scenario_id]
            eligible = [group for group, capacity in remaining_capacity.items() if capacity > 0]
            if not eligible:
                raise RuntimeError("no remaining split capacity")

            def group_score(group: str) -> tuple[float, float, str]:
                deficits = []
                for label in labels:
                    target = targets[group][label]
                    deficit = target - assigned_counts[group][label]
                    deficits.append(deficit / max(target, 1.0))
                capacity_ratio = remaining_capacity[group] / max(group_sizes[group], 1)
                deterministic_tie = _stable_hash(seed, namespace + ":group", scenario_id + group)
                return (sum(deficits) / len(deficits), capacity_ratio, deterministic_tie)

            chosen = max(eligible, key=group_score)
            output[chosen].append(by_id[scenario_id])
            remaining_capacity[chosen] -= 1
            assigned_counts[chosen].update(labels)
            pending.remove(scenario_id)

    actual_sizes = {group: len(group_records) for group, group_records in output.items()}
    if actual_sizes != group_sizes:
        raise RuntimeError(f"split size invariant failed: {actual_sizes} != {group_sizes}")
    return output


def _distribution(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float | int]]]:
    counts: dict[str, Counter[str]] = {field: Counter() for field in STRATIFY_FIELDS}
    for record in records:
        for field, value in _labels(record):
            counts[field][value] += 1
    total = len(records)
    return {
        field: {
            value: {
                "count": count,
                "pct": round(100 * count / total, 4) if total else 0.0,
            }
            for value, count in sorted(field_counts.items())
        }
        for field, field_counts in counts.items()
    }


def _max_distribution_deviation(
    reference: list[dict[str, Any]], subset: list[dict[str, Any]]
) -> float:
    reference_distribution = _distribution(reference)
    subset_distribution = _distribution(subset)
    deviations = []
    for field in STRATIFY_FIELDS:
        for value, stats in reference_distribution[field].items():
            subset_pct = subset_distribution[field].get(value, {}).get("pct", 0.0)
            deviations.append(abs(float(stats["pct"]) - float(subset_pct)))
    return round(max(deviations, default=0.0), 4)


def _ordered(records: list[dict[str, Any]], seed: int, split_name: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: _stable_hash(seed, "output:" + split_name, str(record["scenario_id"])),
    )


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records)
    _atomic_write(path, text)


def _with_annotation(
    record: dict[str, Any], annotations_by_id: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    return {**record, "teacher_annotation": annotations_by_id[str(record["scenario_id"])]}


def _distribution_cell(stats: dict[str, float | int] | None) -> str:
    if not stats:
        return "0 (0.00%)"
    return f"{stats['count']} ({float(stats['pct']):.2f}%)"


def _display_category(field: str, value: str) -> str:
    if field == "construction_class":
        return value.upper()
    if field in {"natural_hazard_requested", "owner_occupied"}:
        return value.title()
    return value


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _render_distribution_report(manifest: dict[str, Any]) -> str:
    source = manifest["source_distribution"]
    stage_one = manifest["stage_1_pools"]
    final = manifest["splits"]
    lines = [
        "# Training Split Distribution Report",
        "",
        "This report documents the deterministic hierarchical split of the 1,490",
        "synthetic, teacher-annotated scenarios. The supplied 50-scenario golden",
        "dataset is excluded and remains reserved for final evaluation.",
        "",
        "## Split sizes",
        "",
        _markdown_table(
            [
                ["Reward calibration", final["reward_calibration"]["records"], "Reward-weight selection"],
                ["SFT train", final["sft_train"]["records"], "Supervised fine-tuning"],
                ["SFT validation", final["sft_validation"]["records"], "SFT checkpoint selection"],
                ["GRPO train", final["grpo_train"]["records"], "Policy optimization"],
                ["GRPO validation", final["grpo_validation"]["records"], "GRPO checkpoint monitoring/selection"],
            ],
            ["Final split", "Records", "Purpose"],
        ),
        "",
        "## Stratification quality",
        "",
        "Values are the largest absolute percentage-point difference across all",
        "balanced category marginals. Stage-one pools are compared with the full",
        "source; final train/validation splits are compared with their parent pool.",
        "",
        _markdown_table(
            [
                ["Reward calibration", stage_one["reward_calibration"]["max_marginal_deviation_from_source_percentage_points"]],
                ["SFT pool", stage_one["sft_pool"]["max_marginal_deviation_from_source_percentage_points"]],
                ["GRPO pool", stage_one["grpo_pool"]["max_marginal_deviation_from_source_percentage_points"]],
                ["SFT train", final["sft_train"]["max_marginal_deviation_from_parent_percentage_points"]],
                ["SFT validation", final["sft_validation"]["max_marginal_deviation_from_parent_percentage_points"]],
                ["GRPO train", final["grpo_train"]["max_marginal_deviation_from_parent_percentage_points"]],
                ["GRPO validation", final["grpo_validation"]["max_marginal_deviation_from_parent_percentage_points"]],
            ],
            ["Pool or split", "Maximum deviation (percentage points)"],
        ),
        "",
        "## Stage-one pool distributions",
        "",
        "Each cell is `count (share within dataset/pool)`.",
        "",
    ]

    stage_columns = ["reward_calibration", "sft_pool", "grpo_pool"]
    for field in STRATIFY_FIELDS:
        values = sorted(source[field])
        rows = []
        for value in values:
            rows.append(
                [field, _display_category(field, value), _distribution_cell(source[field].get(value))]
                + [
                    _distribution_cell(stage_one[name]["distribution"][field].get(value))
                    for name in stage_columns
                ]
            )
        lines.extend(
            [
                f"### {field}",
                "",
                _markdown_table(
                    rows,
                    ["Field", "Category", "Full source", "Reward calibration", "SFT pool", "GRPO pool"],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Final split distributions",
            "",
            "The SFT and GRPO pools were independently stratified a second time.",
            "Each cell is `count (share within split)`.",
            "",
        ]
    )
    final_columns = [
        "reward_calibration",
        "sft_train",
        "sft_validation",
        "grpo_train",
        "grpo_validation",
    ]
    final_headers = [
        "Field",
        "Category",
        "Reward calibration",
        "SFT train",
        "SFT validation",
        "GRPO train",
        "GRPO validation",
    ]
    for field in STRATIFY_FIELDS:
        values = sorted(source[field])
        rows = []
        for value in values:
            rows.append(
                [field, _display_category(field, value)]
                + [
                    _distribution_cell(final[name]["distribution"][field].get(value))
                    for name in final_columns
                ]
            )
        lines.extend(
            [
                f"### {field}",
                "",
                _markdown_table(rows, final_headers),
                "",
            ]
        )

    lines.extend(
        [
            "## Integrity checks",
            "",
            "- All 1,490 source scenarios are assigned exactly once.",
            "- The five final splits are mutually disjoint and exhaustive.",
            "- Quote and teacher-annotation IDs match one-to-one.",
            "- No synthetic risk exactly overlaps a supplied golden risk.",
            "- GRPO training records omit teacher annotations; GRPO validation retains them only for offline metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quoted-input", default=str(DEFAULT_QUOTED))
    parser.add_argument("--annotations-input", default=str(DEFAULT_ANNOTATIONS))
    parser.add_argument("--golden-input", default=str(DEFAULT_GOLDEN))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--reward-size", type=int, default=200)
    parser.add_argument("--sft-train-size", type=int, default=300)
    parser.add_argument("--sft-validation-size", type=int, default=90)
    parser.add_argument("--grpo-train-size", type=int, default=810)
    parser.add_argument("--grpo-validation-size", type=int, default=90)
    args = parser.parse_args()

    quoted_path = Path(args.quoted_input).resolve()
    annotations_path = Path(args.annotations_input).resolve()
    golden_path = Path(args.golden_input).resolve()
    output_dir = Path(args.output_dir).resolve()
    _assert_not_protected(output_dir)

    quoted = _load_jsonl(quoted_path)
    annotations = _load_jsonl(annotations_path)
    golden = _load_jsonl(golden_path)
    _, annotations_by_id = _validate_sources(quoted, annotations, golden)

    sft_pool_size = args.sft_train_size + args.sft_validation_size
    grpo_pool_size = args.grpo_train_size + args.grpo_validation_size
    requested_total = args.reward_size + sft_pool_size + grpo_pool_size
    if requested_total != len(quoted):
        raise SystemExit(
            f"requested split sizes total {requested_total}, but quoted input has {len(quoted)} records"
        )

    stage_one = _iterative_stratified_partition(
        quoted,
        {
            "reward_calibration": args.reward_size,
            "sft_pool": sft_pool_size,
            "grpo_pool": grpo_pool_size,
        },
        args.seed,
        "stage1",
    )
    sft_stage = _iterative_stratified_partition(
        stage_one["sft_pool"],
        {"sft_train": args.sft_train_size, "sft_validation": args.sft_validation_size},
        args.seed,
        "stage2:sft",
    )
    grpo_stage = _iterative_stratified_partition(
        stage_one["grpo_pool"],
        {"grpo_train": args.grpo_train_size, "grpo_validation": args.grpo_validation_size},
        args.seed,
        "stage2:grpo",
    )
    final_splits = {
        "reward_calibration": stage_one["reward_calibration"],
        **sft_stage,
        **grpo_stage,
    }

    all_final_ids = [str(record["scenario_id"]) for records in final_splits.values() for record in records]
    if len(all_final_ids) != len(set(all_final_ids)) or set(all_final_ids) != {
        str(record["scenario_id"]) for record in quoted
    }:
        raise RuntimeError("final splits are not a disjoint, exhaustive partition")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    labeled_splits = {
        "reward_calibration",
        "sft_train",
        "sft_validation",
        "grpo_validation",
    }
    for split_name, split_records in final_splits.items():
        path = output_dir / f"{split_name}.jsonl"
        ordered = _ordered(split_records, args.seed, split_name)
        if split_name in labeled_splits:
            payload = [_with_annotation(record, annotations_by_id) for record in ordered]
        else:
            payload = ordered
        _write_jsonl(path, payload)
        output_paths[split_name] = path

    assignments = []
    for split_name, split_records in final_splits.items():
        pool = "reward_calibration" if split_name == "reward_calibration" else split_name.split("_", 1)[0] + "_pool"
        for record in split_records:
            assignments.append(
                {
                    "scenario_id": record["scenario_id"],
                    "stage_1_pool": pool,
                    "final_split": split_name,
                }
            )
    assignments.sort(key=lambda item: item["scenario_id"])
    assignments_path = output_dir / "split_assignments.jsonl"
    _write_jsonl(assignments_path, assignments)

    source_distribution = _distribution(quoted)
    manifest: dict[str, Any] = {
        "seed": args.seed,
        "strategy": "deterministic two-stage iterative stratification",
        "stratification_fields": list(STRATIFY_FIELDS),
        "sources": {
            "quoted": {"path": str(quoted_path), "sha256": _sha256_file(quoted_path)},
            "annotations": {
                "path": str(annotations_path),
                "sha256": _sha256_file(annotations_path),
            },
            "golden_exclusion_check": {"path": str(golden_path), "records": len(golden)},
        },
        "hierarchy": {
            "stage_1": {
                "reward_calibration": args.reward_size,
                "sft_pool": sft_pool_size,
                "grpo_pool": grpo_pool_size,
            },
            "stage_2": {
                "sft_pool": {
                    "sft_train": args.sft_train_size,
                    "sft_validation": args.sft_validation_size,
                },
                "grpo_pool": {
                    "grpo_train": args.grpo_train_size,
                    "grpo_validation": args.grpo_validation_size,
                },
            },
        },
        "label_policy": {
            "reward_calibration": "teacher_annotation included",
            "sft_train": "teacher_annotation included",
            "sft_validation": "teacher_annotation included",
            "grpo_train": "teacher_annotation intentionally omitted",
            "grpo_validation": "teacher_annotation included for offline monitoring",
        },
        "source_distribution": source_distribution,
        "stage_1_pools": {
            pool_name: {
                "records": len(pool_records),
                "distribution": _distribution(pool_records),
                "max_marginal_deviation_from_source_percentage_points": _max_distribution_deviation(
                    quoted, pool_records
                ),
            }
            for pool_name, pool_records in stage_one.items()
        },
        "splits": {},
        "checks": {
            "source_records": len(quoted),
            "unique_source_ids": len({str(record["scenario_id"]) for record in quoted}),
            "assigned_records": len(all_final_ids),
            "unique_assigned_ids": len(set(all_final_ids)),
            "disjoint_and_exhaustive": True,
            "quoted_annotation_ids_match": True,
            "golden_risk_overlap": 0,
        },
    }
    for split_name, split_records in final_splits.items():
        path = output_paths[split_name]
        reference = (
            quoted
            if split_name == "reward_calibration"
            else stage_one["sft_pool"]
            if split_name.startswith("sft_")
            else stage_one["grpo_pool"]
        )
        manifest["splits"][split_name] = {
            "path": str(path),
            "sha256": _sha256_file(path),
            "records": len(split_records),
            "teacher_annotation_included": split_name in labeled_splits,
            "distribution": _distribution(split_records),
            "max_marginal_deviation_from_parent_percentage_points": _max_distribution_deviation(
                reference, split_records
            ),
        }
    manifest["assignments"] = {
        "path": str(assignments_path),
        "sha256": _sha256_file(assignments_path),
        "records": len(assignments),
    }
    report_path = output_dir / "split_distribution_report.md"
    _atomic_write(report_path, _render_distribution_report(manifest))
    manifest["distribution_report"] = {
        "path": str(report_path),
        "sha256": _sha256_file(report_path),
    }
    manifest_path = output_dir / "split_manifest.json"
    _atomic_write(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest": str(manifest_path),
                "stage_1": manifest["hierarchy"]["stage_1"],
                "final_splits": {
                    name: {
                        "records": len(records),
                        "max_parent_deviation_pp": manifest["splits"][name][
                            "max_marginal_deviation_from_parent_percentage_points"
                        ],
                    }
                    for name, records in final_splits.items()
                },
                "checks": manifest["checks"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
