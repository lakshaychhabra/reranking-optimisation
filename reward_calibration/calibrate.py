#!/usr/bin/env python3
"""Calibrate the WG quote reward against silver teacher rankings offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.create_training_splits import (  # noqa: E402
    STRATIFY_FIELDS,
    _distribution,
    _iterative_stratified_partition,
)
from tools.reward import RewardWeights, score_quotes  # noqa: E402


DEFAULT_INPUT = ROOT / "artifacts/splits/reward_calibration.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts/reward_calibration"
DEFAULT_SEED = 20260926
DEFAULT_TRIALS = 20_000
DEFAULT_FOLDS = 5


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    w_cov: float
    w_ins: float
    w_ded: float
    w_price: float
    w_score: float
    source: str

    @property
    def w_static(self) -> float:
        return self.w_cov + self.w_ded

    def as_weights(self) -> RewardWeights:
        return RewardWeights(
            w_cov=self.w_cov,
            w_ins=self.w_ins,
            w_ded=self.w_ded,
            w_price=self.w_price,
            w_score=self.w_score,
        )


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    quote_ids: tuple[str, ...]
    # Per quote: coverage_n, insured_n, low_deductible_n, score_n, price_n.
    components: tuple[tuple[float, float, float, float, float], ...]
    teacher_top3: tuple[str, str, str]
    fold: int


@dataclass(frozen=True)
class Metrics:
    scenarios: int = 0
    top1_correct: int = 0
    top3_hits: int = 0
    pairwise_correct: int = 0
    pairwise_total: int = 0

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            self.scenarios + other.scenarios,
            self.top1_correct + other.top1_correct,
            self.top3_hits + other.top3_hits,
            self.pairwise_correct + other.pairwise_correct,
            self.pairwise_total + other.pairwise_total,
        )

    def __sub__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            self.scenarios - other.scenarios,
            self.top1_correct - other.top1_correct,
            self.top3_hits - other.top3_hits,
            self.pairwise_correct - other.pairwise_correct,
            self.pairwise_total - other.pairwise_total,
        )

    def rates(self) -> dict[str, float | int]:
        return {
            "scenarios": self.scenarios,
            "top1_correct": self.top1_correct,
            "top1_accuracy": round(self.top1_correct / self.scenarios, 6) if self.scenarios else 0.0,
            "top3_hits": self.top3_hits,
            "top3_recall": round(self.top3_hits / (3 * self.scenarios), 6) if self.scenarios else 0.0,
            "pairwise_correct": self.pairwise_correct,
            "pairwise_total": self.pairwise_total,
            "pairwise_accuracy": round(self.pairwise_correct / self.pairwise_total, 6)
            if self.pairwise_total
            else 0.0,
        }


@dataclass
class CandidateResult:
    candidate: Candidate
    overall: Metrics
    folds: list[Metrics]


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


def _minmax(values: list[float], rel_floor: float = 0.01) -> list[float]:
    """Match tools.reward._minmax without depending on its private API."""
    lo, hi = min(values), max(values)
    spread = hi - lo
    scale = max(abs(hi), abs(lo), 1.0)
    if spread <= rel_floor * scale:
        return [0.5 for _ in values]
    return [(value - lo) / spread for value in values]


def _insured_value(quote: dict[str, Any], scenario_maximum: float) -> float:
    value = quote.get("insured_amount")
    if value == "unbegrenzt" or value is None:
        return scenario_maximum
    return float(value)


def _components(quotes: list[dict[str, Any]]) -> tuple[tuple[float, float, float, float, float], ...]:
    numeric_insured = [
        float(quote["insured_amount"])
        for quote in quotes
        if isinstance(quote.get("insured_amount"), (int, float))
        and not isinstance(quote.get("insured_amount"), bool)
    ]
    maximum_insured = max(numeric_insured) if numeric_insured else 1.0
    insured = _minmax([_insured_value(quote, maximum_insured) for quote in quotes])
    deductible = _minmax([float(quote.get("deductible") or 0) for quote in quotes])
    coverage = _minmax([float(quote.get("num_coverages") or 0) for quote in quotes])
    score = _minmax([float(quote.get("coverage_score") or 0) for quote in quotes])
    price = _minmax([float(quote["price_annual"]) for quote in quotes])
    return tuple(
        (coverage[index], insured[index], 1.0 - deductible[index], score[index], price[index])
        for index in range(len(quotes))
    )


def _fold_assignments(records: list[dict[str, Any]], folds: int, seed: int) -> dict[str, int]:
    base_size, remainder = divmod(len(records), folds)
    sizes = {f"fold_{index}": base_size + (1 if index < remainder else 0) for index in range(folds)}
    partition = _iterative_stratified_partition(records, sizes, seed, "reward-calibration-folds")
    return {
        str(record["scenario_id"]): int(name.split("_")[1])
        for name, fold_records in partition.items()
        for record in fold_records
    }


def _prepare_scenarios(
    records: list[dict[str, Any]], fold_by_id: dict[str, int]
) -> tuple[list[Scenario], dict[str, Any]]:
    scenarios: list[Scenario] = []
    variation = Counter()
    seen_ids: set[str] = set()
    total_quotes = 0
    for record in records:
        scenario_id = record.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen_ids:
            raise SystemExit(f"invalid or duplicate scenario_id: {scenario_id!r}")
        seen_ids.add(scenario_id)
        quotes = record.get("quotes")
        annotation = record.get("teacher_annotation")
        if not isinstance(quotes, list) or len(quotes) < 3 or not isinstance(annotation, dict):
            raise SystemExit(f"scenario {scenario_id} lacks quotes or teacher_annotation")
        quote_ids = tuple(str(quote.get("tariff_id")) for quote in quotes)
        if len(set(quote_ids)) != len(quote_ids):
            raise SystemExit(f"scenario {scenario_id} has duplicate tariff IDs")
        top_3 = annotation.get("top_3")
        if not isinstance(top_3, list) or len(top_3) != 3:
            raise SystemExit(f"scenario {scenario_id} has invalid teacher top_3")
        teacher = tuple(str(item.get("tariff_id")) for item in top_3)
        if len(set(teacher)) != 3 or not set(teacher) <= set(quote_ids):
            raise SystemExit(f"scenario {scenario_id} teacher IDs are invalid")
        total_quotes += len(quotes)
        for field in ("num_coverages", "deductible", "insured_amount", "coverage_score", "price_annual"):
            encoded = {json.dumps(quote.get(field), sort_keys=True) for quote in quotes}
            variation[field] += len(encoded) > 1
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                quote_ids=quote_ids,
                components=_components(quotes),
                teacher_top3=(teacher[0], teacher[1], teacher[2]),
                fold=fold_by_id[scenario_id],
            )
        )
    diagnostics = {
        "scenarios": len(scenarios),
        "quotes": total_quotes,
        "mean_quotes_per_scenario": round(total_quotes / len(scenarios), 4),
        "min_quotes_per_scenario": min(len(scenario.quote_ids) for scenario in scenarios),
        "max_quotes_per_scenario": max(len(scenario.quote_ids) for scenario in scenarios),
        "within_scenario_varying": dict(sorted(variation.items())),
    }
    return scenarios, diagnostics


def _candidate(
    candidate_id: int,
    w_static: float,
    w_ins: float,
    w_score: float,
    w_price: float,
    source: str,
) -> Candidate:
    total = w_static + w_ins + w_score
    if total <= 0:
        raise ValueError("protection weights must have positive sum")
    w_static, w_ins, w_score = w_static / total, w_ins / total, w_score / total
    return Candidate(
        candidate_id=candidate_id,
        w_cov=w_static * 4.0 / 7.0,
        w_ins=w_ins,
        w_ded=w_static * 3.0 / 7.0,
        w_price=w_price,
        w_score=w_score,
        source=source,
    )


def _generate_candidates(trials: int, seed: int) -> list[Candidate]:
    if trials < 20:
        raise SystemExit("--trials must be at least 20")
    anchors = [
        (0.7, 0.3, 0.0, 1.0, "supplied_default"),
        (0.7, 0.3, 0.0, 0.0, "default_without_price"),
        (0.0, 0.0, 1.0, 0.0, "score_only"),
        (0.0, 0.0, 1.0, 0.5, "score_quality_price"),
        (0.0, 0.0, 1.0, 1.0, "score_quality_price"),
        (0.0, 0.0, 1.0, 2.0, "score_quality_price"),
        (0.0, 1.0, 0.0, 0.0, "insured_only"),
        (0.0, 1.0, 0.0, 1.0, "insured_quality_price"),
        (0.5, 0.0, 0.5, 0.5, "static_score_price"),
        (0.25, 0.25, 0.5, 1.0, "balanced_anchor"),
    ]
    candidates = [
        _candidate(index, static, insured, score, price, source)
        for index, (static, insured, score, price, source) in enumerate(anchors)
    ]
    rng = random.Random(seed)
    while len(candidates) < trials:
        # Equivalent to a Dirichlet(1,1,1): broad coverage of the simplex.
        raw = [-math.log(max(rng.random(), 1e-15)) for _ in range(3)]
        total = sum(raw)
        static, insured, score = (value / total for value in raw)
        # Include a small set of no-price candidates; otherwise search 0.05..5 log-uniformly.
        price = 0.0 if rng.random() < 0.02 else 10 ** rng.uniform(math.log10(0.05), math.log10(5.0))
        candidates.append(
            _candidate(len(candidates), static, insured, score, price, "random_simplex")
        )
    return candidates


def _predicted_order(scenario: Scenario, candidate: Candidate) -> tuple[str, ...]:
    denominator = candidate.w_cov + candidate.w_ins + candidate.w_ded + candidate.w_score
    rewards = []
    for index, (coverage, insured, low_deductible, score, price) in enumerate(scenario.components):
        protection = (
            candidate.w_cov * coverage
            + candidate.w_ins * insured
            + candidate.w_ded * low_deductible
            + candidate.w_score * score
        ) / denominator
        reward = round(protection / (1.0 + candidate.w_price * price), 4)
        rewards.append((reward, index))
    order = sorted(range(len(rewards)), key=lambda index: (-rewards[index][0], rewards[index][1]))
    return tuple(scenario.quote_ids[index] for index in order)


def _scenario_metrics(scenario: Scenario, candidate: Candidate) -> Metrics:
    predicted = _predicted_order(scenario, candidate)
    positions = {tariff_id: index for index, tariff_id in enumerate(predicted)}
    teacher = scenario.teacher_top3
    top1 = int(predicted[0] == teacher[0])
    top3_hits = len(set(predicted[:3]) & set(teacher))
    constraints: list[tuple[str, str]] = []
    for better_index in range(3):
        for worse_index in range(better_index + 1, 3):
            constraints.append((teacher[better_index], teacher[worse_index]))
    unranked = [tariff_id for tariff_id in scenario.quote_ids if tariff_id not in set(teacher)]
    constraints.extend((ranked, other) for ranked in teacher for other in unranked)
    pairwise_correct = sum(positions[better] < positions[worse] for better, worse in constraints)
    return Metrics(1, top1, top3_hits, pairwise_correct, len(constraints))


def _evaluate_candidate(
    scenarios: list[Scenario], candidate: Candidate, folds: int
) -> CandidateResult:
    fold_metrics = [Metrics() for _ in range(folds)]
    overall = Metrics()
    for scenario in scenarios:
        metrics = _scenario_metrics(scenario, candidate)
        overall = overall + metrics
        fold_metrics[scenario.fold] = fold_metrics[scenario.fold] + metrics
    return CandidateResult(candidate, overall, fold_metrics)


def _distance_from_default(candidate: Candidate) -> float:
    protection_distance = (
        abs(candidate.w_cov - 0.4)
        + abs(candidate.w_ins - 0.3)
        + abs(candidate.w_ded - 0.3)
        + abs(candidate.w_score)
    )
    price_distance = abs(math.log((candidate.w_price + 0.05) / 1.05))
    return protection_distance + 0.25 * price_distance


def _selection_key(metrics: Metrics, candidate: Candidate) -> tuple[int, int, int, float, int]:
    return (
        metrics.top1_correct,
        metrics.top3_hits,
        metrics.pairwise_correct,
        -_distance_from_default(candidate),
        -candidate.candidate_id,
    )


def _candidate_payload(result: CandidateResult, rank: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "candidate_id": result.candidate.candidate_id,
        "source": result.candidate.source,
        "weights": {
            "w_cov": result.candidate.w_cov,
            "w_ins": result.candidate.w_ins,
            "w_ded": result.candidate.w_ded,
            "w_price": result.candidate.w_price,
            "w_score": result.candidate.w_score,
            "w_static": result.candidate.w_static,
        },
        "metrics": result.overall.rates(),
        "distance_from_default": round(_distance_from_default(result.candidate), 8),
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def _verify_default_scorer(records: list[dict[str, Any]], scenarios: list[Scenario]) -> None:
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    baseline = _candidate(0, 0.7, 0.3, 0.0, 1.0, "supplied_default")
    for record in records:
        scenario = by_id[str(record["scenario_id"])]
        fast_order = _predicted_order(scenario, baseline)
        canonical = tuple(str(quote["tariff_id"]) for quote in score_quotes(record["quotes"], RewardWeights()))
        if fast_order != canonical:
            raise RuntimeError(f"fast scorer diverges from tools.reward for {scenario.scenario_id}")
        expected = str(record.get("reward_recommended_tariff_id"))
        if fast_order[0] != expected:
            raise RuntimeError(f"stored default recommendation mismatch for {scenario.scenario_id}")


def _wilson_interval(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = (proportion + z * z / (2 * trials)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)) / denominator
    return (round(centre - margin, 6), round(centre + margin, 6))


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _pct(value: float | int) -> str:
    return f"{100 * float(value):.2f}%"


def _render_report(summary: dict[str, Any]) -> str:
    baseline = summary["baseline"]
    best = summary["selected"]
    cross_validation = summary["cross_validation"]
    weights = best["weights"]
    lines = [
        "# Reward Calibration Report",
        "",
        "## Outcome",
        "",
        f"A deterministic search over {summary['search']['candidates']:,} candidate reward configurations",
        f"selected candidate `{best['candidate_id']}` using only {summary['dataset']['scenarios']} silver-labeled",
        "calibration scenarios. The supplied golden benchmark and GRPO validation split were not used.",
        "",
        _markdown_table(
            [
                ["Supplied default", _pct(baseline["metrics"]["top1_accuracy"]), _pct(baseline["metrics"]["top3_recall"]), _pct(baseline["metrics"]["pairwise_accuracy"])],
                ["Selected (in-sample)", _pct(best["metrics"]["top1_accuracy"]), _pct(best["metrics"]["top3_recall"]), _pct(best["metrics"]["pairwise_accuracy"])],
                ["Nested 5-fold selection", _pct(cross_validation["held_out_aggregate"]["top1_accuracy"]), _pct(cross_validation["held_out_aggregate"]["top3_recall"]), _pct(cross_validation["held_out_aggregate"]["pairwise_accuracy"])],
            ],
            ["Configuration", "Teacher Top-1", "Teacher Top-3 recall", "Pairwise accuracy"],
        ),
        "",
        "The nested-fold row is the honest estimate of the weight-selection procedure:",
        "each fold was scored with weights selected using only the other four folds.",
        "The final weights were then selected from all 200 calibration scenarios and frozen.",
        "",
        "## Frozen weights",
        "",
        _markdown_table(
            [[name, f"{weights[name]:.8f}"] for name in ("w_cov", "w_ins", "w_ded", "w_score", "w_price")],
            ["Parameter", "Value"],
        ),
        "",
        "## Dataset evidence",
        "",
        _markdown_table(
            [
                ["Scenarios", summary["dataset"]["scenarios"]],
                ["Quotes", summary["dataset"]["quotes"]],
                ["Mean quotes per scenario", summary["dataset"]["mean_quotes_per_scenario"]],
                ["Minimum / maximum quotes", f"{summary['dataset']['min_quotes_per_scenario']} / {summary['dataset']['max_quotes_per_scenario']}"] ,
            ],
            ["Measure", "Value"],
        ),
        "",
        "### Within-scenario signal availability",
        "",
        _markdown_table(
            [[field, count, f"{100 * count / summary['dataset']['scenarios']:.2f}%"] for field, count in summary["dataset"]["within_scenario_varying"].items()],
            ["Quote field", "Scenarios with variation", "Share"],
        ),
        "",
        "Because `num_coverages` and `deductible` never vary within a quote panel,",
        "their individual weights are not identifiable. The search therefore calibrates",
        "their combined static contribution and preserves their default 4:3 ratio.",
        "",
        "## Five-fold stability",
        "",
        _markdown_table(
            [
                [
                    fold["fold"],
                    fold["selected_candidate_id"],
                    _pct(fold["held_out_metrics"]["top1_accuracy"]),
                    _pct(fold["held_out_metrics"]["top3_recall"]),
                    _pct(fold["held_out_metrics"]["pairwise_accuracy"]),
                    f"{fold['weights']['w_score']:.4f}",
                    f"{fold['weights']['w_price']:.4f}",
                ]
                for fold in cross_validation["folds"]
            ],
            ["Held-out fold", "Selected candidate", "Top-1", "Top-3 recall", "Pairwise", "w_score", "w_price"],
        ),
        "",
        f"Nested held-out Top-1 95% Wilson interval: {_pct(cross_validation['top1_wilson_95'][0])}",
        f"to {_pct(cross_validation['top1_wilson_95'][1])}.",
        "",
        "## Search method",
        "",
        "- Protection weights were sampled over a three-part simplex: static baseline, insured amount, and coverage score.",
        "- The price penalty was sampled log-uniformly from 0.05 to 5.0, with explicit zero-price and hand-designed anchor candidates.",
        "- Candidates were ranked lexicographically by Top-1 agreement, Top-3 recall, and pairwise accuracy.",
        "- Distance from the supplied default reward was used only to break otherwise identical results.",
        "- Reward values were rounded to four decimals before ranking, matching `tools/reward.py`.",
        "- The fast calibration scorer was checked against `tools.reward.score_quotes` on all 200 default-weight scenarios.",
        "",
        "## Interpretation",
        "",
        "These weights approximate the GPT-6 Sol silver teacher; they are not independent ground truth.",
        "Do not adjust them using the golden benchmark. During GRPO, keep them fixed and combine",
        "the ranking reward with structural penalties for invalid JSON, duplicate IDs, or IDs absent",
        "from the scenario's quote panel.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == (ROOT / "data").resolve() or (ROOT / "data").resolve() in output_dir.parents:
        raise SystemExit("refusing to write reward-calibration outputs under data/")
    if args.folds < 2 or args.folds > 10:
        raise SystemExit("--folds must be between 2 and 10")

    records = _load_jsonl(input_path)
    fold_by_id = _fold_assignments(records, args.folds, args.seed)
    scenarios, diagnostics = _prepare_scenarios(records, fold_by_id)
    _verify_default_scorer(records, scenarios)
    candidates = _generate_candidates(args.trials, args.seed)

    results: list[CandidateResult] = []
    for index, candidate in enumerate(candidates, 1):
        results.append(_evaluate_candidate(scenarios, candidate, args.folds))
        if index % 2_000 == 0 or index == len(candidates):
            print(f"evaluated {index}/{len(candidates)} candidates", file=sys.stderr)

    baseline = results[0]
    ranked = sorted(results, key=lambda result: _selection_key(result.overall, result.candidate), reverse=True)
    selected = ranked[0]

    fold_summaries = []
    held_out = Metrics()
    for fold_index in range(args.folds):
        fold_selected = max(
            results,
            key=lambda result: _selection_key(
                result.overall - result.folds[fold_index], result.candidate
            ),
        )
        training_metrics = fold_selected.overall - fold_selected.folds[fold_index]
        validation_metrics = fold_selected.folds[fold_index]
        held_out = held_out + validation_metrics
        fold_summaries.append(
            {
                "fold": fold_index,
                "train_scenarios": training_metrics.scenarios,
                "held_out_scenarios": validation_metrics.scenarios,
                "selected_candidate_id": fold_selected.candidate.candidate_id,
                "weights": _candidate_payload(fold_selected)["weights"],
                "training_metrics": training_metrics.rates(),
                "held_out_metrics": validation_metrics.rates(),
            }
        )

    selected_payload = _candidate_payload(selected, 1)
    baseline_payload = _candidate_payload(baseline)
    interval = _wilson_interval(held_out.top1_correct, held_out.scenarios)
    fold_distributions = {
        str(fold_index): _distribution(
            [record for record in records if fold_by_id[str(record["scenario_id"])] == fold_index]
        )
        for fold_index in range(args.folds)
    }
    summary = {
        "experiment": "offline_reward_calibration",
        "seed": args.seed,
        "input": {"path": str(input_path), "sha256": _sha256_file(input_path)},
        "search": {
            "candidates": len(candidates),
            "folds": args.folds,
            "objective": ["top1_accuracy", "top3_recall", "pairwise_accuracy", "proximity_to_default"],
            "price_range": [0.0, 5.0],
            "protection_parameterization": "simplex(static=w_cov+w_ded, insured, score)",
            "static_mapping": "w_cov:w_ded = 4:3",
        },
        "dataset": diagnostics,
        "stratification_fields": list(STRATIFY_FIELDS),
        "fold_distributions": fold_distributions,
        "baseline": baseline_payload,
        "selected": selected_payload,
        "cross_validation": {
            "method": "nested held-out estimate: select on four folds, evaluate on the fifth",
            "folds": fold_summaries,
            "held_out_aggregate": held_out.rates(),
            "top1_wilson_95": list(interval),
        },
        "checks": {
            "fast_scorer_matches_tools_reward_default_on_all_scenarios": True,
            "golden_data_used": False,
            "grpo_validation_used": False,
            "llm_calls": 0,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / "candidate_results.jsonl"
    _write_jsonl(
        candidate_path,
        (_candidate_payload(result, rank) for rank, result in enumerate(ranked, 1)),
    )
    summary["candidate_results"] = {
        "path": str(candidate_path),
        "sha256": _sha256_file(candidate_path),
    }
    best_weights = {
        "status": "frozen",
        "seed": args.seed,
        "source_calibration_sha256": summary["input"]["sha256"],
        "candidate_id": selected.candidate.candidate_id,
        "weights": {
            name: selected_payload["weights"][name]
            for name in ("w_cov", "w_ins", "w_ded", "w_price", "w_score")
        },
        "selection_metrics": selected_payload["metrics"],
        "nested_cross_validation_metrics": held_out.rates(),
        "do_not_tune_on": [
            "artifacts/splits/grpo_validation.jsonl",
            "data/golden_wg_recommendations.jsonl",
        ],
    }
    best_path = output_dir / "best_weights.json"
    _write_json(best_path, best_weights)
    summary["best_weights"] = {"path": str(best_path), "sha256": _sha256_file(best_path)}

    report_path = output_dir / "calibration_report.md"
    _atomic_write(report_path, _render_report(summary))
    summary["report"] = {"path": str(report_path), "sha256": _sha256_file(report_path)}
    summary_path = output_dir / "calibration_summary.json"
    _write_json(summary_path, summary)

    print(
        json.dumps(
            {
                "selected_candidate_id": selected.candidate.candidate_id,
                "weights": best_weights["weights"],
                "baseline": baseline_payload["metrics"],
                "selected_in_sample": selected_payload["metrics"],
                "nested_cross_validation": held_out.rates(),
                "best_weights": str(best_path),
                "report": str(report_path),
                "summary": str(summary_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
