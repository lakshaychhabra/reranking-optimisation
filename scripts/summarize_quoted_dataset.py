#!/usr/bin/env python3
"""Summarize quoted WG episodes into JSON plus report-ready Markdown.

The quoted JSONL contains success records only. Supplying --cache-dir also adds
the full collection funnel (success/no-quotes/insufficient/errors) and subgroup
success rates from every attempted scenario.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (ROOT / "data").resolve()
TOOLS_DIR = (ROOT / "tools").resolve()


def _assert_not_protected(path: Path) -> None:
    resolved = path.resolve()
    for protected in (DATA_DIR, TOOLS_DIR):
        if resolved == protected or protected in resolved.parents:
            raise SystemExit(f"refusing to write under protected path: {resolved} (under {protected})")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"input not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
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


def _load_cache(cache_dir: Path) -> list[dict[str, Any]]:
    if not cache_dir.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(cache_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _round(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(value, digits)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Iterable[int | float]) -> dict[str, Any]:
    clean = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not clean:
        return {"count": 0}
    return {
        "count": len(clean),
        "min": _round(min(clean)),
        "p10": _round(_percentile(clean, 0.10)),
        "p25": _round(_percentile(clean, 0.25)),
        "mean": _round(statistics.fmean(clean)),
        "median": _round(statistics.median(clean)),
        "p75": _round(_percentile(clean, 0.75)),
        "p90": _round(_percentile(clean, 0.90)),
        "max": _round(max(clean)),
    }


def _categorical(values: Iterable[Any]) -> dict[str, dict[str, float | int]]:
    normalized = [str(v) for v in values]
    counts = Counter(normalized)
    total = len(normalized)
    return {
        key: {"count": count, "pct": round(100 * count / total, 2) if total else 0.0}
        for key, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    }


def _age_stratum(year: int) -> str:
    if year <= 1959:
        return "old"
    if year <= 1994:
        return "middle"
    return "modern"


def _risk(record: dict[str, Any]) -> dict[str, Any]:
    risk = record.get("risk") or record.get("normalized_risk")
    if not isinstance(risk, dict):
        raise SystemExit(f"record {record.get('scenario_id')} has no usable risk")
    return risk


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return None
    return numerator / (denom_x * denom_y)


def _variation_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    varying = 0
    for record in records:
        encoded = {json.dumps(q.get(field), sort_keys=True, ensure_ascii=False) for q in record["quotes"]}
        varying += len(encoded) > 1
    total = len(records)
    return {
        "varying_scenarios": varying,
        "flat_scenarios": total - varying,
        "varying_pct": round(100 * varying / total, 2) if total else 0.0,
    }


def _funnel_by(
    records: list[dict[str, Any]],
    key_fn: Callable[[dict[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(key_fn(record))].append(record)
    output: dict[str, dict[str, Any]] = {}
    for key, group in sorted(groups.items()):
        statuses = Counter(str(r.get("status", "unknown")) for r in group)
        attempts = len(group)
        successes = statuses.get("success", 0)
        output[key] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate_pct": round(100 * successes / attempts, 2) if attempts else 0.0,
            "statuses": dict(sorted(statuses.items())),
        }
    return output


def build_summary(
    records: list[dict[str, Any]],
    cache_records: list[dict[str, Any]],
    input_path: Path,
    cache_dir: Path,
    seed: int,
) -> dict[str, Any]:
    ids = [str(r.get("scenario_id")) for r in records]
    if len(set(ids)) != len(ids):
        raise SystemExit("quoted dataset contains duplicate scenario IDs")
    if any(r.get("status") != "success" for r in records):
        raise SystemExit("quoted dataset must contain success records only")
    if any(len(r.get("quotes", [])) < 3 for r in records):
        raise SystemExit("quoted dataset contains a scenario with fewer than three quotes")

    risks = [_risk(r) for r in records]
    all_quotes = [q for r in records for q in r["quotes"]]
    recommended = [r["quotes"][0] for r in records]
    quote_counts = [len(r["quotes"]) for r in records]
    reward_margins = [r["quotes"][0]["reward"] - r["quotes"][1]["reward"] for r in records]
    prices = [float(q["price_annual"]) for q in all_quotes]
    scores = [float(q.get("coverage_score") or 0) for q in all_quotes]

    summary: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "cache_dir": str(cache_dir),
        "seed": seed,
        "dataset": {
            "successful_scenarios": len(records),
            "unique_scenario_ids": len(set(ids)),
            "total_quotes": len(all_quotes),
            "unique_tariff_ids": len({str(q.get("tariff_id")) for q in all_quotes}),
            "unique_insurers": len({str(q.get("insurer")) for q in all_quotes}),
            "unique_products": len({(str(q.get("insurer")), str(q.get("product"))) for q in all_quotes}),
        },
        "scenario_distribution": {
            "hazard_tier": _categorical(r["risk_context"]["hazard_exposure"]["flood_heavy_rain"] for r in records),
            "age_stratum": _categorical(_age_stratum(int(risk["construction_year"])) for risk in risks),
            "building_type": _categorical(risk["building_type"] for risk in risks),
            "construction_class": _categorical(risk["construction_class"] for risk in risks),
            "roof_type": _categorical(risk["roof_type"] for risk in risks),
            "attic_status": _categorical(risk["attic_status"] for risk in risks),
            "owner_occupied": _categorical(risk["owner_occupied"] for risk in risks),
            "with_deductible": _categorical(risk["with_deductible"] for risk in risks),
            "natural_hazard_requested": _categorical(
                r["requested_coverage"]["natural_hazard"] for r in records
            ),
        },
        "quote_statistics": {
            "quotes_per_scenario": _numeric_summary(quote_counts),
            "annual_price_all_quotes": _numeric_summary(q["price_annual"] for q in all_quotes),
            "annual_price_reward_recommended": _numeric_summary(q["price_annual"] for q in recommended),
            "coverage_score_all_quotes": _numeric_summary(q.get("coverage_score", 0) for q in all_quotes),
            "coverage_score_reward_recommended": _numeric_summary(q.get("coverage_score", 0) for q in recommended),
            "reward_all_quotes": _numeric_summary(q["reward"] for q in all_quotes),
            "top1_top2_reward_margin": _numeric_summary(reward_margins),
            "deductible_values": _categorical(q.get("deductible") for q in all_quotes),
            "num_coverages_values": _categorical(q.get("num_coverages") for q in all_quotes),
            "insured_amount_unlimited": sum(q.get("insured_amount") == "unbegrenzt" for q in all_quotes),
            "price_coverage_score_pearson": _round(_pearson(prices, scores)),
        },
        "reward_signal_diagnostics": {
            field: _variation_summary(records, field)
            for field in ("price_annual", "insured_amount", "deductible", "num_coverages", "coverage_score")
        },
        "market_distribution": {
            "insurers": _categorical(q.get("insurer") for q in all_quotes),
            "reward_recommended_insurers": _categorical(q.get("insurer") for q in recommended),
            "products": _categorical(f"{q.get('insurer')} :: {q.get('product')}" for q in all_quotes),
        },
    }

    if cache_records:
        statuses = Counter(str(r.get("status", "unknown")) for r in cache_records)
        attempts = len(cache_records)
        successes = statuses.get("success", 0)
        summary["collection_funnel"] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate_pct": round(100 * successes / attempts, 2) if attempts else 0.0,
            "statuses": dict(sorted(statuses.items())),
            "by_hazard_tier": _funnel_by(
                cache_records,
                lambda r: r["risk_context"]["hazard_exposure"]["flood_heavy_rain"],
            ),
            "by_age_stratum": _funnel_by(
                cache_records,
                lambda r: _age_stratum(int(_risk(r)["construction_year"])),
            ),
            "by_building_type": _funnel_by(cache_records, lambda r: _risk(r)["building_type"]),
            "by_construction_class": _funnel_by(cache_records, lambda r: _risk(r)["construction_class"]),
        }
    return summary


def _markdown_table(rows: list[list[Any]], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _distribution_rows(distribution: dict[str, dict[str, Any]]) -> list[list[Any]]:
    return [[name, values["count"], f"{values['pct']:.2f}%"] for name, values in distribution.items()]


def render_markdown(summary: dict[str, Any]) -> str:
    dataset = summary["dataset"]
    scenario = summary["scenario_distribution"]
    quote = summary["quote_statistics"]
    diagnostics = summary["reward_signal_diagnostics"]
    lines = [
        "# Quoted Dataset Distribution",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        "## Dataset overview",
        "",
        _markdown_table(
            [[dataset["successful_scenarios"], dataset["total_quotes"], dataset["unique_insurers"],
              dataset["unique_products"], dataset["unique_tariff_ids"]]],
            ["Scenarios", "Quotes", "Insurers", "Insurer-products", "Tariff IDs"],
        ),
    ]
    if "collection_funnel" in summary:
        funnel = summary["collection_funnel"]
        lines += [
            "",
            "## Collection funnel",
            "",
            _markdown_table(
                [[funnel["attempts"], funnel["successes"], f"{funnel['success_rate_pct']:.2f}%",
                  funnel["statuses"].get("no_quotes", 0), funnel["statuses"].get("insufficient_quotes", 0),
                  funnel["statuses"].get("transient_error", 0) + funnel["statuses"].get("permanent_error", 0)]],
                ["Attempts", "Successes", "Success rate", "No quotes", "Insufficient", "Errors"],
            ),
        ]
    for title, key in (
        ("Hazard tier", "hazard_tier"),
        ("Age stratum", "age_stratum"),
        ("Building type", "building_type"),
        ("Construction class", "construction_class"),
    ):
        lines += ["", f"## {title}", "", _markdown_table(_distribution_rows(scenario[key]), [title, "Count", "Share"])]

    qps = quote["quotes_per_scenario"]
    lines += [
        "",
        "## Quote and reward statistics",
        "",
        _markdown_table(
            [
                ["Quotes per scenario", qps.get("min"), qps.get("median"), qps.get("mean"), qps.get("max")],
                ["Annual price - all quotes", quote["annual_price_all_quotes"].get("min"),
                 quote["annual_price_all_quotes"].get("median"), quote["annual_price_all_quotes"].get("mean"),
                 quote["annual_price_all_quotes"].get("max")],
                ["Coverage score - all quotes", quote["coverage_score_all_quotes"].get("min"),
                 quote["coverage_score_all_quotes"].get("median"), quote["coverage_score_all_quotes"].get("mean"),
                 quote["coverage_score_all_quotes"].get("max")],
                ["Top-1 vs top-2 reward margin", quote["top1_top2_reward_margin"].get("min"),
                 quote["top1_top2_reward_margin"].get("median"), quote["top1_top2_reward_margin"].get("mean"),
                 quote["top1_top2_reward_margin"].get("max")],
            ],
            ["Metric", "Min", "Median", "Mean", "Max"],
        ),
        "",
        "## Reward-signal variation within scenarios",
        "",
        _markdown_table(
            [[field, values["varying_scenarios"], values["flat_scenarios"], f"{values['varying_pct']:.2f}%"]
             for field, values in diagnostics.items()],
            ["Field", "Varying scenarios", "Flat scenarios", "Varying share"],
        ),
        "",
        "## Reward-recommended insurers",
        "",
        _markdown_table(
            _distribution_rows(summary["market_distribution"]["reward_recommended_insurers"]),
            ["Insurer", "Count", "Share"],
        ),
        "",
        "## Interpretation notes",
        "",
        "- The quoted JSONL contains successful scenarios only; the collection funnel comes from cache records.",
        "- Flat within-scenario fields cannot distinguish tariffs under a simple ranking reward.",
        "- Subgroup success rates based on small counts are descriptive, not statistically conclusive.",
        "- The hazard context is a supplied heuristic, not authoritative ZURS/GDV hazard data.",
        "",
    ]
    return "\n".join(lines)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(ROOT / "artifacts/quoted/quoted_scenarios.jsonl"))
    ap.add_argument("--cache-dir", default=str(ROOT / "artifacts/cache/quotes"))
    ap.add_argument("--output", default=str(ROOT / "artifacts/quoted/distribution_summary.json"))
    ap.add_argument("--markdown-output", default=str(ROOT / "artifacts/quoted/distribution_report.md"))
    ap.add_argument("--seed", type=int, default=42, help="Recorded for experiment provenance.")
    args = ap.parse_args()

    input_path = Path(args.input)
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    markdown_path = Path(args.markdown_output)
    _assert_not_protected(output_path)
    _assert_not_protected(markdown_path)

    records = _load_jsonl(input_path)
    cache_records = _load_cache(cache_dir)
    summary = build_summary(records, cache_records, input_path, cache_dir, args.seed)
    markdown = render_markdown(summary)
    _atomic_write(output_path, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _atomic_write(markdown_path, markdown)

    print(json.dumps({
        "successful_scenarios": summary["dataset"]["successful_scenarios"],
        "total_quotes": summary["dataset"]["total_quotes"],
        "collection_funnel": summary.get("collection_funnel"),
        "json_summary": str(output_path),
        "markdown_report": str(markdown_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
