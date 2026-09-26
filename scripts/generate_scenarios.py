#!/usr/bin/env python3
"""scripts/generate_scenarios.py — generate a reproducible batch of synthetic WG scenarios.

Writes to artifacts/scenarios/ only. Never touches data/ or tools/. Makes zero
network calls — assess_wg_risk_context() is pure Python, so even hazard-stratum
labelling here costs nothing and needs no rate limiting.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scenarios.generate import (  # noqa: E402
    ScenarioValidationError,
    generate_scenarios,
    load_golden_risk_hashes,
    validate_batch,
)

DATA_DIR = (ROOT / "data").resolve()
TOOLS_DIR = (ROOT / "tools").resolve()


def _assert_not_protected(path: Path) -> None:
    resolved = path.resolve()
    for protected in (DATA_DIR, TOOLS_DIR):
        if resolved == protected or protected in resolved.parents:
            raise SystemExit(f"refusing to write under protected path: {resolved} (under {protected})")


def _atomic_write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _counts(records: list[dict], key_fn) -> dict[str, int]:
    out: dict[str, int] = {}
    for rec in records:
        k = str(key_fn(rec))
        out[k] = out.get(k, 0) + 1
    return out


def _summarize(records: list[dict], output_path: Path) -> dict:
    return {
        "total_scenarios": len(records),
        "unique_scenario_ids": len({r["scenario_id"] for r in records}),
        "hazard_stratum_counts": _counts(records, lambda r: r["generation_metadata"]["hazard_stratum"]),
        "age_stratum_counts": _counts(records, lambda r: r["generation_metadata"]["age_stratum"]),
        "building_type_counts": _counts(records, lambda r: r["risk"]["building_type"]),
        "construction_class_counts": _counts(records, lambda r: r["risk"]["construction_class"]),
        "owner_occupied_counts": _counts(records, lambda r: r["risk"]["owner_occupied"]),
        "deductible_counts": _counts(records, lambda r: r["risk"]["with_deductible"]),
        "output_path": str(output_path),
    }


def _run_validate_only(args: argparse.Namespace) -> int:
    golden_hashes = load_golden_risk_hashes(Path(args.golden_path))
    try:
        first = generate_scenarios(args.n, seed=args.seed, golden_hashes=golden_hashes)
        second = generate_scenarios(args.n, seed=args.seed, golden_hashes=golden_hashes)
    except ScenarioValidationError as e:
        print(f"[FAIL] generation raised: {e}")
        return 1

    checks: list[tuple[str, bool, str]] = []
    ids_first = [r["scenario_id"] for r in first]
    ids_second = [r["scenario_id"] for r in second]
    checks.append(("deterministic hashing (same seed -> identical scenario_ids)", ids_first == ids_second, ""))

    label = "schema / enums / years / dwelling-units / rebuild-sums / uniqueness / no golden collision / diversity"
    try:
        validate_batch(first, golden_hashes=golden_hashes, expected_n=args.n)
        checks.append((label, True, ""))
    except ScenarioValidationError as e:
        checks.append((label, False, str(e)))

    ok = all(passed for _, passed, _ in checks)
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    print(f"\nvalidate-only: {'PASS' if ok else 'FAIL'} ({len(first)} scenarios, seed={args.seed})")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=str(ROOT / "artifacts/scenarios/scenarios.jsonl"))
    ap.add_argument("--config-output", default=str(ROOT / "artifacts/scenarios/generate_config.json"))
    ap.add_argument("--summary-output", default=str(ROOT / "artifacts/scenarios/summary.json"))
    ap.add_argument("--golden-path", default=str(ROOT / "data/golden_wg_recommendations.jsonl"))
    ap.add_argument("--validate-only", action="store_true", help="Regenerate in-memory and validate; write nothing.")
    args = ap.parse_args()

    if args.validate_only:
        return _run_validate_only(args)

    output_path = Path(args.output)
    config_path = Path(args.config_output)
    summary_path = Path(args.summary_output)
    for p in (output_path, config_path, summary_path):
        _assert_not_protected(p)

    golden_hashes = load_golden_risk_hashes(Path(args.golden_path))
    records = generate_scenarios(args.n, seed=args.seed, golden_hashes=golden_hashes)
    validate_batch(records, golden_hashes=golden_hashes, expected_n=args.n)

    _atomic_write_jsonl(output_path, records)

    config = {
        "n": args.n,
        "seed": args.seed,
        "output": str(output_path),
        "golden_path": args.golden_path,
        "golden_risks_loaded": len(golden_hashes),
    }
    _atomic_write_json(config_path, config)

    summary = _summarize(records, output_path)
    _atomic_write_json(summary_path, summary)

    print(f"generated {len(records)} unique, non-golden scenarios -> {output_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
