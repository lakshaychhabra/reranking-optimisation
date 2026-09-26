#!/usr/bin/env python3
"""scripts/collect_quotes.py — collect live-engine quotes for scenarios, cached & rate-limited.

Safety, by construction:
  - Defaults to --dry-run (no engine calls, no sleeps) unless --execute-live is given.
  - --max-live-calls has no upper ceiling; it's whatever you pass (default 20).
  - --delay is floored at 3.0s (CLAUDE.md rule 2 / tools/README.md); QuoteEngineClient
    enforces this floor a second time internally, so it can't be bypassed via this flag.
  - Refuses live execution if MM_ID is unset; MM_ID's value is never read for any other
    purpose and never printed.
  - Every scenario is attempted at most once, ever — a cached record of ANY status
    (success, no_quotes, error, ...) is treated as a completed attempt and reused,
    so re-running never re-spends the live-call budget on the same scenario.
  - The cache keeps every attempt (for resumability), but --output only ever
    receives status=="success" records — no_quotes/insufficient/error rows are
    counted in the summary but never written into the training-data output.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.engine.client import DEFAULT_MIN_DELAY, MIN_SUCCESS_QUOTES, QuoteEngineClient  # noqa: E402
from src.scenarios.generate import normalize_risk, scenario_id_for, validate_record  # noqa: E402

DATA_DIR = (ROOT / "data").resolve()
TOOLS_DIR = (ROOT / "tools").resolve()


def _assert_not_protected(path: Path) -> None:
    resolved = path.resolve()
    for protected in (DATA_DIR, TOOLS_DIR):
        if resolved == protected or protected in resolved.parents:
            raise SystemExit(f"refusing to write under protected path: {resolved} (under {protected})")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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


def _public_output_record(record: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal cache record to the stable downstream episode schema."""
    out = dict(record)
    risk = out.get("risk") or out.get("normalized_risk")
    if risk is None:
        raise SystemExit(f"cache record {out.get('scenario_id')} has no risk")
    out["risk"] = risk
    out.pop("normalized_risk", None)
    return out


@contextmanager
def _collector_lock(cache_dir: Path, enabled: bool):
    """Prevent concurrent live collectors from bypassing cache/rate-limit safety."""
    if not enabled:
        yield
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    lock_path = cache_dir / ".collector.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(
                f"another live quote collector is using {cache_dir}; refusing concurrent execution"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _validate_scenario_ids(records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for rec in records:
        validate_record(rec["risk"])
        recomputed = scenario_id_for(normalize_risk(rec["risk"]))
        if recomputed != rec["scenario_id"]:
            raise SystemExit(
                f"scenario_id mismatch for {rec.get('scenario_id')}: recomputed {recomputed} "
                "(input file may be corrupted or hand-edited)"
            )
        if recomputed in seen:
            raise SystemExit(f"duplicate scenario_id in input: {recomputed}")
        seen.add(recomputed)


def _status_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts = {s: 0 for s in ("success", "no_quotes", "insufficient_quotes", "transient_error", "permanent_error")}
    for rec in records:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    return counts


@dataclass
class CollectionResult:
    considered: int = 0
    cache_hits: int = 0
    uncached_calls: int = 0
    successful: int = 0
    no_quotes: int = 0
    insufficient_quotes: int = 0
    errors: int = 0
    stop_reason: str = "input_exhausted"
    new_records: list[dict[str, Any]] = field(default_factory=list)


def collect(
    input_records: list[dict[str, Any]],
    existing_output: list[dict[str, Any]],
    client: QuoteEngineClient,
    target: int,
    max_live_calls: int,
    execute: bool,
    retry_transient_errors: bool = False,
) -> CollectionResult:
    """Shared plan/execute loop. In dry-run (execute=False) this only ever calls
    client.read_cache() — a pure disk read — never the live engine, never sleeps."""
    existing_by_id = {r["scenario_id"] for r in existing_output}
    result = CollectionResult(successful=_status_counts(existing_output)["success"])

    for rec in input_records:
        sid = rec["scenario_id"]
        if sid in existing_by_id:
            continue
        if result.successful >= target:
            result.stop_reason = "target_reached"
            break
        normalized = normalize_risk(rec["risk"])
        cached = client.read_cache(sid)
        reusable = client.cache_is_reusable(cached, normalized, retry_transient_errors)
        if result.uncached_calls >= max_live_calls and not reusable:
            result.stop_reason = "max_live_calls_reached"
            break

        result.considered += 1
        if execute:
            record, was_cached = client.get_quotes(
                rec["risk"], retry_transient_errors=retry_transient_errors
            )
        else:
            if reusable:
                record, was_cached = cached, True
            else:
                result.uncached_calls += 1
                continue  # dry-run: never fetch live, never learn the hypothetical status

        if was_cached:
            result.cache_hits += 1
        else:
            result.uncached_calls += 1

        status = record["status"]
        if status == "success":
            result.successful += 1
            if execute:
                result.new_records.append(_public_output_record(record))
        elif status == "no_quotes":
            result.no_quotes += 1
        elif status == "insufficient_quotes":
            result.insufficient_quotes += 1
        else:
            result.errors += 1
    else:
        result.stop_reason = "input_exhausted"
    return result


def _print_plan(args: argparse.Namespace, result: CollectionResult, cache_dir: Path) -> None:
    total_cache_files = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0
    print("DRY RUN — zero live engine calls made, zero sleeps performed.")
    print(f"  input scenarios considered:        {result.considered}")
    print(f"  cache hits (reusable, any status):  {result.cache_hits}")
    print(f"  known successful (from cache):      {result.successful}")
    print(f"  uncached calls that WOULD be needed: {result.uncached_calls}")
    print(f"  stop reason:                        {result.stop_reason}")
    print(f"  target: {args.target}   max-live-calls: {args.max_live_calls}   delay: {args.delay}s")
    print(f"  total cache records on disk:        {total_cache_files}")
    print(f"  cache dir: {cache_dir}")
    if result.uncached_calls > 0:
        eta = result.uncached_calls * args.delay
        print(
            f"  if run with --execute-live: up to {result.uncached_calls} live call(s), "
            f"~{eta:.0f}s minimum at {args.delay}s/call"
        )


def _run_validate_only(output_path: Path) -> int:
    records = _load_jsonl(output_path)
    if not records:
        print(f"[FAIL] no records found at {output_path}")
        return 1

    seen_ids: set[str] = set()
    dup_problems: list[str] = []
    integrity_problems: list[str] = []
    non_success_problems: list[str] = []
    credential_hits: list[str] = []
    env_secrets = [v for v in (os.environ.get("MM_ID"), os.environ.get("MM_PA")) if v]
    url_query_re = re.compile(r"https?://[^\s\"']+?\?[^\s\"']*")

    for rec in records:
        sid = rec.get("scenario_id")
        if sid in seen_ids:
            dup_problems.append(f"duplicate scenario_id in output: {sid}")
        seen_ids.add(sid)

        risk = rec.get("risk") or rec.get("normalized_risk")
        if risk is None:
            integrity_problems.append(f"{sid}: missing risk")
        else:
            try:
                validate_record(risk)
                if scenario_id_for(normalize_risk(risk)) != sid:
                    integrity_problems.append(f"{sid}: scenario_id does not match risk")
            except (KeyError, ValueError) as exc:
                integrity_problems.append(f"{sid}: invalid risk: {exc}")

        raw = json.dumps(rec, ensure_ascii=False)
        if url_query_re.search(raw):
            credential_hits.append(f"{sid}: authenticated URL pattern found")
        for secret in env_secrets:
            if secret in raw:
                credential_hits.append(f"{sid}: environment secret value present in record")

        if rec.get("status") != "success":
            non_success_problems.append(f"{sid}: status={rec.get('status')!r} (only success belongs in --output)")
            continue
        quotes = rec.get("quotes", [])
        if len(quotes) < MIN_SUCCESS_QUOTES:
            integrity_problems.append(f"{sid}: success status but only {len(quotes)} quotes")
        if rec.get("quote_count") != len(quotes):
            integrity_problems.append(f"{sid}: quote_count {rec.get('quote_count')} != len(quotes) {len(quotes)}")
        tariff_ids = [q.get("tariff_id") for q in quotes]
        if any(not t for t in tariff_ids):
            integrity_problems.append(f"{sid}: empty tariff_id present")
        if len(set(tariff_ids)) != len(tariff_ids):
            integrity_problems.append(f"{sid}: duplicate tariff_ids within scenario")
        for q in quotes:
            if not isinstance(q.get("price_annual"), (int, float)) or q["price_annual"] <= 0:
                integrity_problems.append(f"{sid}: non-positive price_annual on tariff {q.get('tariff_id')}")
            if "reward" not in q or "reward_components" not in q:
                integrity_problems.append(f"{sid}: tariff {q.get('tariff_id')} missing reward/reward_components")
        rewards = [q.get("reward") for q in quotes]
        if rewards != sorted(rewards, reverse=True):
            integrity_problems.append(f"{sid}: quotes not sorted by descending reward")
        if rec.get("reward_recommended_tariff_id") not in tariff_ids:
            integrity_problems.append(
                f"{sid}: reward_recommended_tariff_id {rec.get('reward_recommended_tariff_id')!r} not among quotes"
            )

    checks = [
        ("no duplicate scenario IDs", not dup_problems, "; ".join(dup_problems)[:500]),
        ("quote/tariff/reward integrity for success records", not integrity_problems, "; ".join(integrity_problems)[:500]),
        ("no non-success records in output", not non_success_problems, "; ".join(non_success_problems)[:500]),
        ("no credentials or authenticated URLs in records", not credential_hits, "; ".join(credential_hits)[:500]),
    ]
    ok = all(passed for _, passed, _ in checks)
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    n_success = sum(1 for r in records if r.get("status") == "success")
    print(f"\nvalidate-only: {'PASS' if ok else 'FAIL'} ({len(records)} records, {n_success} successful)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "artifacts/scenarios/scenarios.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "artifacts/quoted/quoted_scenarios.jsonl"))
    ap.add_argument("--cache-dir", default=str(ROOT / "artifacts/cache/quotes"))
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--max-live-calls", type=int, default=20)
    ap.add_argument("--delay", type=float, default=DEFAULT_MIN_DELAY)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config-output", default=str(ROOT / "artifacts/quoted/collect_config.json"))
    ap.add_argument("--summary-output", default=str(ROOT / "artifacts/quoted/summary.json"))
    ap.add_argument("--validate-only", action="store_true", help="Validate an existing --output file; write nothing.")
    ap.add_argument(
        "--retry-transient-errors",
        action="store_true",
        help="Explicitly retry cached transient_error records; other cached statuses remain terminal.",
    )
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="(default) Plan only, no engine calls.")
    mode.add_argument("--execute-live", action="store_true", help="Actually call the live engine.")
    args = ap.parse_args()

    output_path = Path(args.output)

    if args.validate_only:
        return _run_validate_only(output_path)

    if args.max_live_calls <= 0:
        raise SystemExit(f"--max-live-calls must be positive, got {args.max_live_calls}")
    if args.delay < DEFAULT_MIN_DELAY:
        raise SystemExit(
            f"--delay={args.delay} is below the {DEFAULT_MIN_DELAY}s floor required between live engine "
            "calls (CLAUDE.md rule 2 / tools/README.md)."
        )

    execute = bool(args.execute_live)
    config_path = Path(args.config_output)
    summary_path = Path(args.summary_output)
    cache_dir = Path(args.cache_dir)
    paths_to_check = [config_path, summary_path, cache_dir]
    if execute:
        paths_to_check.append(output_path)
    for p in paths_to_check:
        _assert_not_protected(p)

    if execute and not os.environ.get("MM_ID"):
        raise SystemExit("MM_ID is not set in the environment; refusing live execution. (Its value is never printed.)")

    input_path = Path(args.input)
    input_records = _load_jsonl(input_path)
    if not input_records:
        raise SystemExit(f"no input scenarios found at {input_path}")
    _validate_scenario_ids(input_records)
    input_records = list(input_records)
    Random(args.seed).shuffle(input_records)

    config = {
        "input": str(input_path),
        "output": str(output_path),
        "cache_dir": str(cache_dir),
        "target": args.target,
        "max_live_calls": args.max_live_calls,
        "delay": args.delay,
        "seed": args.seed,
        "retry_transient_errors": args.retry_transient_errors,
        "mode": "execute-live" if execute else "dry-run",
    }
    _atomic_write_json(config_path, config)

    with _collector_lock(cache_dir, enabled=execute):
        existing_output = [_public_output_record(r) for r in _load_jsonl(output_path)]
        client = QuoteEngineClient(cache_dir=cache_dir, min_delay=args.delay)
        result = collect(
            input_records=input_records,
            existing_output=existing_output,
            client=client,
            target=args.target,
            max_live_calls=args.max_live_calls,
            execute=execute,
            retry_transient_errors=args.retry_transient_errors,
        )

        if not execute:
            _print_plan(args, result, cache_dir)
            summary = {
                "mode": "dry-run",
                "input_scenarios_considered": result.considered,
                "cache_hits": result.cache_hits,
                "known_successful_from_cache": result.successful,
                "uncached_calls_that_would_be_required": result.uncached_calls,
                "stop_reason": result.stop_reason,
                "cache_dir": str(cache_dir),
            }
            _atomic_write_json(summary_path, summary)
            return 0

        final_records = existing_output + result.new_records
        _atomic_write_jsonl(output_path, final_records)

        summary = {
            "mode": "execute-live",
            "input_scenarios_considered": result.considered,
            "cache_hits": result.cache_hits,
            "uncached_calls": result.uncached_calls,
            "successful_scenarios": result.successful,
            "no_quote_scenarios": result.no_quotes,
            "insufficient_quote_scenarios": result.insufficient_quotes,
            "errors": result.errors,
            "total_records_in_output": len(final_records),
            "stop_reason": result.stop_reason,
            "output_path": str(output_path),
            "cache_dir": str(cache_dir),
        }
        _atomic_write_json(summary_path, summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"\nNOTE: this run made {result.uncached_calls} live engine call(s).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
