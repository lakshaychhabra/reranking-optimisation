#!/usr/bin/env python3
"""Create and manage GPT-6 Sol silver-label annotation batches.

The quoted dataset remains immutable. This script writes sanitized Batch API
requests, validation metadata, raw provider output, and validated annotations
under artifacts/teacher/.

Typical workflow:
  python3 scripts/annotate_teacher_batch.py prepare
  python3 scripts/annotate_teacher_batch.py submit --shard 0 --confirm-submit
  python3 scripts/annotate_teacher_batch.py status --shard 0
  python3 scripts/annotate_teacher_batch.py download --shard 0
  # Repeat submit/status/download for each shard, then:
  python3 scripts/annotate_teacher_batch.py merge

Credentials are read from OPENAI_KEY (or OPENAI_API_KEY) in the repository .env
or from the process environment. Preparing and merging are offline operations.
Only ``submit``, ``status``, and ``download`` contact OpenAI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# try:
from openai import OpenAI
# except ImportError:  # pragma: no cover - only reached in an incomplete runtime
#     OpenAI = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BROKER_SYSTEM, build_broker_prompt  # noqa: E402

DEFAULT_INPUT = ROOT / "artifacts/quoted/final_quoted_scenarios.jsonl"
DEFAULT_WORK_DIR = ROOT / "artifacts/teacher"
MODEL = "gpt-6-sol"
ENDPOINT = "/v1/responses"
BATCH_INPUT_USD_PER_MTOK = 1.00
BATCH_OUTPUT_USD_PER_MTOK = 5.00

SYSTEM = BROKER_SYSTEM


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if not match:
                continue
            value = match.group(2).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[match.group(1)] = value
    values.update(os.environ)
    return values


def client() -> Any:
    if OpenAI is None:
        raise SystemExit("The openai Python package is required: pip install openai")
    env = load_env(ROOT / ".env")
    key = env.get("OPENAI_KEY") or env.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("Missing OPENAI_KEY (or OPENAI_API_KEY) in .env or the environment")
    return OpenAI(api_key=key)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"file not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON at {path}:{line_number}: {exc}") from exc
    return records


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def load_plan(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "batch_plan.json"
    if not path.exists():
        raise SystemExit(f"batch plan not found: {path}; run prepare first")
    return json.loads(path.read_text(encoding="utf-8"))


def save_plan(work_dir: Path, plan: dict[str, Any]) -> None:
    plan["updated_at"] = utc_now()
    write_json(work_dir / "batch_plan.json", plan)


def validate_source(records: list[dict[str, Any]], expected_count: int | None) -> None:
    if expected_count is not None and len(records) != expected_count:
        raise SystemExit(f"expected {expected_count} quoted scenarios, found {len(records)}")
    seen: set[str] = set()
    for index, record in enumerate(records):
        sid = record.get("scenario_id")
        if not isinstance(sid, str) or not sid:
            raise SystemExit(f"record {index} has no scenario_id")
        if sid in seen:
            raise SystemExit(f"duplicate scenario_id: {sid}")
        seen.add(sid)
        if record.get("status") != "success":
            raise SystemExit(f"{sid}: teacher input must contain success records only")
        quotes = record.get("quotes")
        if not isinstance(quotes, list) or len(quotes) < 3:
            raise SystemExit(f"{sid}: fewer than three quotes")
        tariff_ids = [str(q.get("tariff_id", "")) for q in quotes]
        if any(not tariff_id for tariff_id in tariff_ids) or len(set(tariff_ids)) != len(tariff_ids):
            raise SystemExit(f"{sid}: tariff IDs must be non-empty and unique")


def response_schema(tariff_ids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "top_3": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                        "tariff_id": {"type": "string", "enum": tariff_ids},
                        "rationale": {"type": "string"},
                    },
                    "required": ["rank", "tariff_id", "rationale"],
                    "additionalProperties": False,
                },
            },
            "overall_rationale": {"type": "string"},
        },
        "required": ["top_3", "overall_rationale"],
        "additionalProperties": False,
    }


def build_request(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], int]:
    prompt = build_broker_prompt(record, include_rationales=True)
    tariff_metadata = {
        str(quote["tariff_id"]): {
            "insurer": quote.get("insurer"),
            "product": quote.get("product"),
        }
        for quote in record["quotes"]
    }
    schema = response_schema(list(tariff_metadata))
    body = {
        "model": MODEL,
        "instructions": SYSTEM,
        "input": prompt,
        "reasoning": {"effort": "low"},
        "max_output_tokens": 3000,
        "store": False,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "wg_teacher_annotation",
                "strict": True,
                "schema": schema,
            }
        },
    }
    request = {"custom_id": record["scenario_id"], "method": "POST", "url": ENDPOINT, "body": body}
    # Empirical eval prompts averaged almost exactly one token per three chars.
    estimated_input_tokens = math.ceil(
        (len(SYSTEM) + len(prompt) + len(json.dumps(schema, ensure_ascii=False))) / 3.0
    )
    manifest = {
        "scenario_id": record["scenario_id"],
        "tariff_metadata": tariff_metadata,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
    }
    return request, manifest, estimated_input_tokens


def command_prepare(args: argparse.Namespace) -> None:
    source = Path(args.input).resolve()
    work_dir = Path(args.work_dir).resolve()
    plan_path = work_dir / "batch_plan.json"
    if plan_path.exists() and not args.force:
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if any(shard.get("batch_id") for shard in old.get("shards", [])):
            raise SystemExit("existing plan has submitted batches; refuse to overwrite without --force")

    records = read_jsonl(source)
    validate_source(records, args.expected_count if args.limit is None else None)
    if args.limit is not None:
        if args.limit <= 0 or args.limit > len(records):
            raise SystemExit(f"--limit must be between 1 and {len(records)}")
        records = records[: args.limit]

    requests: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    estimated_input_tokens = 0
    for record in records:
        request, mapping, estimate = build_request(record)
        requests.append(request)
        manifest.append(mapping)
        estimated_input_tokens += estimate

    shard_size = args.shard_size
    if shard_size <= 0:
        raise SystemExit("--shard-size must be positive")
    shards: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(requests), shard_size)):
        shard_requests = requests[start : start + shard_size]
        request_path = work_dir / "requests" / f"batch_{shard_index:03d}.jsonl"
        write_jsonl(request_path, shard_requests)
        shards.append(
            {
                "index": shard_index,
                "request_path": str(request_path),
                "request_count": len(shard_requests),
                "request_sha256": sha256_file(request_path),
                "status": "prepared",
                "batch_id": None,
                "input_file_id": None,
                "output_file_id": None,
                "error_file_id": None,
            }
        )

    for item_index, item in enumerate(manifest):
        item["source_index"] = item_index
        item["shard"] = item_index // shard_size
    write_jsonl(work_dir / "manifest.jsonl", manifest)

    estimated_output_tokens = len(records) * args.estimated_output_tokens
    estimated_cost = (
        estimated_input_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MTOK
        + estimated_output_tokens / 1_000_000 * BATCH_OUTPUT_USD_PER_MTOK
    )
    plan = {
        "model": MODEL,
        "endpoint": ENDPOINT,
        "reasoning_effort": "low",
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "scenario_count": len(records),
        "shard_size": shard_size,
        "created_at": utc_now(),
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "estimated_output_tokens_per_scenario": args.estimated_output_tokens,
        "estimated_batch_cost_usd": round(estimated_cost, 4),
        "batch_prices_usd_per_mtok": {
            "input": BATCH_INPUT_USD_PER_MTOK,
            "output": BATCH_OUTPUT_USD_PER_MTOK,
        },
        "shards": shards,
    }
    save_plan(work_dir, plan)
    print(f"prepared {len(records)} requests in {len(shards)} shard(s) -> {work_dir}")
    print(
        f"estimated tokens: {estimated_input_tokens:,} input + "
        f"{estimated_output_tokens:,} output; estimated Batch cost: ${estimated_cost:.2f}"
    )


def get_shard(plan: dict[str, Any], index: int) -> dict[str, Any]:
    shards = plan.get("shards", [])
    if index < 0 or index >= len(shards):
        raise SystemExit(f"invalid shard {index}; choose 0..{len(shards) - 1}")
    return shards[index]


def command_submit(args: argparse.Namespace) -> None:
    if not args.confirm_submit:
        raise SystemExit("submission makes paid API calls; re-run with --confirm-submit")
    work_dir = Path(args.work_dir).resolve()
    plan = load_plan(work_dir)
    shard = get_shard(plan, args.shard)
    if shard.get("batch_id") and not args.allow_resubmit:
        raise SystemExit(
            f"shard {args.shard} already has batch {shard['batch_id']}; "
            "use status/download or explicitly pass --allow-resubmit"
        )
    request_path = Path(shard["request_path"])
    if sha256_file(request_path) != shard["request_sha256"]:
        raise SystemExit(f"request shard hash changed: {request_path}; run prepare again")

    api = client()
    with request_path.open("rb") as fh:
        uploaded = api.files.create(file=fh, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id,
        endpoint=ENDPOINT,
        completion_window="24h",
        metadata={"purpose": "wg-silver-teacher", "shard": str(args.shard)},
    )
    shard.update(
        {
            "input_file_id": uploaded.id,
            "batch_id": batch.id,
            "status": str(batch.status),
            "submitted_at": utc_now(),
        }
    )
    save_plan(work_dir, plan)
    print(f"submitted shard {args.shard}: {batch.id} ({batch.status})")


def refresh_shard(api: Any, shard: dict[str, Any]) -> Any:
    if not shard.get("batch_id"):
        raise SystemExit(f"shard {shard['index']} has not been submitted")
    batch = api.batches.retrieve(shard["batch_id"])
    shard.update(
        {
            "status": str(batch.status),
            "output_file_id": batch.output_file_id,
            "error_file_id": batch.error_file_id,
            "request_counts": batch.request_counts.model_dump() if batch.request_counts else None,
            "last_checked_at": utc_now(),
        }
    )
    return batch


def command_status(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    plan = load_plan(work_dir)
    shard = get_shard(plan, args.shard)
    batch = refresh_shard(client(), shard)
    save_plan(work_dir, plan)
    counts = batch.request_counts.model_dump() if batch.request_counts else {}
    print(f"shard {args.shard}: {batch.id} status={batch.status} counts={counts}")


def extract_output_text(body: dict[str, Any]) -> str:
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def validate_annotation(
    scenario_id: str,
    annotation: dict[str, Any],
    tariff_metadata: dict[str, dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    top = annotation.get("top_3")
    if not isinstance(top, list) or len(top) != 3:
        raise ValueError("top_3 must contain exactly three items")
    ranks = [item.get("rank") for item in top if isinstance(item, dict)]
    tariff_ids = [str(item.get("tariff_id")) for item in top if isinstance(item, dict)]
    if ranks != [1, 2, 3]:
        raise ValueError(f"ranks must be [1, 2, 3], got {ranks}")
    if len(tariff_ids) != 3 or len(set(tariff_ids)) != 3:
        raise ValueError("top_3 tariff IDs must be present and unique")
    if any(tariff_id not in tariff_metadata for tariff_id in tariff_ids):
        raise ValueError(f"unknown tariff ID in {tariff_ids}")
    if any(not isinstance(item.get("rationale"), str) or not item["rationale"].strip() for item in top):
        raise ValueError("every top_3 item needs a non-empty rationale")
    overall = annotation.get("overall_rationale")
    if not isinstance(overall, str) or not overall.strip():
        raise ValueError("overall_rationale must be non-empty")
    return {
        "id": scenario_id,
        "top_3": [
            {
                "rank": item["rank"],
                "tariff_id": str(item["tariff_id"]),
                "insurer": tariff_metadata[str(item["tariff_id"])]["insurer"],
                "product": tariff_metadata[str(item["tariff_id"])]["product"],
                "rationale": item["rationale"].strip(),
            }
            for item in top
        ],
        "overall_rationale": overall.strip(),
        "annotator_model": model,
    }


def command_download(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    plan = load_plan(work_dir)
    shard = get_shard(plan, args.shard)
    api = client()
    batch = refresh_shard(api, shard)
    save_plan(work_dir, plan)
    if str(batch.status) != "completed" or not batch.output_file_id:
        raise SystemExit(f"shard {args.shard} is {batch.status}; output is not ready")

    raw = api.files.content(batch.output_file_id).read()
    raw_path = work_dir / "raw" / f"batch_{args.shard:03d}_output.jsonl"
    atomic_write_text(raw_path, raw.decode("utf-8"))
    if batch.error_file_id:
        error_raw = api.files.content(batch.error_file_id).read()
        atomic_write_text(work_dir / "raw" / f"batch_{args.shard:03d}_errors.jsonl", error_raw.decode("utf-8"))

    manifest_records = read_jsonl(work_dir / "manifest.jsonl")
    manifest = {
        item["scenario_id"]: item
        for item in manifest_records
        if item["shard"] == args.shard
    }
    annotations: list[dict[str, Any]] = []
    validation_errors: list[dict[str, Any]] = []
    input_tokens = output_tokens = 0
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        sid = item.get("custom_id")
        try:
            if sid not in manifest:
                raise ValueError("custom_id is absent from this shard's manifest")
            response = item.get("response") or {}
            if response.get("status_code") != 200:
                raise ValueError(f"HTTP status {response.get('status_code')}: {item.get('error')}")
            body = response.get("body") or {}
            usage = body.get("usage") or {}
            input_tokens += int(usage.get("input_tokens", 0))
            output_tokens += int(usage.get("output_tokens", 0))
            text = extract_output_text(body)
            if not text:
                raise ValueError("response contains no output_text")
            parsed = json.loads(text)
            annotations.append(
                validate_annotation(
                    sid,
                    parsed,
                    manifest[sid]["tariff_metadata"],
                    body.get("model", MODEL),
                )
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            validation_errors.append({"line": line_number, "scenario_id": sid, "error": str(exc)})

    source_order = {item["scenario_id"]: item["source_index"] for item in manifest_records}
    annotations.sort(key=lambda item: source_order[item["id"]])
    annotation_path = work_dir / "annotations" / f"batch_{args.shard:03d}.jsonl"
    write_jsonl(annotation_path, annotations)
    write_json(work_dir / "annotations" / f"batch_{args.shard:03d}_errors.json", validation_errors)
    actual_cost = (
        input_tokens / 1_000_000 * BATCH_INPUT_USD_PER_MTOK
        + output_tokens / 1_000_000 * BATCH_OUTPUT_USD_PER_MTOK
    )
    shard.update(
        {
            "status": "downloaded",
            "downloaded_at": utc_now(),
            "valid_annotations": len(annotations),
            "validation_errors": len(validation_errors),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "actual_cost_usd": round(actual_cost, 6),
            "annotation_path": str(annotation_path),
            "raw_output_path": str(raw_path),
        }
    )
    save_plan(work_dir, plan)
    print(
        f"downloaded shard {args.shard}: {len(annotations)} valid, "
        f"{len(validation_errors)} invalid; actual Batch cost ${actual_cost:.4f}"
    )


def command_merge(args: argparse.Namespace) -> None:
    work_dir = Path(args.work_dir).resolve()
    plan = load_plan(work_dir)
    manifest = read_jsonl(work_dir / "manifest.jsonl")
    source_order = {item["scenario_id"]: item["source_index"] for item in manifest}
    annotations: list[dict[str, Any]] = []
    for shard in plan["shards"]:
        annotation_path = shard.get("annotation_path")
        if not annotation_path:
            raise SystemExit(f"shard {shard['index']} has not been downloaded")
        annotations.extend(read_jsonl(Path(annotation_path)))
    ids = [item["id"] for item in annotations]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate scenario IDs across annotation shards")
    if len(annotations) != plan["scenario_count"]:
        raise SystemExit(
            f"expected {plan['scenario_count']} valid annotations, found {len(annotations)}; "
            "resolve shard errors before merging"
        )
    annotations.sort(key=lambda item: source_order[item["id"]])
    output = Path(args.output).resolve()
    write_jsonl(output, annotations)
    total_input = sum(int(shard.get("input_tokens", 0)) for shard in plan["shards"])
    total_output = sum(int(shard.get("output_tokens", 0)) for shard in plan["shards"])
    total_cost = (
        total_input / 1_000_000 * BATCH_INPUT_USD_PER_MTOK
        + total_output / 1_000_000 * BATCH_OUTPUT_USD_PER_MTOK
    )
    summary = {
        "model": MODEL,
        "annotation_count": len(annotations),
        "input_tokens": total_input,
        "output_tokens": total_output,
        "actual_batch_cost_usd": round(total_cost, 6),
        "source_sha256": plan["source_sha256"],
        "annotations_sha256": sha256_file(output),
        "created_at": utc_now(),
    }
    write_json(work_dir / "annotation_summary.json", summary)
    print(f"merged {len(annotations)} annotations -> {output} (actual cost ${total_cost:.4f})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Build offline Batch API JSONL shards")
    prepare.add_argument("--input", default=str(DEFAULT_INPUT))
    prepare.add_argument("--expected-count", type=int, default=1490)
    prepare.add_argument("--limit", type=int, help="Prepare only the first N records for a pilot")
    prepare.add_argument("--shard-size", type=int, default=745)
    prepare.add_argument("--estimated-output-tokens", type=int, default=450)
    prepare.add_argument("--force", action="store_true")
    prepare.set_defaults(func=command_prepare)

    submit = subparsers.add_parser("submit", help="Upload and submit one paid batch shard")
    submit.add_argument("--shard", type=int, required=True)
    submit.add_argument("--confirm-submit", action="store_true")
    submit.add_argument("--allow-resubmit", action="store_true")
    submit.set_defaults(func=command_submit)

    status = subparsers.add_parser("status", help="Refresh one batch shard's status")
    status.add_argument("--shard", type=int, required=True)
    status.set_defaults(func=command_status)

    download = subparsers.add_parser("download", help="Download and validate a completed shard")
    download.add_argument("--shard", type=int, required=True)
    download.set_defaults(func=command_download)

    merge = subparsers.add_parser("merge", help="Merge all validated annotation shards")
    merge.add_argument("--output", default=str(DEFAULT_WORK_DIR / "annotations.jsonl"))
    merge.set_defaults(func=command_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
