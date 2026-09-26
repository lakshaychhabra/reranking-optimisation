"""Shared, dependency-light helpers for training and local evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = ROOT / "data" / "golden_wg_recommendations.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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
        raise SystemExit(f"no records found: {path}")
    return records


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record_id(record: dict[str, Any]) -> str:
    value = record.get("scenario_id", record.get("id"))
    if value is None:
        raise ValueError("record has neither scenario_id nor id")
    return str(value)


def annotation_top3(record: dict[str, Any]) -> list[str]:
    annotation = record.get("teacher_annotation") or record.get("fable_annotation") or {}
    top = annotation.get("top_3") if isinstance(annotation, dict) else None
    if not isinstance(top, list):
        return []
    ranked: list[tuple[int, str]] = []
    for position, item in enumerate(top, 1):
        if not isinstance(item, dict) or item.get("tariff_id") is None:
            return []
        try:
            rank = int(item.get("rank", position))
        except (TypeError, ValueError):
            return []
        ranked.append((rank, str(item["tariff_id"])))
    return [tariff_id for _, tariff_id in sorted(ranked)][:3]


def ranking_json(tariff_ids: list[str]) -> str:
    if len(tariff_ids) != 3 or len(set(tariff_ids)) != 3:
        raise ValueError("a training target must contain exactly three unique tariff IDs")
    payload = {
        "top_3": [
            {"rank": rank, "tariff_id": tariff_id}
            for rank, tariff_id in enumerate(tariff_ids, 1)
        ]
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_top3(text: str) -> tuple[list[str], bool]:
    """Return leniently extracted IDs and whether the exact output is valid strict JSON."""
    if not text:
        return [], False
    stripped = text.strip()
    strict = False
    try:
        value = json.loads(stripped)
        top = value.get("top_3") if isinstance(value, dict) else None
        if isinstance(top, list) and len(top) == 3:
            ranks = [item.get("rank") for item in top if isinstance(item, dict)]
            ids = [str(item["tariff_id"]) for item in top if isinstance(item, dict) and item.get("tariff_id") is not None]
            exact_item_schema = all(
                isinstance(item, dict)
                and set(item) == {"rank", "tariff_id"}
                and isinstance(item.get("rank"), int)
                and not isinstance(item.get("rank"), bool)
                and isinstance(item.get("tariff_id"), str)
                for item in top
            )
            strict = (
                ranks == [1, 2, 3]
                and len(ids) == 3
                and len(set(ids)) == 3
                and set(value) == {"top_3"}
                and exact_item_schema
            )
            if len(ids) == 3:
                return ids, strict
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            top = value.get("top_3") or value.get("recommendations") or []
            ids = [
                str(item.get("tariff_id")) if isinstance(item, dict) else str(item)
                for item in top
                if (isinstance(item, dict) and item.get("tariff_id") is not None) or not isinstance(item, dict)
            ]
            if ids:
                return ids[:3], False
        except (json.JSONDecodeError, AttributeError):
            pass
    return re.findall(r'"tariff_id"\s*:\s*"?([\w:-]+)"?', stripped)[:3], False


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]
