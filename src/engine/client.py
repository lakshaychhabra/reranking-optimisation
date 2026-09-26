#!/usr/bin/env python3
"""src/engine/client.py — file-cached, rate-limited wrapper around the quote tools.

Every live call to the Mr-Money Rechenkern must go through `QuoteEngineClient`.
Nothing here reads or prints MM_ID / MM_PA; the wrapped tool reads them from the
environment. One cache record per scenario, under <cache_dir>/<scenario_id>.json.
`min_delay` can never be configured below DEFAULT_MIN_DELAY (3s) — a library-level
floor, independent of whatever a caller passes in, to protect the live engine.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TOOLS_DIR = _ROOT / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from src.scenarios.generate import normalize_risk, scenario_id_for  # noqa: E402
from assess_wg_risk_context import assess_wg_risk_context  # noqa: E402
from calculate_quotes_wg import calculate_quotes_wg  # noqa: E402
from reward import score_quotes  # noqa: E402

DEFAULT_CACHE_DIR = _ROOT / "artifacts" / "cache" / "quotes"
DEFAULT_MIN_DELAY = 3.0  # seconds; CLAUDE.md rule 2 — never call the live engine faster than this
MIN_SUCCESS_QUOTES = 3

STATUSES = ("success", "no_quotes", "insufficient_quotes", "transient_error", "permanent_error")

_URL_QUERY_RE = re.compile(r"(https?://[^\s\"']+?)\?[^\s\"']*")
_CRED_PARAM_RE = re.compile(r"\b(id|pa)=([^&\s\"']+)", re.IGNORECASE)


def _sanitize_error(message: str) -> str:
    """Strip anything that could resemble a credential, query string or auth header."""
    message = _URL_QUERY_RE.sub(r"\1?[redacted]", message)
    message = _CRED_PARAM_RE.sub(r"\1=[redacted]", message)
    for env_var in ("MM_ID", "MM_PA"):
        value = os.environ.get(env_var)
        if value:
            message = message.replace(value, "[redacted]")
    return message[:500]


def _looks_transient(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, ConnectionError, OSError)):
        return True
    text = str(error).lower()
    return "timed out" in text or "temporarily unavailable" in text or "connection" in text


def coverage_policy(risk_context: dict[str, Any]) -> dict[str, bool]:
    """fire+glass always requested; natural_hazard only on elevated/high flood exposure."""
    flood = risk_context["hazard_exposure"]["flood_heavy_rain"]
    return {"fire": True, "glass": True, "natural_hazard": flood in ("high", "elevated")}


def _atomic_write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class QuoteEngineClient:
    """Disk-cached, rate-limited wrapper around calculate_quotes_wg / assess_wg_risk_context / score_quotes."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, min_delay: float = DEFAULT_MIN_DELAY):
        self.cache_dir = Path(cache_dir)
        self.min_delay = max(float(min_delay), DEFAULT_MIN_DELAY)
        self._last_call_at: float | None = None

    def cache_path(self, scenario_id: str) -> Path:
        return self.cache_dir / f"{scenario_id}.json"

    def read_cache(self, scenario_id: str) -> dict[str, Any] | None:
        """Pure read: no engine call, no sleep. Returns None if missing or unreadable."""
        path = self.cache_path(scenario_id)
        if not path.exists():
            return None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            return record if isinstance(record, dict) else None
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def cache_is_reusable(
        cached: dict[str, Any] | None,
        normalized_risk: dict[str, Any],
        retry_transient_errors: bool = False,
    ) -> bool:
        if cached is None or cached.get("normalized_risk") != normalized_risk:
            return False
        status = cached.get("status")
        if status not in STATUSES:
            return False
        return not (retry_transient_errors and status == "transient_error")

    def _throttle(self) -> None:
        if self._last_call_at is not None:
            wait = self.min_delay - (time.monotonic() - self._last_call_at)
            if wait > 0:
                time.sleep(wait)
        self._last_call_at = time.monotonic()

    def get_quotes(
        self,
        risk: dict[str, Any],
        use_cache_only: bool = False,
        retry_transient_errors: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        """Return (record, was_cached). Only reaches the live engine on a genuine cache miss.

        Any cached record (any status) for a matching normalized risk counts as reusable —
        a completed attempt is never silently retried, so a scenario with e.g. no_quotes
        doesn't quietly burn the live-call budget on repeat runs.
        """
        normalized_risk = normalize_risk(risk)
        scenario_id = scenario_id_for(normalized_risk)
        cached = self.read_cache(scenario_id)
        if self.cache_is_reusable(cached, normalized_risk, retry_transient_errors):
            return cached, True
        if use_cache_only:
            raise LookupError(f"no reusable cache record for scenario {scenario_id}")
        record = self._fetch_live(normalized_risk, scenario_id)
        _atomic_write_json(self.cache_path(scenario_id), record)
        return record, False

    def _fetch_live(self, normalized_risk: dict[str, Any], scenario_id: str) -> dict[str, Any]:
        self._throttle()
        risk_context = assess_wg_risk_context(normalized_risk)
        requested_coverage = coverage_policy(risk_context)
        engine_risk = {
            **normalized_risk,
            "cover_fire": requested_coverage["fire"],
            "cover_glass": requested_coverage["glass"],
            "cover_natural_hazard": requested_coverage["natural_hazard"],
        }
        base = {
            "scenario_id": scenario_id,
            "risk": normalized_risk,
            "normalized_risk": normalized_risk,
            "engine_risk": engine_risk,
            "requested_coverage": requested_coverage,
            "risk_context": risk_context,
            "attempted_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            quotes = calculate_quotes_wg(engine_risk)
        except Exception as e:  # noqa: BLE001 — one failed scenario must not crash the batch
            status = "transient_error" if _looks_transient(e) else "permanent_error"
            return {
                **base,
                "status": status,
                "quote_count": 0,
                "quotes": [],
                "reward_recommended_tariff_id": None,
                "error": _sanitize_error(str(e)),
            }
        if not quotes:
            status = "no_quotes"
        elif len(quotes) < MIN_SUCCESS_QUOTES:
            status = "insufficient_quotes"
        else:
            status = "success"
        scored = score_quotes(quotes) if quotes else []
        return {
            **base,
            "status": status,
            "quote_count": len(scored),
            "quotes": scored,
            "reward_recommended_tariff_id": scored[0]["tariff_id"] if status == "success" else None,
            "error": None,
        }
