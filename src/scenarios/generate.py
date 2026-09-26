#!/usr/bin/env python3
"""src/scenarios/generate.py — reproducible synthetic WG risk generator.

Builds diverse WgRisk-shaped scenario records for training-data collection.
Only the `risk` field of `data/golden_wg_recommendations.jsonl` is ever read
here (to deduplicate against it) — `fable_annotation` is never touched.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from random import Random
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

from assess_wg_risk_context import assess_wg_risk_context  # noqa: E402
from calculate_quotes_wg import (  # noqa: E402
    ATTIC_WIRE,
    BUILDING_CLASS_WIRE,
    BUILDING_TYPE_WIRE,
    ROOF_WIRE,
)

# The normalized base-risk schema, in a fixed order. Sourced from the same enums
# tools/calculate_quotes_wg.py accepts, so this can never drift from the live tool.
RISK_FIELDS: tuple[str, ...] = (
    "postal_code",
    "building_type",
    "construction_class",
    "construction_year",
    "roof_type",
    "attic_status",
    "living_space_sqm",
    "dwelling_units",
    "owner_occupied",
    "rebuild_sum",
    "with_deductible",
)

BUILDING_TYPES: tuple[str, ...] = tuple(BUILDING_TYPE_WIRE.keys())
CONSTRUCTION_CLASSES: tuple[str, ...] = tuple(BUILDING_CLASS_WIRE.keys())
ROOF_TYPES: tuple[str, ...] = tuple(ROOF_WIRE.keys())
ATTIC_STATUSES: tuple[str, ...] = tuple(ATTIC_WIRE.keys())

HAZARD_STRATA: tuple[str, ...] = ("moderate", "elevated", "high")
AGE_STRATA: tuple[str, ...] = ("old", "middle", "modern")

# Known-valid German city-centre postcodes supplied by the original benchmark
# generator. We reuse the locations, never the golden building risks or labels.
# Sampling arbitrary suffixes from a two-digit prefix can create syntactically
# valid but nonexistent postcodes, which is unsuitable for training data.
KNOWN_VALID_POSTCODES: tuple[str, ...] = (
    "01067", "04109", "07743", "10115", "10585", "13353", "20095", "21073", "22765", "26122",
    "27568", "28195", "30159", "33602", "35091", "37073", "40213", "44135", "45127", "48143",
    "50667", "51063", "53113", "55116", "56068", "60311", "63065", "65183", "67059", "68159",
    "70173", "71065", "72074", "74072", "76133", "77652", "79098", "80331", "81667", "83022",
    "84028", "85049", "86150", "88045", "89073", "90402", "93047", "94032", "97070", "99084",
)

MIN_YEAR = 1900
MAX_YEAR = 2020
AGE_STRATUM_RANGES: dict[str, tuple[int, int]] = {
    "old": (1900, 1959),
    "middle": (1960, 1994),
    "modern": (1995, 2020),
}

# Living space (sqm) ranges per building type; multi-family scales per dwelling unit instead.
LIVING_SPACE_RANGES: dict[str, tuple[int, int]] = {
    "single-family": (70, 260),
    "two-family": (120, 320),
    "semi-detached": (80, 200),
    "terraced": (75, 190),
}
MULTI_FAMILY_UNIT_SQM_RANGE = (45, 90)
MULTI_FAMILY_UNITS_RANGE = (3, 8)
SEMI_TERRACED_TWO_UNIT_PROB = 0.15

COST_PER_SQM_RANGE = (1500, 3200)
MIN_REBUILD_SUM = 40_000
MAX_REBUILD_SUM = 3_000_000
MIN_COST_PER_SQM = 800
MAX_COST_PER_SQM = 5_000


class ScenarioValidationError(ValueError):
    """A generated (or golden) risk record failed a validation check."""


# ---- Canonical hashing -----------------------------------------------------

def canonical_json(risk: dict[str, Any]) -> str:
    return json.dumps(risk, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def scenario_id_for(risk: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(risk).encode("utf-8")).hexdigest()


def normalize_risk(risk: dict[str, Any]) -> dict[str, Any]:
    """Pick exactly the base-risk fields, in a fixed key order."""
    validate_schema(risk)
    return {k: risk[k] for k in RISK_FIELDS}


def load_golden_risk_hashes(golden_path: Path) -> set[str]:
    """Hash only the `risk` field of each golden record; never reads fable_annotation."""
    hashes: set[str] = set()
    if not golden_path.exists():
        return hashes
    with golden_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            risk = rec.get("risk")
            if risk is not None:
                hashes.add(scenario_id_for(normalize_risk(risk)))
    return hashes


# ---- Validation -------------------------------------------------------------

def validate_schema(risk: dict[str, Any]) -> None:
    keys = set(risk.keys())
    expected = set(RISK_FIELDS)
    missing = expected - keys
    extra = keys - expected
    if missing:
        raise ScenarioValidationError(f"risk missing fields: {sorted(missing)}")
    if extra:
        raise ScenarioValidationError(f"risk has unexpected fields: {sorted(extra)}")


def validate_enums(risk: dict[str, Any]) -> None:
    if risk["building_type"] not in BUILDING_TYPES:
        raise ScenarioValidationError(f"invalid building_type: {risk['building_type']!r}")
    if risk["construction_class"] not in CONSTRUCTION_CLASSES:
        raise ScenarioValidationError(f"invalid construction_class: {risk['construction_class']!r}")
    if risk["roof_type"] not in ROOF_TYPES:
        raise ScenarioValidationError(f"invalid roof_type: {risk['roof_type']!r}")
    if risk["attic_status"] not in ATTIC_STATUSES:
        raise ScenarioValidationError(f"invalid attic_status: {risk['attic_status']!r}")


def validate_construction_year(risk: dict[str, Any]) -> None:
    year = risk["construction_year"]
    if not isinstance(year, int) or isinstance(year, bool) or not (MIN_YEAR <= year <= MAX_YEAR):
        raise ScenarioValidationError(f"invalid construction_year: {year!r} (expected {MIN_YEAR}-{MAX_YEAR})")


def validate_dwelling_units(risk: dict[str, Any]) -> None:
    btype = risk["building_type"]
    units = risk["dwelling_units"]
    if not isinstance(units, int) or isinstance(units, bool) or units < 1:
        raise ScenarioValidationError(f"invalid dwelling_units: {units!r}")
    if btype == "single-family" and units != 1:
        raise ScenarioValidationError(f"single-family must have 1 dwelling unit, got {units}")
    if btype == "two-family" and units != 2:
        raise ScenarioValidationError(f"two-family must have 2 dwelling units, got {units}")
    if btype == "multi-family" and not (3 <= units <= 8):
        raise ScenarioValidationError(f"multi-family must have 3-8 dwelling units, got {units}")
    if btype in ("semi-detached", "terraced") and units not in (1, 2):
        raise ScenarioValidationError(f"{btype} must have 1 or 2 dwelling units, got {units}")


def validate_rebuild_sum(risk: dict[str, Any]) -> None:
    rebuild_sum = risk["rebuild_sum"]
    sqm = risk["living_space_sqm"]
    if not isinstance(rebuild_sum, int) or isinstance(rebuild_sum, bool) or rebuild_sum <= 0:
        raise ScenarioValidationError(f"invalid rebuild_sum: {rebuild_sum!r}")
    if not (MIN_REBUILD_SUM <= rebuild_sum <= MAX_REBUILD_SUM):
        raise ScenarioValidationError(
            f"implausible rebuild_sum: {rebuild_sum} (expected {MIN_REBUILD_SUM}-{MAX_REBUILD_SUM})"
        )
    if not isinstance(sqm, int) or isinstance(sqm, bool) or sqm <= 0:
        raise ScenarioValidationError(f"invalid living_space_sqm: {sqm!r}")
    cost_per_sqm = rebuild_sum / sqm
    if not (MIN_COST_PER_SQM <= cost_per_sqm <= MAX_COST_PER_SQM):
        raise ScenarioValidationError(
            f"implausible rebuild_sum: {rebuild_sum} for {sqm}sqm "
            f"({cost_per_sqm:.0f} EUR/sqm, expected {MIN_COST_PER_SQM}-{MAX_COST_PER_SQM})"
        )


def validate_record(risk: dict[str, Any]) -> None:
    validate_schema(risk)
    validate_enums(risk)
    validate_construction_year(risk)
    validate_dwelling_units(risk)
    validate_rebuild_sum(risk)


def validate_batch(
    records: list[dict[str, Any]],
    golden_hashes: set[str] | None = None,
    expected_n: int | None = None,
) -> None:
    golden_hashes = golden_hashes or set()
    seen_ids: set[str] = set()
    hazard_strata: set[str] = set()
    age_strata: set[str] = set()
    for rec in records:
        risk = rec["risk"]
        validate_record(risk)
        sid = rec["scenario_id"]
        recomputed = scenario_id_for(normalize_risk(risk))
        if sid != recomputed:
            raise ScenarioValidationError(f"scenario_id mismatch: stored {sid} != recomputed {recomputed}")
        if sid in seen_ids:
            raise ScenarioValidationError(f"duplicate scenario_id: {sid}")
        seen_ids.add(sid)
        if sid in golden_hashes:
            raise ScenarioValidationError(f"scenario collides with a golden risk: {sid}")
        meta = rec.get("generation_metadata", {})
        hazard_strata.add(meta.get("hazard_stratum"))
        age_strata.add(meta.get("age_stratum"))
    if expected_n is not None and len(records) != expected_n:
        raise ScenarioValidationError(
            f"expected {expected_n} records, got {len(records)} "
            "(failed to generate the requested number of unique, non-golden scenarios)"
        )
    if len(records) >= 3 and len(hazard_strata - {None}) < 2:
        raise ScenarioValidationError(f"insufficient hazard-stratum diversity: {hazard_strata}")
    if len(records) >= 3 and len(age_strata - {None}) < 2:
        raise ScenarioValidationError(f"insufficient age-stratum diversity: {age_strata}")


# ---- Stratified sampling helpers --------------------------------------------

def _classify_postcodes() -> dict[str, list[str]]:
    """Classify known-valid postcodes via the authoritative local context tool."""
    tiers: dict[str, list[str]] = {t: [] for t in HAZARD_STRATA}
    for postal_code in KNOWN_VALID_POSTCODES:
        ctx = assess_wg_risk_context({"postal_code": postal_code})
        tier = ctx["hazard_exposure"]["flood_heavy_rain"]
        tiers.setdefault(tier, []).append(postal_code)
    return tiers


def _balanced_column(rng: Random, values: list[Any], n: int) -> list[Any]:
    """A length-n list that cycles through `values` in shuffled blocks, then is
    itself shuffled — guarantees marginal coverage of every value for n >= len(values)."""
    if not values:
        raise ScenarioValidationError("cannot build a balanced column from an empty value list")
    pool: list[Any] = []
    while len(pool) < n:
        block = list(values)
        rng.shuffle(block)
        pool.extend(block)
    pool = pool[:n]
    rng.shuffle(pool)
    return pool


def _weighted_bool_column(rng: Random, n: int, true_ratio: float) -> list[bool]:
    n_true = round(n * true_ratio)
    values: list[bool] = [True] * n_true + [False] * (n - n_true)
    rng.shuffle(values)
    return values


def _weighted_stratum_column(rng: Random, n: int, weights: dict[str, float]) -> list[str]:
    strata = list(weights.keys())
    counts = {s: round(n * w) for s, w in weights.items()}
    counts[strata[0]] += n - sum(counts.values())  # absorb rounding drift in the first bucket
    values = [s for s, c in counts.items() for _ in range(c)]
    rng.shuffle(values)
    return values


def _dwelling_units_for(rng: Random, building_type: str) -> int:
    if building_type == "single-family":
        return 1
    if building_type == "two-family":
        return 2
    if building_type == "multi-family":
        return rng.randint(*MULTI_FAMILY_UNITS_RANGE)
    return 2 if rng.random() < SEMI_TERRACED_TWO_UNIT_PROB else 1


def _living_space_for(rng: Random, building_type: str, dwelling_units: int) -> int:
    if building_type == "multi-family":
        per_unit = rng.randint(*MULTI_FAMILY_UNIT_SQM_RANGE)
        return per_unit * dwelling_units
    lo, hi = LIVING_SPACE_RANGES[building_type]
    if dwelling_units == 2 and building_type in ("semi-detached", "terraced"):
        hi = int(hi * 1.4)
    return rng.randint(lo, hi)


def _rebuild_sum_for(rng: Random, living_space_sqm: int) -> int:
    cost_per_sqm = rng.randint(*COST_PER_SQM_RANGE)
    return round(living_space_sqm * cost_per_sqm, -3)


def _construction_year_for(rng: Random, age_stratum: str) -> int:
    lo, hi = AGE_STRATUM_RANGES[age_stratum]
    return rng.randint(lo, hi)


def _postal_code_for(rng: Random, postcode_tiers: dict[str, list[str]], hazard_stratum: str) -> str:
    postcodes = postcode_tiers.get(hazard_stratum) or []
    if not postcodes:
        raise ScenarioValidationError(f"no known-valid postcodes classified as {hazard_stratum!r}")
    return rng.choice(postcodes)


# ---- Generation --------------------------------------------------------------

def generate_scenarios(
    n: int,
    seed: int = 42,
    golden_hashes: set[str] | None = None,
    max_attempts_factor: int = 20,
) -> list[dict[str, Any]]:
    """Generate `n` unique, non-golden WG risk scenarios with a seeded RNG.

    Diversity across hazard exposure, building age, building type, construction
    class, roof/attic type, ownership and deductible is enforced via stratified
    (not purely random) column sampling, so it doesn't depend on `n` or luck.
    """
    if n <= 0:
        raise ScenarioValidationError(f"n must be positive, got {n}")
    golden_hashes = golden_hashes or set()
    rng = Random(seed)
    postcode_tiers = _classify_postcodes()

    oversample = max(n * 2, n + 20)  # wider than n to absorb (near-impossible) dedup rejects
    hazard_col = _weighted_stratum_column(rng, oversample, {"moderate": 0.4, "elevated": 0.3, "high": 0.3})
    age_col = _balanced_column(rng, list(AGE_STRATA), oversample)
    building_type_col = _balanced_column(rng, list(BUILDING_TYPES), oversample)
    construction_class_col = _balanced_column(rng, list(CONSTRUCTION_CLASSES), oversample)
    roof_col = _balanced_column(rng, list(ROOF_TYPES), oversample)
    attic_col = _balanced_column(rng, list(ATTIC_STATUSES), oversample)
    owner_col = _weighted_bool_column(rng, oversample, true_ratio=0.7)
    deductible_col = _weighted_bool_column(rng, oversample, true_ratio=0.5)

    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    attempts = 0
    max_attempts = n * max_attempts_factor
    idx = 0
    while len(records) < n and attempts < max_attempts and idx < oversample:
        attempts += 1
        hazard_stratum = hazard_col[idx]
        age_stratum = age_col[idx]
        building_type = building_type_col[idx]
        dwelling_units = _dwelling_units_for(rng, building_type)
        living_space_sqm = _living_space_for(rng, building_type, dwelling_units)
        risk = {
            "postal_code": _postal_code_for(rng, postcode_tiers, hazard_stratum),
            "building_type": building_type,
            "construction_class": construction_class_col[idx],
            "construction_year": _construction_year_for(rng, age_stratum),
            "roof_type": roof_col[idx],
            "attic_status": attic_col[idx],
            "living_space_sqm": living_space_sqm,
            "dwelling_units": dwelling_units,
            "owner_occupied": owner_col[idx],
            "rebuild_sum": _rebuild_sum_for(rng, living_space_sqm),
            "with_deductible": deductible_col[idx],
        }
        idx += 1
        sid = scenario_id_for(risk)
        if sid in seen_ids or sid in golden_hashes:
            continue
        seen_ids.add(sid)
        actual_context = assess_wg_risk_context(risk)
        actual_hazard = actual_context["hazard_exposure"]["flood_heavy_rain"]
        records.append(
            {
                "scenario_id": sid,
                "risk": risk,
                "generation_metadata": {
                    "seed": seed,
                    "hazard_stratum": actual_hazard,
                    "age_stratum": age_stratum,
                },
            }
        )

    if len(records) < n:
        raise ScenarioValidationError(
            f"failed to generate {n} unique, non-golden scenarios "
            f"(got {len(records)} after {attempts} attempts, oversample pool {oversample})"
        )
    return records
