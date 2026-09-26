#!/usr/bin/env python3
"""
assess_wg_risk_context.py — the reasoning tool inside the digital twin.

`calculate_quotes_wg` gives the model NUMBERS; this tool gives it CONTEXT. Given a
WG risk it returns a structured risk profile — natural-hazard exposure derived from
the postcode, construction-risk notes from the building attributes, and a
"coverage emphasis" hint (which optional covers matter for THIS building).

Why it exists
-------------
Without context, the optimal policy is a trivial argmax over the reward — no
learning required. With it, a good recommendation must REASON: a flood-exposed
postcode should push `natural_hazard` (Elementar) cover even at a higher premium;
an old timber-frame building shifts weight toward construction/fire considerations.
That reasoning gap is what the fine-tune has to close.

IMPORTANT: this is a transparent HEURISTIC stub, not ground truth. It does not
leak the Fable gold label — it is decision *context*, not the answer. Candidates
are expected to extend/replace it (e.g. real ZÜRS flood zones, GDV storm/hail
data, construction-class tables). Deterministic: same risk → same context.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

# Coarse flood/heavy-rain exposure by 2-digit postcode prefix (illustrative, NOT authoritative).
# High: river-valley / alpine-foreland regions; Elevated: large catchments; else Moderate/Low.
_FLOOD_HIGH_PREFIXES = {"01", "04", "07", "83", "94", "01", "56", "67", "76", "77"}
_FLOOD_ELEVATED_PREFIXES = {"20", "21", "26", "27", "28", "50", "51", "53", "68", "80", "81", "84", "93"}

# Storm/hail belt (south + south-west), coarse.
_HAIL_BELT_PREFIXES = {"70", "71", "72", "73", "74", "75", "76", "77", "78", "88", "89", "80", "81", "82", "83", "84", "85", "86", "87"}


def _prefix(postal_code: str) -> str:
    return (postal_code or "").strip()[:2]


def _flood_tier(postal_code: str) -> str:
    p = _prefix(postal_code)
    if p in _FLOOD_HIGH_PREFIXES:
        return "high"
    if p in _FLOOD_ELEVATED_PREFIXES:
        return "elevated"
    return "moderate"


def _hail_tier(postal_code: str) -> str:
    return "elevated" if _prefix(postal_code) in _HAIL_BELT_PREFIXES else "moderate"


def assess_wg_risk_context(risk: dict[str, Any]) -> dict[str, Any]:
    """Return a structured risk context + coverage-emphasis hint for a WG risk."""
    plz = str(risk.get("postal_code", ""))
    year = int(risk.get("construction_year", 1990) or 1990)
    btype = risk.get("building_type", "single-family")
    bclass = risk.get("construction_class", "BAK1")
    units = int(risk.get("dwelling_units", 1) or 1)
    owner_occupied = bool(risk.get("owner_occupied", True))

    flood = _flood_tier(plz)
    hail = _hail_tier(plz)
    surge_relevant = year <= 2000  # older electrics → surge/overvoltage cover more valuable

    age = 2026 - year
    construction_notes: list[str] = []
    if age >= 60:
        construction_notes.append("old building (>60y): higher water-damage / wiring risk, scrutinise construction class")
    if bclass in ("FHG2", "FHG3"):
        construction_notes.append("timber-frame / lightweight class: elevated fire exposure")
    if units >= 3:
        construction_notes.append("multi-unit: liability & rented-share considerations")
    if not owner_occupied:
        construction_notes.append("rented: loss-of-rent and tenant-liability exposure")

    # Coverage-emphasis hint: which optional add-on covers matter for THIS building (0-1 salience).
    # These are the WG engine's real optional covers: natural-hazard (Elementar) and glass.
    # Surge/Überspannung is not a separate WG cover — the engine folds it into fire — so it is
    # reported here only as an informational note, not an emphasis a policy can act on.
    emphasis = {
        "natural_hazard": {"high": 0.9, "elevated": 0.6, "moderate": 0.25}[flood],
        "glass": 0.5 if btype in ("multi-family", "two-family") else 0.3,
    }

    return {
        "postal_code": plz,
        "hazard_exposure": {
            "flood_heavy_rain": flood,     # high | elevated | moderate
            "storm_hail": hail,            # elevated | moderate
            "surge_overvoltage_relevant": surge_relevant,
        },
        "building_age_years": age,
        "construction_notes": construction_notes,
        "coverage_emphasis": emphasis,     # 0-1 salience per optional cover; higher = matters more here
        "disclaimer": "Heuristic context, not authoritative hazard data. Extend with real ZÜRS/GDV sources.",
    }


def _main() -> None:
    ap = argparse.ArgumentParser(description="Assess natural-hazard / construction risk context for a WG risk.")
    ap.add_argument("--risk", help="Path to a JSON risk file (default: stdin).")
    args = ap.parse_args()
    risk = json.loads(open(args.risk, encoding="utf-8").read() if args.risk else sys.stdin.read())
    print(json.dumps(assess_wg_risk_context(risk), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
