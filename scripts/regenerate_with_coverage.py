#!/usr/bin/env python3
"""Regenerate the golden quotes WITH requested optional covers.

Coverage policy (risk-context-driven): every building requests fire + glass; a
building in a flood-exposed postcode (risk-context flood tier elevated/high) also
requests natural-hazard (Elementar). This makes the returned coverage flags real
and ties the coverage decision to the building's actual risk.

Re-prices the existing 50 building risks (live, 3s apart), tops up from working
postcodes to keep TARGET with >= MIN_QUOTES priceable quotes, re-scores, and writes
data/quotes.jsonl with a `requested_coverage` field per scenario."""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from calculate_quotes_wg import calculate_quotes_wg  # noqa: E402
from reward import score_quotes  # noqa: E402
from assess_wg_risk_context import assess_wg_risk_context  # noqa: E402

DATA = ROOT / "data"
TARGET = 50
MIN_QUOTES = 3
BASE_RISK_KEYS = ["postal_code", "building_type", "construction_class", "construction_year",
                  "roof_type", "attic_status", "living_space_sqm", "dwelling_units",
                  "owner_occupied", "rebuild_sum", "with_deductible"]
BUILDING_TYPES = ["single-family", "two-family", "multi-family", "semi-detached", "terraced"]
CLASSES = ["BAK1", "BAK2", "FHG1", "FHG2", "FHG3"]
ROOFS = ["gable", "flat", "other"]
ATTICS = ["not-converted", "converted", "partially-converted", "none"]


def coverage_for(risk: dict) -> dict:
    """Fire + glass always; Elementar when the postcode is flood-exposed."""
    ctx = assess_wg_risk_context(risk)
    flood = ctx["hazard_exposure"]["flood_heavy_rain"]
    return {"cover_fire": True, "cover_glass": True, "cover_natural_hazard": flood in ("elevated", "high")}


def price_one(base_risk: dict) -> dict | None:
    risk = {**{k: base_risk[k] for k in BASE_RISK_KEYS if k in base_risk}, **coverage_for(base_risk)}
    try:
        quotes = calculate_quotes_wg(risk)
    except Exception as e:  # noqa: BLE001
        print(f"    ERROR {e}", file=sys.stderr)
        return None
    if len(quotes) < MIN_QUOTES:
        return None
    scored = score_quotes(quotes, None)
    return {
        "risk": {k: risk[k] for k in BASE_RISK_KEYS},
        "requested_coverage": {"fire": risk["cover_fire"], "glass": risk["cover_glass"],
                               "natural_hazard": risk["cover_natural_hazard"]},
        "risk_context": assess_wg_risk_context(risk),
        "quotes": scored,
        "reward_recommended_tariff_id": scored[0]["tariff_id"],
    }


def main() -> None:
    seed_risks = [json.loads(l)["risk"] for l in (DATA / "quotes.jsonl").read_text().splitlines() if l.strip()]
    working_pcs = sorted({r["postal_code"] for r in seed_risks})
    kept: list[dict] = []

    print(f"re-pricing {len(seed_risks)} existing risks with coverage requests...")
    for i, br in enumerate(seed_risks):
        rec = price_one(br)
        cov = coverage_for(br)["cover_natural_hazard"]
        print(f"  {i:02d} {br['postal_code']} {br['living_space_sqm']}m² elem={cov} -> {len(rec['quotes']) if rec else 0} quotes kept={'Y' if rec else 'n'}")
        if rec:
            kept.append(rec)
        time.sleep(3)

    rng = random.Random(20260803)
    attempt = 0
    while len(kept) < TARGET and attempt < 80:
        attempt += 1
        sqm = rng.choice([70, 90, 110, 130, 150, 175, 200, 240, 300])
        br = {
            "postal_code": rng.choice(working_pcs), "building_type": rng.choice(BUILDING_TYPES),
            "construction_class": rng.choice(CLASSES), "construction_year": rng.randint(1905, 2018),
            "roof_type": rng.choice(ROOFS), "attic_status": rng.choice(ATTICS),
            "living_space_sqm": sqm, "dwelling_units": rng.choice([1, 1, 2, 2, 3]),
            "owner_occupied": rng.random() < 0.7, "rebuild_sum": round(sqm * rng.randint(1900, 2700), -3),
            "with_deductible": rng.random() < 0.5,
        }
        rec = price_one(br)
        print(f"  topup#{attempt} {br['postal_code']} {sqm}m² -> kept={'Y' if rec else 'n'} ({len(kept)}/{TARGET})")
        if rec:
            kept.append(rec)
        time.sleep(3)

    final = kept[:TARGET]
    for i, r in enumerate(final):
        r["id"] = f"wg-{i:03d}"
        r["quote_count"] = len(r["quotes"])
        r = {k: r[k] for k in ["id", "risk", "requested_coverage", "risk_context", "quote_count", "quotes", "reward_recommended_tariff_id"]}
        final[i] = r
    (DATA / "quotes.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in final) + "\n")
    ncov = sum(r["requested_coverage"]["natural_hazard"] for r in final)
    print(f"\nfinal: {len(final)} scenarios ({ncov} with Elementar requested) -> {DATA / 'quotes.jsonl'}")


if __name__ == "__main__":
    main()
