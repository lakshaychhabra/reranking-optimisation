#!/usr/bin/env python3
"""Top up the golden quotes to TARGET scenarios using only postcodes that actually
yield tariffs, then re-score EVERY scenario with the current reward + risk-context.
Live Mr-Money calls (3s apart) only for the new scenarios; the existing ones are
just re-scored locally."""
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
BUILDING_TYPES = ["single-family", "two-family", "multi-family", "semi-detached", "terraced"]
CLASSES = ["BAK1", "BAK2", "FHG1", "FHG2", "FHG3"]
ROOFS = ["gable", "flat", "other"]
ATTICS = ["not-converted", "converted", "partially-converted", "none"]


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def rescore(rec: dict) -> dict:
    scored = score_quotes(rec["quotes"], None)
    return {
        "id": rec["id"],
        "risk": rec["risk"],
        "risk_context": assess_wg_risk_context(rec["risk"]),
        "quote_count": len(scored),
        "quotes": scored,
        "reward_recommended_tariff_id": scored[0]["tariff_id"] if scored else None,
    }


def main() -> None:
    existing = load(DATA / "quotes.jsonl")
    working_pcs = sorted({r["risk"]["postal_code"] for r in existing})
    have = len(existing)
    print(f"start: {have} scenarios; {len(working_pcs)} working postcodes")

    rng = random.Random(424242)
    new_recs: list[dict] = []
    attempt = 0
    while have + len(new_recs) < TARGET and attempt < 60:
        attempt += 1
        sqm = rng.choice([70, 90, 110, 130, 150, 175, 200, 230, 260, 300, 340])
        risk = {
            "postal_code": rng.choice(working_pcs),
            "building_type": rng.choice(BUILDING_TYPES),
            "construction_class": rng.choice(CLASSES),
            "construction_year": rng.randint(1905, 2018),
            "roof_type": rng.choice(ROOFS),
            "attic_status": rng.choice(ATTICS),
            "living_space_sqm": sqm,
            "dwelling_units": rng.choice([1, 1, 2, 2, 3]),
            "owner_occupied": rng.random() < 0.7,
            "rebuild_sum": round(sqm * rng.randint(1900, 2700), -3),
            "with_deductible": rng.random() < 0.5,
        }
        rid = f"wg-1{attempt:02d}"
        try:
            quotes = calculate_quotes_wg(risk)
        except Exception as e:  # noqa: BLE001
            print(f"  {rid} ERROR {e}", file=sys.stderr)
            time.sleep(3)
            continue
        print(f"  {rid} {risk['postal_code']} {sqm}m² -> {len(quotes)} quote(s)")
        if len(quotes) >= MIN_QUOTES:
            new_recs.append({"id": rid, "risk": risk, "quotes": quotes})
        time.sleep(3)

    all_recs = existing + new_recs
    final = [rescore(r) for r in all_recs][:TARGET]
    # Re-id sequentially so the golden set is clean wg-000..wg-049.
    for i, r in enumerate(final):
        r["id"] = f"wg-{i:03d}"
    (DATA / "quotes.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in final) + "\n")
    print(f"final: {len(final)} scenarios re-scored -> {DATA / 'quotes.jsonl'}")


if __name__ == "__main__":
    main()
