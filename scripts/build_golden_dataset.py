#!/usr/bin/env python3
"""
build_golden_dataset.py — reproducible pipeline that builds the held-out golden set.

Stages
------
  1. generate  : deterministically sample N varied WG risk scenarios.
  2. price      : call the live Mr-Money Rechenkern for each (3s between calls),
                  compute the value-for-money reward per quote, keep scenarios
                  with >= MIN_QUOTES priceable quotes until TARGET are collected.
  3. annotate   : Fable independently ranks the top-3 recommended products per
                  scenario (blind to the reward) — the "known correct answers".
                  (Run by dispatching Fable model agents; this script writes the
                  quotes file the annotators read and merges their output.)

Outputs (under ../data/):
  scenarios.jsonl                 raw WG risk inputs
  quotes.jsonl                    scenarios + priced+scored quotes (annotator input)
  golden_wg_recommendations.jsonl scenarios + quotes + reward + Fable top-3  (the golden set)

Usage:
  MM_ID=<broker id> MM_PA=AFO python build_golden_dataset.py generate --n 70
  MM_ID=<broker id> MM_PA=AFO python build_golden_dataset.py price --target 50 --delay 3
  python build_golden_dataset.py merge   # after Fable annotations land in annotations.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from calculate_quotes_wg import calculate_quotes_wg, WgRisk  # noqa: E402
from reward import RewardWeights, score_quotes  # noqa: E402
from assess_wg_risk_context import assess_wg_risk_context  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)
MIN_QUOTES = 3

# Real German postcodes spread across hazard regions so the twin sees varied context.
POSTCODES = [
    "01067", "04109", "07743", "10115", "10585", "13353", "20095", "21073", "22765", "26122",
    "27568", "28195", "30159", "33602", "35091", "37073", "40213", "44135", "45127", "48143",
    "50667", "51063", "53113", "55116", "56068", "60311", "63065", "65183", "67059", "68159",
    "70173", "71065", "72074", "74072", "76133", "77652", "79098", "80331", "81667", "83022",
    "84028", "85049", "86150", "88045", "89073", "90402", "93047", "94032", "97070", "99084",
]
BUILDING_TYPES = ["single-family", "two-family", "multi-family", "semi-detached", "terraced"]
CONSTRUCTION_CLASSES = ["BAK1", "BAK2", "FHG1", "FHG2", "FHG3"]
ROOFS = ["gable", "flat", "other"]
ATTICS = ["not-converted", "converted", "partially-converted", "none"]


def generate(n: int, seed: int = 20260802) -> list[dict]:
    rng = random.Random(seed)
    scenarios = []
    for i in range(n):
        sqm = rng.choice([60, 80, 95, 110, 120, 140, 160, 180, 210, 240, 280, 320, 360, 400])
        per_sqm = rng.randint(1900, 2800)
        scenarios.append(
            {
                "id": f"wg-{i:03d}",
                "postal_code": rng.choice(POSTCODES),
                "building_type": rng.choice(BUILDING_TYPES),
                "construction_class": rng.choice(CONSTRUCTION_CLASSES),
                "construction_year": rng.randint(1900, 2020),
                "roof_type": rng.choice(ROOFS),
                "attic_status": rng.choice(ATTICS),
                "living_space_sqm": sqm,
                "dwelling_units": rng.choice([1, 1, 1, 2, 2, 3, 4]),
                "owner_occupied": rng.random() < 0.7,
                "rebuild_sum": round(sqm * per_sqm, -3),
                "with_deductible": rng.random() < 0.5,
            }
        )
    (DATA / "scenarios.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in scenarios) + "\n")
    print(f"generated {len(scenarios)} scenarios -> {DATA / 'scenarios.jsonl'}")
    return scenarios


def price(target: int, delay: float, weights: RewardWeights | None = None) -> None:
    scenarios = [json.loads(l) for l in (DATA / "scenarios.jsonl").read_text().splitlines() if l.strip()]
    kept = []
    with (DATA / "quotes.jsonl").open("w", encoding="utf-8") as fh:
        for s in scenarios:
            if len(kept) >= target:
                break
            risk = {k: v for k, v in s.items() if k != "id"}
            try:
                quotes = calculate_quotes_wg(risk)
            except Exception as e:  # noqa: BLE001 — a single failure must not abort the batch
                print(f"  {s['id']} ERROR: {e}", file=sys.stderr)
                time.sleep(delay)
                continue
            print(f"  {s['id']} {s['postal_code']} {s['living_space_sqm']}m² -> {len(quotes)} quote(s)")
            if len(quotes) >= MIN_QUOTES:
                scored = score_quotes(quotes, weights)
                rec = {
                    "id": s["id"],
                    "risk": risk,
                    "risk_context": assess_wg_risk_context(risk),
                    "quote_count": len(scored),
                    "quotes": scored,
                    "reward_recommended_tariff_id": scored[0]["tariff_id"],
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept.append(rec)
            time.sleep(delay)
    print(f"kept {len(kept)}/{target} scenarios with >= {MIN_QUOTES} quotes -> {DATA / 'quotes.jsonl'}")


def merge() -> None:
    """Merge Fable annotations (data/annotations.jsonl, keyed by id) into the golden set."""
    quotes = {json.loads(l)["id"]: json.loads(l) for l in (DATA / "quotes.jsonl").read_text().splitlines() if l.strip()}
    ann_path = DATA / "annotations.jsonl"
    annotations = {}
    if ann_path.exists():
        annotations = {json.loads(l)["id"]: json.loads(l) for l in ann_path.read_text().splitlines() if l.strip()}
    with (DATA / "golden_wg_recommendations.jsonl").open("w", encoding="utf-8") as fh:
        for rid, rec in quotes.items():
            a = annotations.get(rid, {})
            rec["fable_annotation"] = {
                "top_3": a.get("top_3", []),
                "overall_rationale": a.get("overall_rationale"),
                "annotator_model": a.get("annotator_model", "claude-fable-5"),
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"merged {len(annotations)} annotations -> {DATA / 'golden_wg_recommendations.jsonl'}")


def _main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate"); g.add_argument("--n", type=int, default=70)
    p = sub.add_parser("price"); p.add_argument("--target", type=int, default=50); p.add_argument("--delay", type=float, default=3.0)
    sub.add_parser("merge")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.n)
    elif args.cmd == "price":
        price(args.target, args.delay)
    elif args.cmd == "merge":
        merge()


if __name__ == "__main__":
    _main()
