#!/usr/bin/env python3
"""
calculate_quotes_wg.py — Wohngebäude (WG / residential building) quote tool.

This is ONE of the tools inside the digital twin. Given a structured WG building
risk, it calls the live Mr-Money "Rechenkern" quote engine and returns a list of
normalised quotes (one per available tariff). It is the same pricing engine Afori
uses in production — the twin is a faithful copy of the real workflow, not a mock.

Usage
-----
    # single risk from a JSON file (or stdin), pretty-print quotes:
    MM_ID=<broker id> MM_PA=AFO python calculate_quotes_wg.py --risk risk.json

    # or programmatically:
    from calculate_quotes_wg import calculate_quotes_wg, WgRisk
    quotes = calculate_quotes_wg(risk_dict, broker_id="...", hersteller_id="AFO")

Credentials are read from env (MM_ID / MM_PA) so they never live in code.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import urllib.parse
import urllib.request

BASE_URL = os.environ.get("MRMONEY_CALCULATOR_BASE_URL", "https://www.mr-money.de/module/kern")

# ── Public → German wire enums (mirror the production WG mapper) ──────────────
BUILDING_TYPE_WIRE = {
    "single-family": "Einfamilienhaus",
    "two-family": "Zweifamilienhaus",
    "multi-family": "Mehrfamilienhaus",
    "semi-detached": "Doppelhaushälfte",
    "terraced": "Reihenhaus",
}
BUILDING_CLASS_WIRE = {"BAK1": "BAK 1", "BAK2": "BAK 2", "FHG1": "FHG 1", "FHG2": "FHG 2", "FHG3": "FHG 3"}
ROOF_WIRE = {"gable": "Giebeldach", "flat": "Flachdach", "other": "Sonstiges"}
ATTIC_WIRE = {
    "not-converted": "nicht ausgebaut",
    "converted": "ausgebaut",
    "partially-converted": "teilweise ausgebaut",
    "none": "kein Dachgeschoss",
}


@dataclass
class WgRisk:
    """A residential-building risk. Fields map 1:1 onto the engine's wire params."""

    postal_code: str
    building_type: str = "single-family"         # -> gebaeude
    construction_class: str = "BAK1"             # -> bauart
    construction_year: int = 1990                # -> baujahr
    roof_type: str = "gable"                     # -> dachart
    attic_status: str = "not-converted"          # -> dachgeschoss
    living_space_sqm: int = 140                  # -> wohnfl_og
    dwelling_units: int = 1                      # -> wohneh
    owner_occupied: bool = True                  # -> Selbstgenutzt
    rebuild_sum: int = 350_000                   # -> wert (grund=Neubausumme)
    with_deductible: bool = False                # -> sb
    # ── Requested optional covers (the "Leistungsfragen"). Without these the engine
    #    returns every optional cover as "nein". Fire (Feuerschutz, which also carries
    #    surge/Überspannung) and glass are cheap/standard; natural-hazard (Elementar)
    #    is the expensive flood/quake cover, worth requesting mainly in exposed areas.
    cover_fire: bool = True                       # -> Feuerschutz=ja
    cover_glass: bool = True                      # -> glas=ja
    cover_natural_hazard: bool = False            # -> elementar=ja (+ schaeden_anz=0)

    def to_wire_params(self) -> dict[str, str]:
        p: dict[str, str] = {
            "plz": self.postal_code,
            "gebaeude": BUILDING_TYPE_WIRE[self.building_type],
            "bauart": BUILDING_CLASS_WIRE[self.construction_class],
            "baujahr": str(self.construction_year),
            "dachart": ROOF_WIRE[self.roof_type],
            "dachgeschoss": ATTIC_WIRE[self.attic_status],
            "wohnfl_og": str(self.living_space_sqm),
            "wohneh": str(self.dwelling_units),
            "Selbstgenutzt": "ja" if self.owner_occupied else "nein",
            "grund": "Neubausumme",
            "wert": str(self.rebuild_sum),
            "sb": "ja" if self.with_deductible else "nein",
            "versbeginn": "sofort",
        }
        if self.cover_fire:
            p["Feuerschutz"] = "ja"
        if self.cover_glass:
            p["glas"] = "ja"
        if self.cover_natural_hazard:
            p["elementar"] = "ja"
            p["schaeden_anz"] = "0"  # no prior claims — required companion when Elementar is requested
        return p


# ── German number / value parsing ────────────────────────────────────────────
def _german_number(s: str | None) -> float | None:
    if s is None:
        return None
    s = str(s).strip()
    if s == "":
        return None
    # "1.037,89" -> 1037.89 ; "469.213902" (dot decimal) -> 469.21 ; "531,15" -> 531.15
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return round(float(s), 2)
    except ValueError:
        return None


def _insured_sum(s: str | None) -> int | str | None:
    """Insured sum: integer EUR, or the literal 'unbegrenzt' (unlimited), or None."""
    if s is None or str(s).strip() == "":
        return None
    raw = str(s).strip()
    if raw.lower() == "unbegrenzt":
        return "unbegrenzt"
    n = _german_number(raw.replace(".", ""))  # thousands-dots only for a sum
    return int(n) if n is not None else None


def _arr_brutto(s: str | None) -> dict[str, float]:
    """'531.15;273.54;139.43;0' -> {yearly, half_yearly, quarterly, monthly} (drop 0s)."""
    out: dict[str, float] = {}
    if not s:
        return out
    keys = ["yearly", "half_yearly", "quarterly", "monthly"]
    for k, part in zip(keys, str(s).split(";")):
        try:
            v = round(float(part), 2)
            if v > 0:
                out[k] = v
        except ValueError:
            pass
    return out


def _yes(v: str | None) -> bool:
    return str(v or "").strip().lower() == "ja"


# ── Engine call + parse ──────────────────────────────────────────────────────
def _fetch_raw(params: dict[str, str], broker_id: str, hersteller_id: str, timeout: int = 30) -> dict[str, Any]:
    query = {
        **params,
        "id": broker_id,
        "pa": hersteller_id,
        "IP_USER": f"10.0.0.1-{random.randint(10000, 99999)}",
        "out": "json",
    }
    url = f"{BASE_URL}/wg.php?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urllib.error.HTTPError as e:
        # The engine returns HTTP 500 with a valid {"daten":[]} body for a no-result calc.
        body = e.read().decode("utf-8", errors="ignore")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            raise RuntimeError(f"Rechenkern HTTP {e.code}: {body[:300]}") from e


# Optional add-on coverages the WG engine reports as ja/nein flags. (Überspannung/surge
# has no separate WG flag — the engine folds it into fire cover — so it is not tracked.)
COVERAGE_FLAGS = {"glass": "Glas", "natural_hazard": "Elementar"}


def _normalise_tariff(t: dict[str, Any]) -> dict[str, Any]:
    coverages = {name: _yes(t.get(key)) for name, key in COVERAGE_FLAGS.items()}
    return {
        "tariff_id": str(t.get("tarifnrtemp") or t.get("tarifnr") or ""),
        "insurer": t.get("ges_kurz") or t.get("ges"),   # ges_kurz is the real carrier; ges is often a pool
        "insurer_pool": t.get("ges"),
        "product": t.get("tar_kurz") or t.get("tar"),
        "price_annual": _german_number(t.get("beitrag")),          # gross annual premium, EUR (incl. tax)
        "price_net": _german_number(t.get("beitrag_netto")),
        "insured_amount": _insured_sum(t.get("Versicherungssumme")),
        "deductible": int(_german_number(t.get("selbst")) or 0),   # Selbstbeteiligung, EUR
        "term_years": int(t["laufzeit"]) if str(t.get("laufzeit", "")).isdigit() else None,
        "payment_method": t.get("zahlungsart"),
        "fee_by_frequency": _arr_brutto(t.get("arr_brutto")),
        "coverages": coverages,
        "num_coverages": sum(coverages.values()),                  # optional add-on covers included
        "coverage_score": int(_german_number(t.get("tar_punkte")) or 0),  # engine quality score 0-100
    }


def calculate_quotes_wg(
    risk: dict[str, Any] | WgRisk,
    broker_id: str | None = None,
    hersteller_id: str | None = None,
) -> list[dict[str, Any]]:
    """Price a WG risk and return normalised quotes (one per tariff). Never raises on a no-result calc."""
    broker_id = broker_id or os.environ.get("MM_ID")
    hersteller_id = hersteller_id or os.environ.get("MM_PA", "AFO")
    if not broker_id:
        raise RuntimeError("Missing broker id (set MM_ID or pass broker_id).")

    wg = risk if isinstance(risk, WgRisk) else WgRisk(**risk)
    raw = _fetch_raw(wg.to_wire_params(), broker_id, hersteller_id)
    tariffs = [x for x in (raw.get("daten") or []) if isinstance(x, dict) and (x.get("tarifnrtemp") or x.get("tarifnr"))]
    quotes = [_normalise_tariff(t) for t in tariffs]
    # Keep only priceable quotes (a real gross premium).
    return [q for q in quotes if isinstance(q.get("price_annual"), (int, float)) and q["price_annual"] > 0]


def _main() -> None:
    ap = argparse.ArgumentParser(description="Price a WG (Wohngebäude) risk via the Mr-Money Rechenkern.")
    ap.add_argument("--risk", help="Path to a JSON risk file (default: stdin).")
    args = ap.parse_args()
    text = open(args.risk, encoding="utf-8").read() if args.risk else sys.stdin.read()
    risk = json.loads(text)
    quotes = calculate_quotes_wg(risk)
    print(json.dumps({"risk": risk, "quote_count": len(quotes), "quotes": quotes}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _main()
