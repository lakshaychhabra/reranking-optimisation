# Tools — the digital twin

Three tools + the pipeline. All are plain Python 3 (standard library only; no
third-party deps) so they run anywhere. The quote engine needs credentials in the
environment: `MM_ID` (broker id) and `MM_PA` (default `AFO`), provided separately.

```
tools/
  calculate_quotes_wg.py     # price a WG risk via the live Rechenkern
  assess_wg_risk_context.py  # risk → hazard exposure + coverage emphasis
  reward.py                  # value-for-money reward / scorer
scripts/
  build_golden_dataset.py    # scenarios → price → reward → (Fable annotate) → golden set
data/
  scenarios.jsonl            # the 50+ raw WG risks
  quotes.jsonl               # scenarios + priced+scored quotes
  golden_wg_recommendations.jsonl   # + Fable top-3 (the held-out benchmark)
```

## A WG risk

Every tool speaks the same risk shape:

```json
{
  "postal_code": "35091",
  "building_type": "single-family",      // single-family | two-family | multi-family | semi-detached | terraced
  "construction_class": "BAK1",          // BAK1 | BAK2 | FHG1 | FHG2 | FHG3
  "construction_year": 1970,
  "roof_type": "gable",                  // gable | flat | other
  "attic_status": "not-converted",       // not-converted | converted | partially-converted | none
  "living_space_sqm": 160,
  "dwelling_units": 1,
  "owner_occupied": true,
  "rebuild_sum": 400000,                 // Neubausumme, EUR
  "with_deductible": false,
  "cover_fire": true,                    // request fire cover (Feuerschutz; also carries surge)
  "cover_glass": true,                   // request glass cover (Glasversicherung)
  "cover_natural_hazard": false          // request Elementar (flood/quake/snow) — the expensive optional cover
}
```

The three `cover_*` flags are the **Leistungsfragen** (coverage requests). Without
them the engine returns every optional cover as `nein`; the golden set requests
fire + glass on every building and natural-hazard only in flood-exposed postcodes
(see `assess_wg_risk_context`). Requesting a cover also filters the results to
insurers that offer it.

## calculate_quotes_wg.py — price a building

```bash
echo '{"postal_code":"35091","living_space_sqm":160,"rebuild_sum":400000}' \
  | MM_ID=xxxxxxxx MM_PA=AFO python3 tools/calculate_quotes_wg.py
```

Returns `{ risk, quote_count, quotes: [...] }`. Each quote:

```json
{
  "tariff_id": "266",
  "insurer": "Allianz",            // real carrier (ges_kurz); `insurer_pool` keeps the raw ges
  "product": "classic",
  "price_annual": 770.0,           // gross annual premium, EUR (incl. tax)
  "price_net": 661.85,
  "insured_amount": 400000,        // EUR, or "unbegrenzt" (unlimited)
  "deductible": 1000,              // Selbstbeteiligung, EUR
  "term_years": 1,
  "payment_method": "Lastschrift",
  "fee_by_frequency": {"yearly": 770.0, "half_yearly": 396.55, "quarterly": 202.13, "monthly": 67.38},
  "coverages": {"glass": true, "natural_hazard": false},
  "num_coverages": 1,
  "coverage_score": 51            // the engine's own 0–100 quality score
}
```

Programmatic:

```python
from calculate_quotes_wg import calculate_quotes_wg
quotes = calculate_quotes_wg({"postal_code": "35091", "living_space_sqm": 160, "rebuild_sum": 400000})
```

The engine returns HTTP 500 with an empty body for a no-result calc — the tool
handles that and returns `[]` rather than raising. Space calls out (~3s).

## assess_wg_risk_context.py — reason about the building

```bash
echo '{"postal_code":"77652","construction_year":1926,"building_type":"single-family"}' \
  | python3 tools/assess_wg_risk_context.py
```

Returns hazard exposure (`flood_heavy_rain`, `storm_hail`, `surge_overvoltage_relevant`),
building age, construction notes, and a `coverage_emphasis` (0–1 salience per
optional cover). A heuristic — extend it with real data.

## reward.py — score the quotes

```bash
MM_ID=xxxxxxxx MM_PA=AFO python3 tools/calculate_quotes_wg.py --risk risk.json \
  | python3 tools/reward.py --w_cov 0.4 --w_ins 0.3 --w_ded 0.3 --w_price 1.0
```

Returns `{ recommended_tariff_id, scored: [...] }`, quotes sorted best-first with a
`reward` and `reward_components` on each. Programmatic:

```python
from reward import score_quotes, recommend, RewardWeights
scored = score_quotes(quotes, RewardWeights(w_score=0.2))   # weight the engine quality score
best = recommend(quotes)
```

See `../PROBLEM_FRAMING.md` §3 for the formula and its known blind spot.

## build_golden_dataset.py — reproduce the benchmark

```bash
python3 scripts/build_golden_dataset.py generate --n 70
MM_ID=xxxxxxxx MM_PA=AFO python3 scripts/build_golden_dataset.py price --target 50 --delay 3
# Fable annotations land in data/annotations.jsonl (top-3 per scenario), then:
python3 scripts/build_golden_dataset.py merge
```

`generate` is deterministic (fixed seed). `price` keeps scenarios with ≥3
priceable quotes until it reaches `--target`. `merge` folds the Fable annotations
into the final `golden_wg_recommendations.jsonl`.
