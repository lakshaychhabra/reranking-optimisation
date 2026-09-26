"""Canonical broker prompt shared by evaluation and teacher annotation."""
from __future__ import annotations

import json
from typing import Any


BROKER_SYSTEM = (
    "You are an expert German Wohngebäude (residential building) insurance broker. "
    "Given a building and its available tariffs, recommend the best products by value-for-money "
    "appropriate to the building's actual risk. Consider price, coverage quality (coverage_score), "
    "optional covers vs the building's hazard exposure, deductible, and insured amount adequacy."
)

BROKER_QUOTE_FIELDS = (
    "tariff_id",
    "insurer",
    "product",
    "price_annual",
    "insured_amount",
    "deductible",
    "coverages",
    "num_coverages",
    "coverage_score",
)


def build_broker_prompt(record: dict[str, Any], include_rationales: bool = False) -> str:
    """Build the common benchmark prompt, optionally requesting teacher rationales."""
    lines = [
        "BUILDING:",
        json.dumps(record["risk"], ensure_ascii=False),
        "",
        "RISK CONTEXT (hazard exposure + coverage emphasis):",
        json.dumps(record["risk_context"], ensure_ascii=False),
        "",
        "AVAILABLE TARIFFS:",
    ]
    for quote in record["quotes"]:
        lines.append(
            json.dumps(
                {field: quote.get(field) for field in BROKER_QUOTE_FIELDS},
                ensure_ascii=False,
            )
        )
    if include_rationales:
        output_shape = (
            '{"top_3":[{"rank":1,"tariff_id":"...","rationale":"..."},'
            '{"rank":2,"tariff_id":"...","rationale":"..."},'
            '{"rank":3,"tariff_id":"...","rationale":"..."}],'
            '"overall_rationale":"..."}'
        )
        instruction = (
            "Recommend the TOP 3 products for this building, best first. "
            "Give one concise, factual rationale for each selection and one concise overall rationale. "
            f"Return STRICT JSON only, no prose: {output_shape}"
        )
    else:
        instruction = (
            "Recommend the TOP 3 products for this building, best first. "
            'Return STRICT JSON only, no prose: {"top_3":[{"rank":1,"tariff_id":"..."},'
            '{"rank":2,"tariff_id":"..."},{"rank":3,"tariff_id":"..."}]}'
        )
    lines.extend(["", instruction])
    return "\n".join(lines)
