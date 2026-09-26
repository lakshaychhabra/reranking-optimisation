#!/usr/bin/env python3
"""
run_eval.py — baseline benchmark of candidate models on the golden WG set.

For each golden scenario, a model is shown the building (risk + risk_context) and
its available tariffs (NO reward, NO Fable answer) and must recommend the top-3
products by tariff_id. We score each model on:

  top1_accuracy   : fraction where the model's #1 == Fable's #1
  top3_recall     : mean |model_top3 ∩ Fable_top3| / 3
  reward_ratio    : mean( reward(model #1) / max reward in scenario )   [value-for-money]
  tokens          : mean prompt+completion tokens per recommendation
  cost_per_1k     : estimated cost to produce 1,000 recommendations, reported
                    in the currency used by that provider's price table

Providers (creds read from this repository's .env or the process environment):
  openai — OpenAI Responses API
  hf     — HF serverless router (open-source models)
  google — Gemini generative-language API
  azure  — Azure OpenAI (v1 responses API)

Usage:
  python scripts/run_eval.py                         # runs GPT-6 Sol on all 50 scenarios
  python scripts/run_eval.py --models gptsol --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prompts import BROKER_SYSTEM, build_broker_prompt  # noqa: E402

DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
ENV_PATH = ROOT / ".env"


def load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
            if m:
                v = m.group(2).strip()
                if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                    v = v[1:-1]
                env[m.group(1)] = v
    # Explicitly exported variables take precedence over values in .env.
    env.update(os.environ)
    return env


ENV = load_env(ENV_PATH)

# ── Price table: provider currency per 1,000,000 tokens. ADJUST to your contract. ──
# Best-effort public estimates for Aug-2026 models; the cost figure is only as good
# as these numbers, so they are surfaced in the results for transparency.
PRICES_PER_MTOK = {
    "qwen3.5-9b":       {"in": 0.18, "out": 0.18, "note": "HF serverless est.; open-source self-host is typically far cheaper at scale"},
    "gemini-3-flash":   {"in": 0.30, "out": 2.50, "note": "Gemini 3 Flash est. (preview)"},
    "gpt-5.4":          {"in": 1.10, "out": 8.80, "note": "Azure GPT-5.4 est. (sub for unprovisioned 5.5)"},
    # OpenAI publishes API prices in USD. The cost field is emitted as USD for
    # this entry rather than pretending that the same number is an EUR price.
    "gpt-6-sol":        {"in": 2.00, "out": 10.00, "currency": "USD", "note": "OpenAI Standard processing"},
}

# ── Model specs ──────────────────────────────────────────────────────────────
MODELS = {
    "gptsol": {
        "provider": "openai", "model": "gpt-6-sol", "price_key": "gpt-6-sol",
        "api_key": ENV.get("OPENAI_KEY") or ENV.get("OPENAI_API_KEY"),
        "label": "GPT-6 Sol (frontier, OpenAI Responses API)",
    },
    "qwen": {
        # HF serverless routes through an inference provider; the base id 403s, so pin a live provider.
        "provider": "hf", "model": "Qwen/Qwen3.5-9B:together", "price_key": "qwen3.5-9b",
        "label": "Qwen3.5-9B (open-source, HF serverless via Together)",
    },
    "gemini": {
        "provider": "google", "model": "gemini-3-flash-preview", "price_key": "gemini-3-flash",
        "label": "Gemini 3 Flash (frontier, Google)",
    },
    "gpt54": {
        "provider": "azure", "model": ENV.get("AZURE_OPENAI_API_GPT_5_4_DEPLOYMENT_NAME", "gpt-5.4"),
        "instance": ENV.get("AZURE_OPENAI_API_GPT_5_4_INSTANCE_NAME"),
        "api_key": ENV.get("AZURE_OPENAI_API_GPT_5_4_API_KEY"), "price_key": "gpt-5.4",
        "label": "gpt-5.4 (frontier, Azure; sub for unprovisioned GPT-5.5)",
    },
}

SYSTEM = BROKER_SYSTEM


def build_prompt(rec: dict) -> str:
    return build_broker_prompt(rec, include_rationales=False)


# ── Provider callers → (text, prompt_tokens, completion_tokens) ──────────────
# A browser-like UA: HF's router sits behind Cloudflare, which 403s the default
# Python-urllib agent with an HTML challenge page.
_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"


def _post(url: str, headers: dict, payload: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "User-Agent": _UA, "Accept": "application/json", **headers},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="ignore"))


def call_hf(spec, prompt):
    d = _post(
        "https://router.huggingface.co/v1/chat/completions",
        {"Authorization": f"Bearer {ENV['HUGGINGFACE_API_KEY']}"},
        {"model": spec["model"], "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
         "max_tokens": 700, "temperature": 0.0,
         # Qwen3.5 is a reasoning model; without this it burns the whole budget "thinking"
         # and returns empty content. Disable thinking for a direct JSON answer.
         "chat_template_kwargs": {"enable_thinking": False}},
    )
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_google(spec, prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{spec['model']}:generateContent?key={ENV['GOOGLE_API_KEY']}"
    d = _post(url, {}, {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        # JSON mode + generous budget: gemini-3-flash "thinks", so it needs room before the JSON.
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 4000, "responseMimeType": "application/json"},
    })
    cand = (d.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    u = d.get("usageMetadata", {})
    return text, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def call_azure(spec, prompt):
    url = f"https://{spec['instance']}.openai.azure.com/openai/v1/chat/completions"
    d = _post(url, {"api-key": spec["api_key"]}, {
        "model": spec["model"],
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        "max_completion_tokens": 3000,  # reasoning model — leave room
    })
    u = d.get("usage", {})
    return d["choices"][0]["message"]["content"], u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def call_openai(spec, prompt):
    if not spec.get("api_key"):
        raise RuntimeError("Missing OPENAI_KEY (or OPENAI_API_KEY) in the repository .env")
    d = _post(
        "https://api.openai.com/v1/responses",
        {"Authorization": f"Bearer {spec['api_key']}"},
        {
            "model": spec["model"],
            "instructions": SYSTEM,
            "input": prompt,
            "reasoning": {"effort": "low"},
            "max_output_tokens": 3000,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "wg_top_3_recommendations",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "top_3": {
                                "type": "array",
                                "minItems": 3,
                                "maxItems": 3,
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "rank": {"type": "integer", "minimum": 1, "maximum": 3},
                                        "tariff_id": {"type": "string"},
                                    },
                                    "required": ["rank", "tariff_id"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                        "required": ["top_3"],
                        "additionalProperties": False,
                    },
                }
            },
        },
    )
    text = "".join(
        part.get("text", "")
        for item in d.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
    u = d.get("usage", {})
    return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


CALLERS = {"openai": call_openai, "hf": call_hf, "google": call_google, "azure": call_azure}


def parse_top3(text: str) -> list[str]:
    """Extract ranked tariff_ids from the model's JSON (lenient)."""
    if not text:
        return []
    m = re.search(r"\{.*\}", text, re.S)
    blob = m.group(0) if m else text
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        ids = re.findall(r'"tariff_id"\s*:\s*"?([\w:-]+)"?', text)
        return ids[:3]
    top = obj.get("top_3") or obj.get("recommendations") or []
    out = []
    for item in top:
        tid = item.get("tariff_id") if isinstance(item, dict) else item
        if tid is not None:
            out.append(str(tid))
    return out[:3]


def eval_model(key: str, golden: list[dict], delay: float) -> dict:
    spec = MODELS[key]
    caller = CALLERS[spec["provider"]]
    per_scenario = []
    for rec in golden:
        reward_by_id = {q["tariff_id"]: q["reward"] for q in rec["quotes"]}
        argmax_reward = max(reward_by_id.values())
        fable = rec.get("fable_annotation", {}).get("top_3", [])
        fable_ids = [str(t["tariff_id"]) for t in fable]
        prompt = build_prompt(rec)
        text, pt, ct, err = "", 0, 0, None
        for attempt in range(2):
            try:
                text, pt, ct = caller(spec, prompt)
                break
            except (urllib.error.URLError, KeyError, TimeoutError) as e:
                err = str(e)[:160]
                time.sleep(2)
        top3 = parse_top3(text)
        pick = top3[0] if top3 else None
        pick_reward = reward_by_id.get(pick)
        per_scenario.append({
            "id": rec["id"],
            "model_top3": top3,
            "model_pick": pick,
            "fable_top3": fable_ids,
            "top1_hit": bool(pick and fable_ids and pick == fable_ids[0]),
            "top3_recall": (len(set(top3) & set(fable_ids)) / 3.0) if fable_ids else None,
            "pick_reward": pick_reward,
            "argmax_reward": argmax_reward,
            "reward_ratio": (pick_reward / argmax_reward) if (pick_reward and argmax_reward) else None,
            "prompt_tokens": pt, "completion_tokens": ct,
            "error": err if not top3 else None,
        })
        time.sleep(delay)

    scored = [s for s in per_scenario if s["model_pick"] is not None]
    n = len(scored) or 1
    tot_in = sum(s["prompt_tokens"] for s in per_scenario)
    tot_out = sum(s["completion_tokens"] for s in per_scenario)
    price = PRICES_PER_MTOK[spec["price_key"]]
    total_cost = (tot_in / 1e6) * price["in"] + (tot_out / 1e6) * price["out"]
    cost_per_1k = total_cost / (len(per_scenario) or 1) * 1000
    currency = price.get("currency", "EUR")
    cost_field = f"cost_per_1k_recommendations_{currency.lower()}"

    def mean(field):
        vals = [s[field] for s in scored if s.get(field) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {
        "model": key, "label": spec["label"], "model_id": spec["model"],
        "scenarios": len(per_scenario), "answered": len(scored),
        "top1_accuracy": round(sum(s["top1_hit"] for s in scored) / n, 4),
        "top3_recall": mean("top3_recall"),
        "reward_ratio": mean("reward_ratio"),
        "mean_prompt_tokens": round(tot_in / len(per_scenario)),
        "mean_completion_tokens": round(tot_out / len(per_scenario)),
        cost_field: round(cost_per_1k, 4),
        "cost_currency": currency,
        "price_assumption_per_mtok": price,
    }
    (RESULTS / f"{key}.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in per_scenario) + "\n")
    print(f"  {key}: answered {len(scored)}/{len(per_scenario)} | top1 {summary['top1_accuracy']} "
          f"| top3_recall {summary['top3_recall']} | reward_ratio {summary['reward_ratio']} "
          f"| {currency} {summary[cost_field]}/1k")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=sorted(MODELS), default=["gptsol"])
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--delay", type=float, default=0.5)
    args = ap.parse_args()
    golden = [json.loads(l) for l in (DATA / "golden_wg_recommendations.jsonl").read_text().splitlines() if l.strip()]
    golden = golden[: args.limit]
    print(f"eval on {len(golden)} golden scenarios: {args.models}")
    summaries = [eval_model(k, golden, args.delay) for k in args.models]
    reward_argmax = {
        "model": "reward-argmax", "label": "Baseline: pick argmax(reward) — the twin scorer itself",
        "top1_accuracy": None, "reward_ratio": 1.0, "cost_per_1k_recommendations_eur": 0.0,
        "note": "Not a model; the reward's own best pick. reward_ratio is 1.0 by definition; top1 vs Fable measured below.",
    }
    # reward-argmax agreement with Fable (free reference)
    ra_hits = ru = 0
    for rec in golden:
        f = [str(t["tariff_id"]) for t in rec.get("fable_annotation", {}).get("top_3", [])]
        if f:
            ru += 1
            ra_hits += int(rec["reward_recommended_tariff_id"] == f[0])
    reward_argmax["top1_accuracy_vs_fable"] = round(ra_hits / ru, 4) if ru else None
    out = {"eval_scenarios": len(golden), "models": summaries, "reference": reward_argmax}
    (RESULTS / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"wrote {RESULTS / 'summary.json'}")


if __name__ == "__main__":
    main()
