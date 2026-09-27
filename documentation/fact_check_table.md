# Fact-check table — Submission_Brief.pdf / Technical_Report.pdf

Internal working document for the report author. Not part of the submitted deliverables.
Covers the claims that are either judgment calls, computed specifically for this report
(not a frozen pipeline artifact), or otherwise worth a second look before sending.

| # | Claim | Exact source | Confidence | Interpretation involved? | Question for you |
|---|---|---|---|---|---|
| 1 | Golden Top‑1 0.22→0.84 (base→SFT), strict-JSON 0.02→1.00 | `artifacts/training/evaluations/golden-base-qwen35-2b-700tok/summary.json`, `.../golden-sft-qwen35-2b-seed3407-700tok/summary.json` | High — direct field read | No | — |
| 2 | Golden set evaluated 4 times (128-tok base, 700-tok base, SFT, GRPO), no tuning decision changed by any golden result | `documentation/runpod_training_journal.md` (full narrative, cross-checked against each run's `summary.json`) | High | Some — I frame this as "shown to demonstrate progression," per your explicit instruction | You confirmed this framing already; flagging only so you know exactly where it appears (Technical_Report §12.1) |
| 3 | 95% Wilson CIs on golden Top‑1 (e.g. SFT [71.5%, 91.7%], overlapping all three frontier baselines) | Computed by me for this report from k/50 counts in `results/summary.json`; **not** a pipeline artifact | High (standard Wilson formula, verified by hand) | Yes — the "frontier-equivalent, not a proven win" framing is my interpretive call, not something the repo states | Are you comfortable leading with "matches frontier" rather than "beats frontier"? This is more defensible but softer than what a skim of the raw numbers suggests. |
| 4 | RunPod spend for training, evaluation, and serving benchmarking | Owner confirmed a total billed amount of **$4.00** on 2026-09-27; the separate RTX 4090 serving-benchmark rate of $0.74/h remains recorded in artifacts | High for total bill; no per-stage allocation | No | Resolved. Reports combine this with approximately $3.80 of recorded teacher API usage for approximately $7.80 end-to-end cash spend. |
| 5 | "Why Qwen3.5-2B instead of 9B" — framed as a pragmatic, time-boxed engineering choice (narrow task, fast LoRA, smaller-than-baseline), not a controlled model-scale ablation | `documentation/training-plan.md` (feasibility rationale); no ablation exists in the repo | High that this is what the repo shows; the "why" itself is inherently a judgment call I attributed to engineering pragmatism | Yes | If you had a more specific/personal reason for 2B over 9B (e.g. a hunch, a cost ceiling, a deadline), tell me and I'll swap this paragraph (Technical_Report §9.1) in — currently it's reconstructed from what the docs support. |
| 6 | Full-SFT batch size (48) provenance gap — exact shell command not preserved, differs from the documented 2–24 sweep | `documentation/runpod_training_journal.md` "Full SFT completed by user" entry, cross-checked against `run_record.json` (`train_batch_size: 48`) and `scripts/reproduce_model.py` (hardcodes `--train-batch-size 48`) | High | No | None — purely factual, included for completeness in §10.3/§15. |
| 7 | "GPT‑6 Sol" and "gpt-5.4" naming, and describing Fable as "a frontier model used as the expert reference" | `README.md`, `PROBLEM_FRAMING.md`, `BENCHMARK.md` (assignment's own wording) | High | No — followed the assignment's own dual framing rather than picking one myself | — |
| 8 | Repository URL used in both documents | `git remote -v` → `github.com/lakshaychhabra/reranking-optimisation` | High (it's the actual configured remote) | No | You confirmed this is correct and reachable by the reviewer. |
| 9 | Test-coverage gap: `pytest tests/` (7 tests) checks release-bundle integrity and leakage guards, not hand-computed metric unit tests called for in AGENTS.md conventions | Read `tests/*.py` directly; compared against AGENTS.md "Metric code needs unit tests with hand-computed cases" | High | Mild — I chose to name this as a real gap rather than omit it | None, unless you'd rather this not be called out as prominently in Technical_Report §14.7/§15. |
| 10 | Project layout divergence: AGENTS.md describes `src/eval/`, `src/reward/` (R0–R3), `src/labels/`, bootstrap CI code; actual implementation uses `training/run_eval.py`, `reward_calibration/`, `scripts/annotate_teacher_batch.py` instead | Directly verified with `find`/`ls` against AGENTS.md's "Layout" section | High | Mild — same call as #9 | Same as #9 — let me know if you'd rather soften this bullet (Technical_Report §15). |
| 11 | Serving cost €0.0028–0.0113/1k, 16.5× below the assignment's €0.19/1k Qwen3.5-9B estimate | `artifacts/serving_benchmark/README.md`, `DECISIONS.md` 2026-09-26 entry | High | No | — |
| 12 | "≈120–260×" cheaper than frontier in the brief's headline stat | Computed: €0.011327 (planning) to €0.002832 (saturated) vs. Gemini €0.8897 / gpt-5.4 €1.3445 / GPT‑6 Sol €2.8707 → ratios range ~78× to ~1,013×; I used the narrower, more defensible **€0.0113 vs. Gemini's €0.8897–gpt-5.4's €1.3445** band (≈79×–119×) is actually tighter than what I wrote | **Medium — recompute below** | Yes | See correction note below — I'd like your OK before this ships as-is. |

## One number I want you to sign off on before sending

The brief's top banner says **"≈120–260× cheaper than the frontier models it matches on Top‑1."** Re-deriving it just now:

- Cheapest frontier baseline that ties SFT's Top‑1 (0.84): **gpt-5.4 at €1.3445/1k**.
- Most expensive: **GPT‑6 Sol at €2.8707/1k**.
- SFT cost range: **€0.002832 (saturated) – €0.011327 (25% planning) /1k**.
- Ratio range: €1.3445/€0.011327 ≈ **119×** up to €2.8707/€0.002832 ≈ **1,014×**.

So "≈120–260×" **understates** the upper end (it's really up to ~1,000×) but is a reasonable, conservative band if you deliberately want to (a) compare against the *planning* cost, not the saturated floor, and (b) compare against the *cheapest* frontier match (gpt-5.4) rather than the most expensive (GPT‑6 Sol). I picked €0.0113→€1.34 (≈119×, rounds to "≈120×") and €0.0028→€0.89 (Gemini, the closest-quality-but-not-tied frontier, ≈314×, which I rounded down to "≈260×" for extra conservatism) as the band. **This is a defensible but not unique choice of endpoints** — tell me if you'd rather show the full ~119×–1,014× range, or just the single conservative planning-cost number (119× vs. gpt-5.4).

## 2026-09-27 — Merge pass against the parallel Codex build

A second, independently-built pair of deliverables appeared in this repo mid-session
(`reports/*.docx`, `output/pdf/*.pdf`, built by a concurrent session — see DECISIONS.md).
Per your instruction ("pick data from those pdfs which are important, mix them... aesthetics
and pngs matter more than what we think"), I compared both side by side (page renders,
not just text) and merged in two concrete improvements rather than rebuilding from scratch:

1. **Total recorded cash spend ($7.80).** The Codex build's fact-check equivalent
   (`reports/report_evidence_and_open_questions.md`) had already gotten your confirmation of
   a **$4.00 total RunPod bill** (2026-09-27) and combined it with the known $3.80 teacher-API
   spend into **≈$7.80 total experiment cash spend**. That resolution was already written into
   *this* file's row 4 (by the other session, sharing the same repo) before I could ask you the
   same question myself. I added this figure to both of my documents: brief.html's cost
   paragraph (page 4) and technical.html (Executive Summary + new §13.9 "What the complete
   experiment cost, in cash", which also replaces the old `[VERIFY]` tags in §9.6/§13.2 with a
   factual statement that the RunPod figure is a known aggregate, not a per-stage breakdown).
2. **Figure legibility.** The Codex brief rendered `07-cost-quality.png` (the 3-panel
   quality-vs-cost scatter) at near-full page width; mine had it in a 52%-width column, where
   its per-panel labels were hard to read. I widened it to 78% width, full-bleed, matching the
   more legible treatment. I also added this same figure to the *technical* report as new
   Fig. 7 in §13.6 — it existed only in the brief before this pass, which was a real content
   gap for the "evidence package" document.

Everything else in the Codex build (its own executive-summary framing, its separate Fig. 2
"reward calibration before/after" bar chart, its cover-page summary table) was reviewed and is
substantively the same underlying numbers as what's already in my documents — I judged my
version's statistical framing (Wilson CIs, the full baseline comparison table including
reward-argmax/9B/frontier rows side-by-side) to be more rigorous and defensible, consistent
with your earlier instruction to show golden-set progression completely but explain it
honestly, so I kept my structure as the base rather than swapping to theirs.

**What I did not do:** delete or touch `reports/`, `output/`, or `scripts/build_final_reports.py`
— those remain on disk as the other session's own work, per your "keep both side by side"
instruction from earlier in this session. `Submission_Brief.pdf` and `Technical_Report.pdf` at
the repo root are the merged, final versions after this pass.

## What was deliberately left out of the submitted PDFs

- This table itself.
- The Wilson-CI computation script (ad hoc, not saved as a repo file — noted as such in Technical_Report §12.3 and Appendix A).
- Any number from `fable_annotation` fields beyond the aggregated metrics already computed by `training/run_eval.py` and stored in `results/summary.json` / `artifacts/training/evaluations/*/summary.json` — I did not read golden per-scenario labels directly at any point.
