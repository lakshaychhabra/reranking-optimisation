# Decisions log

Format per entry: options considered / choice / why / what evidence would change it.

## 2026-09-26 — Preserve CUDA evaluation and add a portable Transformers backend

- **Options considered:** keep evaluation CUDA/Unsloth-only, replace it with a
  generic backend, or add an explicit portable backend while preserving the
  original path.
- **Choice:** `training/run_eval.py --backend auto` selects Unsloth when CUDA
  and Unsloth are available, otherwise Transformers; either backend can be
  requested explicitly.
- **Why:** this leaves the reported RTX A6000 execution path unchanged while
  allowing a fresh Apple-Silicon clone to load the same PEFT adapter. The Mac
  MPS run matched all 50 frozen SFT outputs exactly on every substantive field.
- **What evidence would change it:** a backend-specific prediction mismatch
  would make CUDA the sole reproducibility claim and demote portable inference
  to a format/load smoke test.

## 2026-09-26 — Track a minimal Git release; keep full experiment state local

- **Options considered:** commit all of `artifacts/`, require a separate hosted
  artifact archive, or build a small allowlisted release inside the Git repo.
- **Choice:** track `release/` with the two portable adapters, prepared
  non-golden data, development evaluation inputs, exact configs, aggregate
  summaries, and hashes; keep caches, raw provider data, golden rows, base
  weights, and optimizer checkpoints outside Git.
- **Why:** a fresh clone can evaluate the submitted model and retrain SFT/GRPO
  without a second private download, while Git LFS carries only 80 MB of model
  files and the held-out/data-security boundary remains explicit.
- **What evidence would change it:** a submission size policy that prohibits
  Git LFS would move `release/model/` to a private Hugging Face model repo and
  add an authenticated fetch step; a fully offline requirement would instead
  require separately packaging the verified base weights.

## 2026-09-26 — Deliver SFT; retain GRPO as an optional ablation

- **Options considered:** deliver the completed SFT adapter, deliver the
  ten-step GRPO continuation, or continue GRPO for more steps.
- **Choice:** select SFT as the primary model and preserve GRPO separately as
  an optional reproducible stage.
- **Why:** both adapters achieved 42/50 golden Top-1 and `0.811724` supplied
  reward ratio, while GRPO reduced Top-3 recall from `0.866667` to `0.860000`.
  Continuing after inspecting golden would also violate the held-out boundary.
- **What evidence would change it:** a predeclared GRPO run selected only on
  the development split that materially improves ranking quality without
  format or reward regression, followed by a fresh untouched final benchmark.

## 2026-09-26 — Reproduce from pinned Hub weights instead of storing base weights

- **Options considered:** keep the 4.3 GB base-model copy from the GPU archive,
  or download the public model at an immutable Hugging Face revision and store
  only LoRA checkpoints locally.
- **Choice:** pin revision `15852e8c16360a2fea060d615a32b45270f8a8fc`, verify
  the published weight checksum, and retain SFT/GRPO checkpoints separately.
- **Why:** this keeps the handoff compact while preserving the exact base model,
  portable adapters, and optimizer state needed to reproduce or resume runs.
- **What evidence would change it:** loss of Hub availability or a delivery
  requirement for a fully offline package would justify bundling the verified
  base-model snapshot separately.

## 2026-09-25 — Post-pilot hardening before dataset scale-up

- **Options considered:** retain synthesized postcode suffixes and the pilot's
  internal cache schema unchanged, or harden the pipeline before collecting a
  large dataset.
- **Choice:** sample only known-valid postcodes from the supplied benchmark
  generator; expose `risk` consistently in downstream episode files; validate
  complete input risks; protect the cache path; make collection order seeded;
  prevent concurrent collectors; and permit transient-error retries only via an
  explicit flag.
- **Why:** the 20-call pilot proved the basic loop works, so the next risk is
  silently scaling data-quality or operational defects. These changes preserve
  the pilot cache while making future collection safer and reproducible.
- **What evidence would change it:** an authoritative postcode dataset could
  replace the supplied curated list; a coordinated distributed collection
  design could replace the single-process lock; and observed engine
  nondeterminism could justify additional explicit retry policies.

## 2026-09-25 — collect_quotes.py --max-live-calls: no code-enforced ceiling

- **Options considered:** (a) hard-cap `--max-live-calls` in code at a fixed
  number (originally 20, matching CLAUDE.md rule 2's "without explicit
  approval" threshold); (b) leave it as a plain user-supplied parameter with
  no ceiling, trusting the operator's explicit approval each run.
- **Choice:** (b), per explicit user instruction — removed `HARD_MAX_LIVE_CALLS`
  entirely; `--max-live-calls` now only requires `> 0`.
- **Why:** a code-level hard cap on a number the user can already choose (and
  approve) each invocation was redundant with — and a worse enforcement point
  than — the human-in-the-loop approval CLAUDE.md rule 2 actually asks for.
  The 3s inter-call throttle (the part that protects the live engine itself)
  stays enforced in two independent places and was left untouched.
- **What evidence would change it:** if live runs start happening without a
  human explicitly choosing `--max-live-calls` each time (e.g. from an
  automated/scheduled job), re-introduce a code-level ceiling — the removed
  safeguard was specifically substituting for manual judgment, which no
  longer applies in an unattended context.

## 2026-09-25 — quoted_scenarios.jsonl: success-only output vs. full audit log

- **Options considered:** (a) write every attempted record (success,
  no_quotes, insufficient_quotes, transient_error, permanent_error) to the
  output file, tagged by `status`, as a complete audit trail; (b) write only
  `status == "success"` records to the output file, keeping the cache as the
  complete audit trail instead.
- **Choice:** (b), per explicit user request — `collect()` in
  `scripts/collect_quotes.py` now only appends to `result.new_records` when
  `status == "success"`; `--validate-only` now flags any non-success record
  found in `--output` as a failure.
- **Why:** `quoted_scenarios.jsonl` feeds straight into training-data
  construction downstream; mixing in failed/empty attempts risked a future
  consumer (teacher labelling, prompt building) accidentally treating an
  error row as real data if it forgot to filter on `status`. The cache
  already has every attempt (success or not), so resumability and the
  no-repeat-live-calls guarantee are unaffected — only the *output* file's
  contents changed.
- **What evidence would change it:** if a downstream stage needs to reason
  about failure rates per scenario (not just the aggregate summary counts),
  we'd want those rows back — as a separate file, not mixed into the
  success-only output.

## 2026-09-25 — Scenario diversity: stratified columns vs. plain random sampling

- **Options considered:** (a) sample every field independently via `rng.choice`
  over its enum/range each draw; (b) stratify each dimension (hazard tier, age
  bucket, building type, construction class, roof, attic, ownership,
  deductible) into a balanced, seed-shuffled column, then combine columns by
  index.
- **Choice:** (b), implemented in `_balanced_column` / `_weighted_bool_column` /
  `_weighted_stratum_column` in `src/scenarios/generate.py`.
- **Why:** plain random sampling makes diversity a probabilistic side-effect —
  fine at `n=100` but not guaranteed at small `n`, and not literally "enforced"
  as CLAUDE.md's validation language implies. Stratified columns guarantee
  marginal coverage of every enum value regardless of `n` or seed, while still
  letting cross-dimension combinations vary randomly (columns are
  independently shuffled before zipping).
- **What evidence would change it:** if a future consumer of the scenarios
  needs *joint* (not just marginal) balance across dimensions — e.g. an equal
  count of every (hazard × age × building_type) combination — this design
  would need a full factorial or Latin-hypercube sampler instead.

## 2026-09-25 — Hazard stratum: call assess_wg_risk_context() vs. re-derive postcode logic

- **Options considered:** (a) hardcode/copy the flood-tier postcode-prefix
  sets from `tools/assess_wg_risk_context.py` into the generator; (b) classify
  the supplied known-valid postcodes by actually calling
  `assess_wg_risk_context()` and bucket the results.
- **Choice:** (b), `_classify_postcodes()`.
- **Why:** tools/ must never be modified or duplicated-by-copy — hardcoding the
  private `_FLOOD_HIGH_PREFIXES`/`_FLOOD_ELEVATED_PREFIXES` sets would silently
  drift the moment that file changes upstream. Calling the real function is
  also explicitly what the task spec asked for ("use assess_wg_risk_context()
  to determine the actual hazard stratum"), and it's cheap — the tool is pure
  Python, no network I/O. Restricting the input pool to known-valid postcodes
  avoids fabricating nonexistent locations from random suffixes.
- **What evidence would change it:** none expected; this is strictly safer
  than the alternative and costs nothing.

## 2026-09-25 — Quote cache: treat any cached status as a terminal, reusable attempt

- **Options considered:** (a) only `status: success` records count as
  "reusable" — cache misses on `no_quotes`/`insufficient_quotes`/error records
  would re-attempt the live call on every run; (b) any cached record for a
  matching `normalized_risk`, regardless of status, is reused and never
  automatically retried.
- **Choice:** (b), in `QuoteEngineClient.get_quotes`.
- **Why:** CLAUDE.md rule 2 caps live calls tightly (20 without approval) and
  asks us to be gentle with the engine. A scenario that returned `no_quotes` or
  errored is unlikely to change on retry, and auto-retrying it on every
  `collect_quotes.py` run would silently burn the call budget on scenarios
  that were already resolved (just not successfully). `collect_quotes.py`'s
  dry-run explicitly reports "cache hits" separately from "reusable successful
  cache records" so a non-success cache entry is still visible, just not
  re-fetched.
- **What evidence would change it:** if the live engine turns out to be
  meaningfully non-deterministic run-to-run (e.g. `no_quotes` today, quotable
  tomorrow due to a carrier's own catalogue change), we'd want an explicit
  `--retry-status no_quotes,transient_error` flag rather than blanket reuse.

## 2026-09-25 — Rebuild-sum cost-per-sqm plausibility bounds

- **Options considered:** (a) no plausibility check beyond "positive integer";
  (b) bound both the absolute `rebuild_sum` (€40k–€3M) and the derived
  €/sqm (€800–€5,000) so validation catches a generator bug that produces a
  nonsensical risk (e.g. a 1,000 sqm house rebuildable for €50k).
- **Choice:** (b), `validate_rebuild_sum()` in `src/scenarios/generate.py`.
- **Why:** the golden-set generator (`scripts/build_golden_dataset.py`) uses a
  narrower €1,900–€2,800/sqm range; ours widens it to €1,500–€3,200/sqm on
  purpose for more training diversity, so the validation bound needed to be a
  genuine plausibility check (order-of-magnitude sanity), not a tight
  re-assertion of the sampling range itself.
- **What evidence would change it:** if a real WG underwriting reference
  showed German rebuild costs routinely falling outside €800–€5,000/sqm for
  in-scope building types, the bounds should be widened to match.

## 2026-09-26 — Qwen3.5-2B serving cost shown as a range, not a measured price

- **Options considered:** (a) omit the 2B models from the price–quality figure;
  (b) assign them the 9B managed price; (c) scale the 9B estimate directly by
  parameter count and report a single number; (d) show a sensitivity range
  from parameter-linear scaling to 50% of the 9B price, with the geometric
  midpoint used only to place the marker.
- **Choice:** (d). The resulting estimate is €0.046–€0.093 per 1,000
  recommendations, with a €0.066 display midpoint. It is labelled as estimated
  everywhere and stored in `artifacts/figures/resolved_config.json`.
- **Why:** the repository records local throughput but no A6000 hourly price,
  so an infrastructure-derived cost would create false precision. Parameter
  count gives a useful lower scenario; the wider upper scenario acknowledges
  serving overhead and non-linear provider pricing. A range makes the central
  cost-quality claim visible without presenting a guess as a bill or quote.
- **What evidence would change it:** a controlled serving benchmark with an
  actual hourly GPU or endpoint price, measured utilization/concurrency, and
  the same prompt/completion lengths would replace the estimate entirely.

## 2026-09-26 — Replace the parameter-scaled 2B estimate with measured vLLM economics

- **Options considered:** (a) retain the €0.046–€0.093 parameter-scaled range;
  (b) report only the best saturated benchmark result; (c) report a measured
  utilization curve and choose a clearly labelled planning point.
- **Choice:** (c), which supersedes the estimate above. A RunPod RTX 4090 at
  $0.74/hour served the selected Qwen3.5-2B SFT adapter with vLLM. The cost
  figure now uses a linear axis and shows the measured range from €0.002832/1k
  at sustained concurrency 64 to €0.053899/1k one request at a time. Where a
  scalar `price_assumption_eur_per_mtok` is required, use €0.010640/Mtok at a
  conservative 25% fleet-utilization planning assumption (€0.011327/1k).
- **Why:** self-hosted cost is determined by GPU-hour price, achieved throughput,
  workload token length, batching, and idle capacity—not parameter count alone.
  The sustained run completed 1,200 requests at concurrency 64 at 63.66 rec/s,
  p50 1.003 s, p95 1.105 s, and 90.7% mean GPU utilization, with no failures or
  invalid top-3 JSON. Raw request records and GPU telemetry are retained under
  `artifacts/serving_benchmark/`.
- **What evidence would change it:** production arrival-rate traces, an autoscaling
  policy, redundancy requirements, or a benchmark on the actual deployment GPU
  would change the planning utilization and possibly the instance choice. The
  raw saturated result should remain the hardware-specific lower bound.

## 2026-09-27 — Separate the submission brief from the technical experiment report

- **Options considered:** submit only the required four-page write-up; compress
  all methodological detail into four pages; or deliver a four-page submission
  brief with a separate technical report and evidence register.
- **Choice:** produce a self-contained four-page brief as the official write-up,
  plus a detailed technical report that documents the full experiment and links
  each material claim to frozen repository evidence.
- **Why:** the required brief must remain readable and within the stated limit,
  while the dataset, reward calibration, teacher labelling, SFT, GRPO, serving,
  and reproducibility work contains useful evidence that cannot fit in four
  pages. Keeping the documents separate preserves the limit instead of treating
  an appendix as extra submission pages.
- **What evidence would change it:** submission instructions that prohibit
  supplementary material would leave the technical report in the repository
  only and make the four-page brief the sole uploaded narrative.

## 2026-09-27 — Unresolved: two independent report builds exist, neither promoted

- **Options considered:** two concurrent sessions independently built the same
  pair of deliverables — this one produced `Submission_Brief.pdf` (4 pages) and
  `Technical_Report.pdf` (24 pages) via a hand-authored HTML/CSS + headless-Chrome
  pipeline, plus `documentation/fact_check_table.md`; the other produced
  `reports/wg_recommendation_submission_brief.docx` /
  `reports/wg_recommendation_technical_report.docx` (4 + 20 pages) via
  `scripts/build_final_reports.py` (python-docx), rendered to
  `output/pdf/wg_recommendation_*.pdf`, plus `reports/report_evidence_and_open_questions.md`.
  Both independently reached the same headline numbers and largely the same open
  questions (RunPod A6000 price, Fable wording, golden-disclosure prominence).
- **Choice:** neither was deleted or promoted as canonical. Both sets of files,
  and both sessions' edits to this log and to `AGENTS.md`'s status line, are left
  in place, per explicit instruction, so the owner can compare and pick one
  before submission.
- **Why:** deleting either build unilaterally risked discarding work the owner
  had not yet reviewed; the two builds were generated by separate sessions
  running against the same repository state without coordination.
- **What evidence would change it:** an explicit owner choice of one build (or a
  merge of both) — at that point the losing build's output files, its
  DECISIONS.md/AGENTS.md edits, and any now-redundant fact-check register should
  be removed so the repository has exactly one canonical submission package.

## 2026-09-27 — Resolved: merge the two report builds into one submission pair

- **Options considered:** keep both builds side by side indefinitely (the
  previous entry's holding position); promote the other session's
  python-docx build (`reports/`, `output/pdf/`) as-is and discard this
  session's HTML/Chrome build; or compare both page-by-page and merge the
  stronger elements of each into a single final `Submission_Brief.pdf` /
  `Technical_Report.pdf`.
- **Choice:** merged. The owner asked directly to "pick best of both... mix
  them," rating aesthetics and figure legibility as more important than
  either build's prose alone. I kept this session's HTML/CSS build as the
  base — its statistical framing (Wilson CIs; a single table comparing
  reward-argmax, the untuned 9B baseline, SFT, GRPO, and all three frontier
  models side by side) was more rigorous — and pulled in two things from the
  other build: (1) the owner-confirmed **$4.00 total RunPod bill** /
  **≈$7.80 total recorded cash spend**, which that session had already
  captured in its own evidence register before I could ask the same
  question, added to both documents (brief page 4; technical §13.9, which
  also replaces the prior `[VERIFY]` tags on training GPU cost in §9.6/§13.2
  with a factual "known aggregate, not itemized by stage" statement); and
  (2) full-width sizing for the quality-vs-cost scatter (`07-cost-quality.png`),
  which the other build rendered far more legibly than my original
  52%-width column, plus adding that same figure to the technical report
  (previously brief-only) as its new Fig. 7. Full comparison notes are in
  `documentation/fact_check_table.md` under "Merge pass against the parallel
  Codex build."
- **Why:** both builds drew from the same frozen artifacts, so there was no
  factual conflict to arbitrate — only a design/coverage comparison. A
  targeted merge (two additions, not a rewrite) preserved the more rigorous
  document while fixing its one real content gap (missing total cash-spend
  figure) and one real aesthetic weakness (an under-sized figure), without
  discarding the more thorough statistical framing by starting over from the
  other build.
- **What evidence would change it:** if the owner reviews the merged PDFs and
  prefers the other build's prose/layout wholesale, that build's files
  (`reports/*.docx`, `output/pdf/*.pdf`, `scripts/build_final_reports.py`)
  are untouched and still available to promote instead. Neither build's
  files were deleted in this pass.
