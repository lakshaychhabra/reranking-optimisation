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
