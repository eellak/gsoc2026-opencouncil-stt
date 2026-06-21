# Experiment log — fix-task prompt eval harness

Detailed running log of the on-box (harold-venusseries) text-only eval. The
machine-readable timeline lives in [`timeline.json`](timeline.json) and feeds the
HTML report (`data/reports/fix-task-eval/report.html`). This file is the prose
narrative: what was tried, what broke, what was decided.

## Environment

- Box: `harold-venusseries`, Ubuntu, 16 cores, 60 GB RAM.
- Data: `data-1779206108158.csv` (235 MB, 393,970 rows, 287,605 utterances).
- venv: `.venv-eval` (pandas, rapidfuzz, anthropic, pytest, pyarrow).
- Inference: **no `ANTHROPIC_API_KEY`** → `claude -p` (sonnet) over Claude Code
  OAuth. Flags: `--system-prompt` (full override → clean output), `--allowedTools ""`
  (tools off), `--output-format text`. NOT `--bare` (it forces API-key auth).

## Step 0 — TDD self-tests (2026-06-20 ~17:55)

Wrote the 3 spec self-tests first (chains / leakage / scoring) + a categorizer
sanity test, watched them fail (modules absent), then implemented to green.
**25 tests pass** (`eval/tests/`). Modules: `chains.py`, `splits.py`,
`glossary.py`, `scoring.py`, `categorize.py`.

## Steps 1–3 — dataset build (18:05)

`python -m eval.build_dataset` → 287,605 chains.

- **Bug caught:** `edit_timestamp` is a datetime *string*. Initial
  `pd.to_numeric(errors="coerce")` nulled the whole column, so the timestamp sort
  was a no-op and chains kept CSV order (newest-first → reversed). Symptoms:
  `links_ok_rate` 0.73 (should be ~1.0) and `task_then_user` 247 (should be ~73k).
  **Fix:** `pd.to_datetime`. After fix: links_ok **99.97%** (spec 99.997%),
  task_then_user **73,600** (spec ~73,643). Chain logic validated against the
  spec's published stats.
- chain_type split: pure_correction 235,118 / mixed 37,417 / semantic_rewrite 15,070.

- **Glossary design fix:** first cut produced **0 per-city terms** — the "global =
  ≥2 meetings anywhere" rule subsumed the per-city bucket. Redefined: **global =
  term spans ≥2 cities; per-city = recurs across ≥2 meetings within one city.**
  Result: 5,894 global + 3,543 per-city across 11 cities. Per-city terms look
  right (e.g. `Βάρκιζα` for vari-voula, `Άλσος Πετραλώνων` for athens).

## Inference path verification (18:17)

`claude -p` on 3 hand-picked held-out rows → **3/3 clean numbered lines**, same
count, no preamble, no tool use. Real homophone fix observed
(`ηλίκιο`→`λύκειο`, `συλληπιθώ`→`συλλυπηθώ`). Latency ~3.6–7 s/call (avg 4.8).

## Step 5 — sampling + A/B (18:18–)

- No category column in the CSV → built a **heuristic, text-only categorizer**
  (no audio, no NER). Merges person/place/org → `named_entity`,
  verb/noun/article → `morph_grammar`. Explicitly triage, not ground truth.
- Held-out category distribution healthy: every category ≥100 rows. Stratified
  sample = **1000 rows, 100/category, 10 categories**.
- **Glossary retrieval** (per utterance, fuzzy from input only): tightened to
  length≥4 + cutoff 88/90 after seeing short-token distractor noise
  (`καλησπέρα`→`Εμένα,Ένα,ΑΕΠΟ…`). After: real entities survive
  (`Ξυλόκαστρο`), most noise gone.

### Incident — first A/B run (18:19 → 18:55)

Launched workers=8. **259/1000 ok, 741 `exit 1` failures** starting ~17 min in
(empty stderr) — a sustained-concurrency rate/usage throttle. A single manual
call worked again afterward (limit reset).

**Fixes:** (1) resume now excludes error rows from the done-set so failed rows are
retried; (2) longer backoff (5/15/30/60/90 s, 6 tries). Relaunched **workers=4**,
resuming the 741.

## Reporting

- `eval/report.py` → markdown tables + `SUMMARY.md`.
- `eval/report_html.py` → self-contained `report.html` (timeline + live results),
  regenerable mid-run.

## Codex review (acted on, 2026-06-20 20:10)

Codex (high effort) found real scoring bugs. Fixed via TDD:
- **Pure deletions** previously always scored `edit_application=1.0` — now require
  the spurious token to actually be gone.
- **Hallucinated insertions** previously counted as zero harm — overcorrection now
  counts every token the model added/dropped that gold didn't.
- Multiset-based scoring → robust to duplicate tokens (no positional aliasing).
- Categorizer: `number_date` now catches digit *changes/removals* (not just
  introductions); `named_entity` checked before `insertion_deletion`;
  `final_sigma` made reachable.
- Reports **recompute** category + scores from stored outputs, so fixes apply
  uniformly to all rows with no LLM re-calls. 32 tests green.

## Result (956/1000 rows, strict scoring)

- **Glossary helps `named_entity` +9.9pp** — the hypothesised win.
- Glossary mildly **hurts** most other categories (acronym −6.5, number_date −6.0,
  accent −5.9) via retrieval-noise overcorrection.
- `task_then_user` residuals hardest: 42.6% baseline edit-applied, 16% exact.
- **Conclusion: apply the glossary selectively to entities, not globally; improve
  retrieval precision before any broad rollout.**

## Quota incident + cost strategy (2026-06-21)

- Re-run hit the **Claude session limit** (`resets 6am Europe/Athens`); stalled
  overnight retrying, finished after reset. 956/1000 done.
- The user refactored `fix_call.py` to a process-wide **SerializedRetryGate**
  (serializes `claude -p`, shared cooldown) to stop OAuth retry storms.
- Cost fixes: **empty-glossary skip** (−17% calls); **pluggable backends**
  (`eval/backends.py`): claude / codex (gpt-5.4-mini low) / gemini(stub). Use
  cheap backends for sweeps, sonnet for final validation.

## Open / pending

- Finish the 44 missing rows (in progress) + small segment-level proxy pass.
- Gemini Flash backend: needs a `gemini` CLI or API key.
- The actual improvement loop: test prompt variants (acronym/legal instruction,
  selective entity-glossary, few-shot) on a cheap backend, validate on sonnet.
