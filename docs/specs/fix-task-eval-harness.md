# Spec — fix-task prompt eval harness (text-only)

> Status: **design, under review.** Implementation plan for the
> [fix-task improvement loop](fix-task-improvement-loop.md). Reviewed by Codex
> (high effort, 2026-06-20) — its corrections are folded in and marked
> _[Codex]_. Decisions taken with the user: text-only first; glossary→prompt is
> the lever; STT/Gladia path out of scope (see
> [dynamic vocabulary](../reference/dynamic-vocabulary-and-entities.md)); the
> heavy run is offloadable to the mini PC.

## Goal

Cheaply (text-only, no audio) iterate the fix-task **prompt** so it:

1. **Reproduces** what the current task already fixes — no regression.
2. **Also catches** the residual errors the task missed — the `task → user`
   chains. Headroom = fewer corrections for humans now, fewer residual errors for
   the finetuned ASR model later.

Primary lever: inject a **per-city + global glossary** (acronyms, toponyms, orgs,
legal terms) into the prompt, mined from our correction CSV.

Secondary output: a **routing report** — per error category, can an improved LLM
prompt reliably fix it (→ keep out of the Whisper finetune training set) or not
(phonetic/acoustic → keep for ASR finetune).

## Data we have (verified 2026-06-20)

`data-1779206108158.csv` — 393 970 rows, 287 605 unique utterances. Columns:
`before_text, after_text, edited_by∈{task,user}, utterance_id, meeting_id,
city_id, timestamps`. Edits form **clean sequential chains**: 106 362/106 365
chain links have `before_text == previous after_text`. Counts: `task` 207 594,
`user` 186 376. 85 857 utterances have multi-edit chains; dominant pattern
`task>user` (59 974) — 73 643 chains have a task edit before a user edit.

⇒ In a `task>user` chain the user edit's input **is** the task's output, so user
edits are the residual errors the fix-task left uncorrected. Eval input = **first
`before_text`** of a chain (raw STT); gold reference = **final `after_text`**.

## Design

### 1. Chain reconstruction + classification _[Codex #1, #5]_

- [ ] Reconstruct per-utterance chains; derive `(input_raw, gold_final)` and the
      `edited_by` sequence.
- [ ] Classify each chain — **only pure-correction chains are scored as
      primary**:
  - `pure_correction` — orthographic/lexical/entity fixes only.
  - `mixed` — correction + rewrite.
  - `semantic_rewrite` — meaning changed (not an ASR fix).
  Mixed/semantic go to a **reported-but-not-scored** bucket. Heuristic first
  (length delta, edit distance, semantic_rewrite category), refine on a sample.
- [ ] Tag each chain `has_task`, `has_user`, `task_then_user`. Report metrics
      **separately** for: task-only, user-only residual, task+user, pure vs mixed.

### 2. Splits — leakage control _[Codex #2]_

- [ ] **Split by `meeting_id`** (never by utterance). Glossary is mined from the
      **training meetings only**; eval rows never contribute glossary terms.
- [ ] Nested guardrail: build glossary from earlier meetings, eval on later
      held-out meetings per city.
- [ ] Second split **by `city_id`** for a true zero-shot-city test (does a
      global glossary + prompt help a city whose terms we never saw?).
- [ ] A leakage assertion in the harness: prove eval prompts never contain a
      held-out glossary term (self-test #2 below).

### 3. Candidate prompts (A/B)

- [ ] **Baseline** = current verbatim prompt
      ([fix-task-prompt-v2.md](../reference/fix-task-prompt-v2.md)).
- [ ] **Glossary-augmented** = baseline + a glossary block in the user prompt
      (global terms + this city's terms), parallel to the existing roster/agenda
      injection.
- [ ] Glossary mining from training split: all-caps Greek acronym tokens,
      capitalised multi-token names, org/party patterns, frequent
      `before→after` entity substitutions; grouped **global vs per-city** by
      cross-meeting frequency.
- [ ] Model: Claude sonnet-4-6 (matches production fix task).

### 4. Granularity — two passes _[Codex #3]_

- [ ] **Cheap sweep:** per-utterance (single line) over the full stratified
      sample. Fast, broad, but loses cross-line context.
- [ ] **Segment-level validation subset:** reconstruct true speaker segments
      from cached meeting JSON for the context-sensitive categories
      (`person_name`, `org_party_name`, `acronym_abbreviation`, `place_name`,
      `word_boundary`) and re-run there. Guards against over-estimating recall on
      exactly the classes the glossary targets.

### 5. Scoring — layered _[Codex #4]_

Greek-normalise before scoring (accent-, punctuation-, final-sigma-,
whitespace-insensitive), then compute:

- [ ] **Edit-application accuracy** — was the targeted `before→after` correction
      span actually applied? (token-level alignment on the diff span, not
      full-line exact match).
- [ ] **Normalised exact match** on the canonical form (sanity).
- [ ] **Overcorrection / harm rate** — tokens changed that gold left unchanged.
- [ ] **Surface-fidelity** score computed separately (un-normalised), so we can
      see punctuation/casing/digit-style behaviour without it polluting the
      correctness signal.
- Raw diff-token alignment kept only as a **debugging** metric, not headline.

### 6. Routing report _[Codex #6]_

- [ ] Per category, compute the LLM-fix rate for **baseline AND glossary**
      prompts; route a category to LLM-only when **both** agree it is reliably
      fixed. Define routing on stable failure patterns, not on a single prompt's
      misses (avoid encoding prompt limits as "acoustic-only"):
  - prompt-reliable → orthographic, lexical, acronym, entity normalisation with
    text context;
  - ASR-finetune → phonetic substitutions, acoustic segmentation errors, heavy
    OOV misrecognition, non-local hallucination.

### 7. Sampling _[Codex]_

- [ ] ≥ **100 held-out rows per high-priority category**; for sparse categories
      (`acronym_abbreviation`, `place_name`, `person_name`) use all held-out
      examples and report **uncertainty**, not point estimates.

## Offload to the mini PC

Long runs (LLM calls over thousands of rows) run on the mini PC, not this Mac, to
survive disconnects. Verified env: `ssh minipc` → Ubuntu 24.04, node v22,
python 3.12, 16 cores, 60 GB RAM, 199 GB free, tmux. Repo **not yet present**.

- [ ] `rsync` repo + the CSV (or a slimmed parquet of needed columns) to
      `minipc:~/opencouncil-fine-tuning`.
- [ ] `ANTHROPIC_API_KEY` available on the mini PC.
- [ ] Run the harness inside **tmux**; **checkpoint per batch** to disk so a
      dropped connection resumes, not restarts.
- [ ] Pull results (`data/reports/`) back to the Mac/vault.
- A runbook will be written at implementation time.

## Harness self-tests (TDD — write first) _[Codex]_

1. **Chain integrity** — synthetic CSV with task-only, user-only, task+user, and
   mixed/semantic chains ⇒ reconstruction yields correct first-before /
   final-after, preserves task/user tags, buckets mixed/semantic per policy.
2. **Leakage** — two-fold synthetic split with a glossary term only in the
   held-out fold ⇒ glossary builder excludes it; eval prompts provably never
   contain it; the held-out acronym/name metric drops once leakage is removed.
3. **Scoring normalisation** — paired Greek strings differing only by
   accent/punct/final-sigma/spacing mark equivalent; a real lexical correction is
   still caught; overcorrection stays zero on normalised-equivalent changes.

## Decisions (2026-06-20, with user)

- **Glossary granularity = global + per-city (`cityId`) only.** Not
  per-administrative-body (terms don't vary by body within a city — a tag at
  most, not a scope) and not per-agenda/per-meeting (meeting entities are already
  injected via the existing `agendaBlock`; the store holds stable terms, the
  agenda stays dynamic).
- **Per-utterance vs segment is not pre-decided — the harness measures it.** Run
  both on the same subset, report the per-category gap. Default: trust the
  per-utterance sweep for non-context categories (homophone, accent, number);
  use the segment subset to calibrate name/acronym/place recall.

## Open questions

- [?] Provenance edge: are some `task` edits themselves non-trivial rewrites
      (so "reproduce the task" isn't purely ASR correction)? Sample-check during
      chain classification.

## Links

- Loop spec: [fix-task-improvement-loop.md](fix-task-improvement-loop.md)
- Prompt: [fix-task-prompt-v2.md](../reference/fix-task-prompt-v2.md)
- Vocabulary: [dynamic-vocabulary-and-entities.md](../reference/dynamic-vocabulary-and-entities.md)
- Routing taxonomy: [error-taxonomy.md](../reference/error-taxonomy.md)
- Upstream feature issue: [oc-dynamic-vocabulary.md](../issues/oc-dynamic-vocabulary.md)
