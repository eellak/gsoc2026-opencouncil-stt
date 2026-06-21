# On-box agent brief — fix-task eval harness

> **You are Claude Code running on the mini PC (`harold-venusseries`).** You have
> no prior conversation context — this file is your full instruction set. Build
> and run the text-only eval harness described here. Work in this repo
> (`~/opencouncil-fine-tuning`). Read the linked specs before coding.

## Read first (source of truth)

1. [`docs/specs/fix-task-eval-harness.md`](../specs/fix-task-eval-harness.md) —
   the full design: chain logic, splits, layered scoring, glossary A/B, routing.
   **Follow it.** This brief only adds execution order + on-box specifics.
2. [`docs/reference/fix-task-prompt-v2.md`](../reference/fix-task-prompt-v2.md) —
   the verbatim baseline prompt you are A/B-testing.
3. [`docs/reference/ui-error-categories.md`](../reference/ui-error-categories.md)
   and [`error-taxonomy.md`](../reference/error-taxonomy.md) — category labels +
   routing buckets.

## Mission

Iterate the OpenCouncil fix-task **prompt** (text-only, no audio) so it (a)
reproduces what the current task already fixes and (b) also catches the residual
errors the task missed (`task→user` chains). Primary lever: inject a per-city +
global **glossary** mined from the corrections. Secondary deliverable: a
**routing report** — per error category, can an improved prompt reliably fix it
(→ exclude from Whisper finetune data) or not (phonetic/acoustic → keep for ASR
finetune).

## Environment (already set up)

- Data: `data-1779206108158.csv` at repo root (235M, 393 970 rows). Columns:
  `edit_id, utterance_id, edit_timestamp, edit_updated_at, before_text,
  after_text, edited_by∈{task,user}, utterance_start, utterance_end, audio_url,
  youtube_url, meeting_name, meeting_date, meeting_id, city_id`.
- Python venv: `.venv-eval/` (has `pandas`, `rapidfuzz`, `anthropic`).
  Activate: `source .venv-eval/bin/activate`.
- `claude` and `codex` CLIs are on PATH in a **login shell**
  (`~/.local/bin`). `node` is at `~/.local/opt/node/bin`.

## Inference path — IMPORTANT (no API key)

There is **no `ANTHROPIC_API_KEY`** here. Do **not** use the `anthropic` SDK for
fix-calls. Instead make each fix-call by shelling out to the **on-box `claude`
CLI in print mode**, which uses the existing Claude Code auth:

- Build one thin wrapper `fix_call(system_prompt, user_prompt) -> str`.
- Use `claude -p` (print/headless) with the fix-task **system prompt** applied,
  **tools disabled**, deterministic, output = raw text only. Discover the exact
  flags yourself: run `claude --help`, check `-p`, `--append-system-prompt` /
  system-prompt override, `--allowedTools ""` / `--disallowedTools`,
  `--model` (target **sonnet**, matching production), and any
  `--output-format`. **Verify on 3 hand-picked rows** that output is clean
  numbered lines (same count, no preamble, no tool use) before scaling.
- The orchestrator (you) must **not** make the fix-calls inline in your own
  context — write a **script** that loops and subprocesses `claude -p`. Keep your
  own context small; you build, test, launch, and monitor.

If the CLI route proves too slow/unreliable for the sample size, stop and report
back — do not silently switch to an unauthenticated SDK path.

## Execution order

### Step 0 — TDD self-tests first (write before harness code)

Implement the 3 self-tests from the design spec, watch them fail, then build the
code that makes them pass:

1. **Chain integrity** — synthetic CSV (task-only / user-only / task+user /
   mixed-semantic chains) ⇒ reconstruction yields correct first-`before` &
   final-`after`, preserves `edited_by` tags, buckets mixed/semantic per policy.
2. **Leakage** — two-fold split with a glossary term only in the held-out fold ⇒
   glossary builder excludes it; eval prompts provably never contain it.
3. **Scoring normalisation** — Greek pairs differing only by
   accent/punctuation/final-sigma/spacing score equivalent; a real lexical
   correction is still caught; overcorrection stays 0 on normalised-equiv changes.

Put tests under `eval/tests/`, code under `eval/`. Use `pytest`.

### Step 1 — chains

Reconstruct per-utterance chains from the CSV (sort by timestamp; verify
`before==prev after`). Emit `(input_raw=first before, gold_final=last after,
edited_by_seq, has_task, has_user, task_then_user, chain_type)`.
Classify `chain_type ∈ {pure_correction, mixed, semantic_rewrite}` (heuristic:
length delta, edit distance, presence of meaning change). **Score only
`pure_correction` as primary**; report the rest separately.

### Step 2 — splits (leakage control)

Split **by `meeting_id`** (train/eval). Glossary mined from **train meetings
only**. Add a second split **by `city_id`** for a zero-shot-city eval. Assert no
held-out glossary term leaks into eval prompts.

### Step 3 — glossary mining (train split only)

Global + per-city terms: all-caps Greek acronym tokens, capitalised multi-token
names, org/party patterns, frequent `before→after` entity substitutions; group
global vs per-city by cross-meeting frequency. Save to `data/glossary/`.

### Step 4 — scoring (layered, Greek-normalised)

Normalise (accent-, punctuation-, final-sigma-, whitespace-insensitive) then:
edit-application accuracy on the targeted diff span; normalised exact match;
overcorrection/harm rate; separate un-normalised surface-fidelity. Stratify by
category and by `edited_by` class.

### Step 5 — A/B run (the long job — in tmux, checkpointed)

Stratified sample: ≥100 held-out rows per high-priority category; for sparse
categories (`acronym_abbreviation`, `place_name`, `person_name`) use all
held-out and report uncertainty. Run **baseline** vs **glossary-augmented**
prompt over the same rows. **Checkpoint every N rows to disk** (resume, don't
restart). Cheap pass = per-utterance; then a **segment-level subset** (reconstruct
segments — if meeting JSON isn't cached, group consecutive same-meeting
same-speaker utterances as a proxy and note the caveat) for the context-sensitive
categories, to measure the per-utterance-vs-segment gap.

### Step 6 — routing report

Per category, LLM-fix rate for **baseline AND glossary**; route to LLM-only only
when both agree it's reliably fixed. Write `data/reports/fix-task-eval/` with: per-
category tables (baseline vs glossary, task vs user, pure vs mixed), the
per-utterance-vs-segment gap, the routing recommendation, and example wins/losses.

## Operational rules

- **Long runs in tmux** (the human may disconnect). Checkpoint per batch.
- Keep your orchestrator context lean — push work into scripts + subprocesses.
- Commit code/specs if asked, but **never commit** secrets, `.env`, or raw API
  output dumps beyond the reports.
- Log progress to a file (`eval/run.log`) and print a short status line each batch.
- If you hit a real blocker (inference path, ambiguous policy, leakage you can't
  prevent), **stop and write the question** to `eval/QUESTIONS.md` rather than
  guessing.

## When done

Write a summary to `data/reports/fix-task-eval/SUMMARY.md`: did glossary help, per
category; which categories are prompt-reliable vs ASR-finetune; the
per-utterance-vs-segment gap; and the top open questions. That summary is what
gets pulled back to the Mac.
