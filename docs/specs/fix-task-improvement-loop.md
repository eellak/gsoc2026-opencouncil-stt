# Spec — LLM transcript-fix task improvement loop

> Status: **accepted as worth doing, not the main point.** The main track is the
> Whisper finetuning pipeline. This is a parallel side-project, captured here so it
> is not lost. Raised by Ethniki, agreed by Christos in the
> [2026-06-16 sync](../meetings/2026-06-16.md).

## What the "fix task" is

OpenCouncil already runs an **LLM transcript-fix task**: after STT produces a raw
transcript, an LLM corrects it (the prompt is public in the OC task code; it
changed recently — current version to be located, see prereq below). This task is
separate from the STT finetuning — it cleans up whatever the transcription model
produced.

## Goal

Improve the **prompt** the fix task gives the LLM so it corrects Greek
municipal-council transcripts better — measured, not guessed.

## Approach — automated experiment loop

A Karpathy-style "auto-research" loop:

1. Take held-out council audio with human-corrected reference.
2. Run transcription → run the fix task with a candidate prompt → get the LLM's
   corrected output.
3. Score the output against the human reference on defined eval metrics.
4. Keep scores; try other prompt variants / experiments; iterate to maximise the
   metric.

Model: a **Claude** model for the fix task (simple to wire up).

## Eval metrics (to define)

- WER / **NE-WER** of the fixed transcript vs human-corrected reference, on a
  held-out set (reuse the finetune test set / bench where possible).
- Ideally per-error-category, using the review-tool taxonomy.

## Where the prompt lives (located 2026-06-16 via deepwiki-research)

- Repo **`schemalabz/opencouncil-tasks`**, file **`src/tasks/fixTranscript.ts`**
  (the `/fixTranscript` endpoint). Calls **claude-sonnet-4-6** directly (Adaline
  removed Apr 2026).
- The recent change Christos mentioned = commit **`4624bac1`** (2026-06-11),
  *"feat: retune fixTranscript for Scribe v2 clean-read output"* — task version 2.
- **Verbatim prompt text captured** (2026-06-20):
  [fix-task-prompt-v2.md](../reference/fix-task-prompt-v2.md).
- **Structure of the current prompt:**
  - Input: one speaker segment as numbered lines; output the same lines, same
    count, no merge/split/reorder (timestamps ride on boundaries). Up to 3 retries
    with `parseNumberedUtterances` validation.
  - Fix priority: **1. Names** (check against roster + optional agenda) →
    **2. homophone misspellings** (ο/ω, η/ι/υ, αι/ε) → **3. house style** (numbers
    & dates as digits) → **4. punctuation/accents** (Greek «;», τόνοι, casing).
  - Don't-touch: meaning, speaker grammar/colloquialisms, crosstalk fragments,
    low-confidence words ("an unfixed error is recoverable; a wrong fix corrupts
    the record").
  - User prompt injects: city, speaker, roster (`Party: members`), optional agenda.

## Improvement hypotheses (grounded in project stats — to validate, not assumed)

The prompt was **freshly retuned (2026-06-11)** for Scribe v2, so it is already
good. Highest-value move is **measure its per-category lift first** (raw STT →
fixed, on held-out, per error category) — that says *whether* there is headroom and
*where* — before touching prompt text. Candidate gaps vs the error clusters in
[finetuning-101](../reference/finetuning-101.md#why-were-doing-this):

- **Acronyms / legal references** (ΚΕΔΕ, ΟΠΕΚΑ, ΦΕΚ, Ν.4412/2016, decision numbers)
  are a named error cluster but are **not** an explicit priority category in the
  prompt. Could add one, sourced from a domain glossary.
- The prompt grounds names on roster + agenda but **not on a domain glossary** of
  acronyms/terms. The entity list already exists in Postgres → feed a per-meeting
  or global glossary.
- Optimise/measure on **NE-WER** and per-category WER, not plain WER.

## Prerequisites / open

- [x] Locate the current fix-task prompt → `opencouncil-tasks/src/tasks/fixTranscript.ts`.
- [ ] Measure the **current fix-task lift per error category** before changing it.
- [ ] Define the eval harness for the fix task (can share the finetune bench).

## Relationship to other ideas

Distinct from, but complementary to, the **LLM-ensemble provider** idea
(dual/triple-model transcription + LLM-as-judge) tracked in the
[2026-06-16 meeting note](../meetings/2026-06-16.md): that improves *transcription*;
this improves the *post-transcription correction* prompt.
