# If we built the acoustic model again from scratch (2026-07-26)

A design note, not a plan of record. Written after the
[fine-tune postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md) and the
[biasing runs](../reports/2026-07-25-hotwords-biasing.md), while the evidence is fresh.
The question it answers: *what would we change, and why*.

> **Tested (2026-07-29):** the central "lexical costume" thesis was put through three
> controlled experiments — it holds only in weakened form (≈half the edit volume is
> genuinely acoustic; whisper-hotword biasing is saturated), but the LLM post-editor
> it recommended is now the best measured system on the corrected subset (0.119 WER).
> See [reports/2026-07-29-lexical-thesis-experiments.md](../reports/2026-07-29-lexical-thesis-experiments.md).

## What the evidence says the problem actually is

Three measurements from this project, taken together, point somewhere different than the
original plan assumed:

1. On the corrected held-out utterances, **Scribe's raw output (0.155 WER) was already
   closer to the human reference than base whisper (0.158) or the fine-tune (0.176)**. On
   those utterances the acoustic model was not the bottleneck.
2. On name-heavy real utterances, every model sits at **WER ~0.31–0.35** — twice the
   corrected-subset figure. Those clips are the hard ones (far-field mics, PA reverb,
   crosstalk, overlap). That is where acoustics genuinely fail.
3. **Roster hotwords bought +8.8pp name recall for free** (p = 0.021), while 8h31m of GPU
   fine-tuning bought nothing measurable.

So the correction dataset is mostly a **lexical and textual** signal (names, domain terms,
punctuation, casing) wearing an acoustic costume. The acoustic frontier is elsewhere: noisy
far-field council-chamber audio and overlapping speech.

## What we would change

### 1. Build the evaluation before the dataset — and make it a gold set

The single mistake that invalidated a month of numbers was comparing models across serving
stacks. The second, quieter one: `val_manifest.csv` references are **uncorrected
OpenCouncil text**, so we were scoring against noisy targets and calling it WER.

- ~2–3 hours of audio, **human-verified transcripts**, spread across cities and recording
  conditions, frozen before any training, never used for anything else.
- Three slices reported together, always: general held-out · name-focused · corrected.
- One command, one serving stack, one normalization. Any number that cannot be reproduced
  by that command does not get quoted.
- Report entity/name accuracy and number accuracy next to WER. A council transcript that
  gets every word right except the councillor's surname and the budget figure is broken in
  the way that matters.

Cost: a few days of transcription. It is the cheapest thing on this list and it is what
would have caught the artifact on day one.

### 2. Design a *contextual* ASR system, not a better model

Before a single second of a council meeting is transcribed, the pipeline already knows: the
speaker roster, the parties, the agenda/subjects, the city's glossary, and the transcripts
of every previous meeting in that municipality. A generic ASR model cannot have that.
Exploiting it is the structural advantage, and it is where our one confirmed win came from.

That reframes model choice. The question is not "which model has the lowest published Greek
WER" but **"which model lets me inject per-meeting context at decode time?"**

- Whisper prompt `hotwords` — works (proved), but it is a soft nudge in a 224-token window;
  it does not scale to a 500-term glossary.
- A CTC or transducer stack with **word-level boosting / shallow fusion of a per-meeting
  LM** — a much stronger lever for exactly this problem, and it degrades gracefully with
  hundreds of terms. *To verify before committing: current Greek quality of the candidate
  toolkits and whether their boosting APIs are still maintained.*
- An LLM post-editor conditioned on the roster + glossary — strongest lever on the textual
  errors, and cheap to prototype. Untried so far; ranked #5 in the postmortem, probably
  underrated.

Design goal: the acoustic model produces a faithful, context-free transcript; the context
gets applied where it is cheap to change and easy to evaluate.

### 3. Change the training unit and the data mix

If we do fine-tune acoustically:

- **Train on 30-second context windows, not isolated cut utterances.** Whisper is a
  window model. Feeding it surgically cut single utterances taught it that every segment
  starts clean — which is the onset-dropping we observed in the fine-tune's output. This is
  a bug in the data preparation, not in the recipe.
- **Do not train mostly on corrections.** Corrections are a doubly-biased sample: they are
  where the ASR failed *and* where a human bothered to fix it. Train on that and the model
  learns the error distribution, not the language. Mix: majority verified-clean audio, a
  minority of hard corrected cases.
- **Drop the text-only edits from the training targets entirely.** Punctuation, casing and
  spelling normalization are not acoustic problems; including them teaches the acoustic
  model a text-formatting task it will do badly and inconsistently.
- Keep the whole-meeting, held-out-city split. That part was right.

### 4. Route errors by who can actually fix them

Decide this before collecting data, not after:

| Error class | Fixed by | Evidence |
|---|---|---|
| Names, parties, local toponyms | Contextual biasing at decode | +8.8pp recall, p = 0.021 |
| Domain terminology, legal formulas | Biasing + glossary; LLM post-edit | untested, cheap |
| Punctuation, casing, number formatting | Post-processing / LLM | not an acoustic task |
| Far-field noise, reverb, PA distortion | Acoustic training or better audio front-end | WER 0.34 on hard clips |
| Overlapping speech, crosstalk | Diarization + separation front-end | unmeasured, likely large |
| Dialect, accent, fast disfluent speech | Acoustic training on more in-domain hours | plausible |

Only the bottom three justify GPU time. The fine-tune we ran spent it on the top three.

### 5. Judge everything by human edit effort, not WER alone

This project has something most ASR work does not: a **production signal for how much
humans have to fix**. The review pipeline measures edits per meeting. That is the honest
north-star — "did the transcript get cheaper to correct?" — and it is already instrumented.
WER stays as the comparable, publishable metric; edit effort decides whether a change is
worth shipping. (Related: [metric-hir.md](../decisions/metric-hir.md), where HIR was pushed
back on as *the* metric — the point here is narrower, as a shipping gate, not a benchmark.)

## What we would keep

- The review UI and the correction dataset pipeline — the data is genuinely good, it was
  just pointed at the wrong task.
- The whole-meeting, held-out-city split.
- The cheap CPU eval loop on the mini-PC: 50–60 clips through two models in ~15 minutes is
  a fast enough iteration cycle to keep everyone honest.
- The roster/glossary assets (`data/pii/rosters_full.json`, `data/glossary/glossary.json`)
  — they turned out to be the most valuable artifact in the repo.
- Publishing negative results. The postmortem is worth more than the adapter.

## Open questions before any of this becomes a plan

- Does the OpenCouncil production pipeline have the per-meeting roster available at
  transcription time, reliably, for every meeting? (Believed yes; confirm.)
- How much overlapping speech is there really? Nobody has measured it, and it may dominate
  the hard-clip WER.
- Is there enough verified-clean Greek council audio to train on, or does the correction
  pipeline have to produce it first?
- Current state of Greek support and context-boosting APIs in the non-whisper toolkits —
  needs research, not recollection.
