# The two dense validation references were not repaired, and why that is the finding

Date: 2026-08-19. Experiment: `exp-2026-08-19-dense-reference-repair` (CLOSED).
Protocol: [`2026-08-19-dense-reference-repair.md`](../specs/2026-08-19-dense-reference-repair.md).

## What was attempted

The blind listening audit judged both insertion-heavy validation windows
`material_omission`, so the dense arm's extra words could not be split into genuine
insertions and recovered speech. This experiment set out to produce an audio-faithful
reference for exactly those two windows — `win_argos_oct31__2_2025_2353650` (149.675 s,
244 published tokens) and `win_argos_sep24_2025_371824` (149.263 s, 313 tokens).

Two review designs were built and served privately:

1. **Hidden-reference, audio-first.** The published text was withheld; the reviewer
   walked 20 s intervals and marked each one as complete or annotated what was missing.
2. **Interval repair.** The published reference was force-aligned onto the audio with
   the CTC aligner already used by `anchor_timings.py`, split by aligned start time into
   20 s intervals, and shown editable beside per-interval playback.

## What happened

Design 1 was rejected immediately and correctly: a listener cannot say what is absent
from a text they are not shown. The question was unanswerable as posed.

Design 2 was usable but stopped in trial. The reviewer worked through 5 of the first
window's 8 intervals and found material to add in **every** interval. Both windows are
continuous interruption and simultaneous speech: several people talking over each other,
with the published reference capturing the microphone speaker and dropping the rest. The
reviewer's judgement was that a reference produced this way would be neither reliable
nor cheap — high effort per minute, and an output that still would not be a defensible
gold.

Partial answers remain private under
`~/.cache/oc-public/dense-reference-repair-2026-08/`. They are not an artifact and
nothing is scored from them; `eval/results_dense_reference_repair.json` records the
incomplete state only.

## What this does and does not settle

- **Unresolved:** whether the dense arm's extra words on these two windows were
  insertions or recovered speech. `SCREEN — STOP` stands on its own preregistered
  gates and is unaffected either way.
- **Not established:** any rate of omission in the published references generally. Two
  outcome-selected windows estimate nothing about the corpus.
- **Established as experience, not measurement:** on heavily overlapped council audio, an
  audio-faithful reference is expensive to produce and hard to trust even with the
  published text time-aligned in front of the listener. Every interval attempted needed
  additions.

That last point redirects the work. If overlap is what makes reference text unreliable,
it is also what makes training labels unreliable, and it is cheaper to *select* against
overlap in the training data than to *repair* references after the fact. The next
question is therefore a data-selection question — single-speaker, overlap-free,
clean-boundary clips packed to a useful density — and not another reference audit.

## Reusable harness

- `eval/controlled_eval/align_published_reference.py` — force-aligns a published
  reference onto window audio; the timings reproduce the reference tokens exactly.
- `eval/controlled_eval/build_reference_repair_page.py` — private interval review page,
  model-blind, with per-interval playback and autosave.
- `eval/controlled_eval/score_reference_repair.py` — content-free aggregate with the
  S/D/I arithmetic identity asserted.
- `eval/controlled_eval/audit_server.py` — now takes `HOST` (default unchanged).
