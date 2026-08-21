# Frozen criteria: overlap-free clip selection for the next training arm

Frozen 2026-08-19, before any diarization job was submitted, so the thresholds cannot
be tuned to make the surviving fraction look good.

## Why this replaces reference repair

`exp-2026-08-19-dense-reference-repair` was stopped in use: on heavily overlapped
council audio a human cannot cheaply write a reference anyone should trust
([report](../reports/2026-08-19-dense-reference-repair-stopped.md)). The same overlap
that makes a reference unreliable makes a *training label* unreliable. It is cheaper to
select against overlap than to repair text afterwards.

Scope, decided by the user on 2026-08-19: **in-domain OpenCouncil clips only**
(42,289 utterances, 45.21 h of speech, 190 meetings, 8 cities, 267 audio files).
External packs (hparl2/stoma/cv/eurospeech) are out of this arm — their audio context
is not available, so boundaries and neighbours cannot be judged.

## Unit of processing

One diarization job per **audio file**, on the full meeting recording, never on the cut
clip: three seconds cannot establish how many people are speaking or what surrounds the
cut. Speaker labels are local to one job — a label in one file says nothing about a
label in another, so nothing may be grouped across files.

Diarization is the non-exclusive `precision-2` output, which preserves overlapping
turns. `exclusiveDiarization` must not be used for acceptance: it removes by
construction the evidence this selection depends on.

## Frozen acceptance predicate

For a training utterance over `[start, end)` of duration `D`, with the guard band
`[start - 0.3, end + 0.3)` clipped to the recording:

1. **Dominant speaker.** Let each label's coverage be the seconds it is active inside
   `[start, end)`. The primary label is the one with the largest coverage; a tie is a
   rejection. Any number of separate turns may carry the primary label — a diarizer
   splits one person at every pause.
2. **Coverage.** `primary_coverage / D >= 0.90`.
3. **Competing speech inside.** Seconds where a non-primary label is active inside
   `[start, end)` `<= 0.05` (one 50 ms tolerance for diarizer resolution, not a
   tuned value).
4. **Overlap in the guarded interval.** Seconds where two or more distinct labels are
   simultaneously active anywhere in the guard band `<= 0.05`.
5. **Competing speech in the guard band.** Seconds where a non-primary label is active
   in the guard band `<= 0.20`. A different person speaking alone immediately outside
   the cut is exactly the case that leaks into a 30 s window.

Simultaneous segments carrying the *same* label count once: that is one person, not
two.

**Boundary safety is deliberately not an acceptance criterion.** Diarization cannot
prove a cut misses a word: a boundary inside an active turn means somebody is speaking
across the cut, which is a risk, not a guarantee. Distance from each boundary to the
nearest turn edge is *measured and reported* as `boundary_clearance`, and a word-safe
collar would need a VAD or acoustic check that this pilot does not run.

## Frozen reporting

The census emits a content-free aggregate: accepted rows and speech-hours, each
criterion as an independent flag **and** an ordered mutually-exclusive rejection
waterfall (an utterance can fail several at once, so the two views never reconcile by
accident), meeting-clustered intervals, per city / per source (`correction` vs
`no_edit`) breakdowns, the dominant-speaker margin distribution, `boundary_clearance`
distribution, and the denominator of failures: missing jobs, schema errors, utterances
outside the recording.

## Same-speaker runs and packing

A run of consecutive accepted utterances may be packed only while all of these hold; it
breaks otherwise:

- same audio file and same diarization job;
- same primary speaker label;
- chronological, non-overlapping database intervals;
- no rejected or missing utterance in between;
- inter-utterance gap `<= 2.0 s`.

Packs target **20–25 s of speech**, and the separator counts toward the waveform: with
0.4 s of unlabelled silence between utterances the waveform must still fit the 30 s
Whisper window, so the packer bounds waveform duration explicitly. The separator is
audio silence with a short fade, and it is *not* labelled in the target text.

## What the pilot may and may not decide

The first meetings are an **engineering smoke test**: pipeline, timebase mapping, cost,
failure modes. Their surviving fraction must not be used to decide whether to diarize
the corpus — four meetings are four clustered observations, and cached-audio
availability is not random.

A yield decision needs a stratified sample of at least one meeting per city, reported
with meeting-clustered uncertainty, and the gate is stated in hours of packable clean
speech rather than an appealing percentage.

## Standing risks, recorded now

- Training only on clean single-speaker monologue can weaken the model exactly where
  inference is hardest — interruptions and overlap. Any screen must carry a
  non-inferiority slice on overlap-heavy windows; a total-WER win that loses there is
  not a win. Rejected clips are kept and recorded for a protected/interruption lane.
- The eventual screen's control must be **density-matched**: the same packer, the same
  supervised seconds, the same number of windows and joins. Otherwise it measures
  packing, which `exp-2026-08-19-dense-screen-300` already measured and stopped.
- Diarization is not ground truth. Before any GPU spend, a small outcome-blind human
  audit of accepted and rejected clips checks only four things: overlap present, other
  speaker present, primary speaker dominant, boundary clipped.
