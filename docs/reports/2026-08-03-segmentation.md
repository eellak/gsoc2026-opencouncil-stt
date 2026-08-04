# Cutting at speaker changes does not help, and cutting at all costs a lot

2026-08-03. Script: `eval/controlled_eval/exp_segmentation.py`, analysis `_analyze.py` ·
raw: `results_segmentation.json` · 227 windows, 141 meetings, 10 cities,
whisper-large-v3 on one RTX A4500, 681 decodes, ~100 minutes, $0.32.

Preregistered in
[segmentation-experiment-preregistration.md](../specs/segmentation-experiment-preregistration.md)
with the decision rule, the thresholds and the analysis script all frozen before the
first decode.

## What was asked

The [corpus analysis](2026-08-03-overlap-screen.md#7-the-full-precision-2-pass-the-association-survives-the-variable-does-not) found speaker-turn density is a
better predictor of window error than overlap, and that overlap adds nothing once turn
density is known. Turn density is free at inference time. So: **does cutting the audio at
speaker changes let Whisper decode better?**

Three arms, same audio, same checkpoint, same decoder. Arm 1 the whole two-minute window.
Arm 2 cut near precision-2 speaker changes with 0.75 s padding and seam de-duplication.
Arm 3 the control — same chunk count, same padding, same de-duplication, boundaries placed
as far from every speaker change as the length limits allow (median margin 9.2 s).

## Result

| arm | WER |
|---|---|
| 1. whole window | **0.1607** |
| 2. cut at speaker changes | 0.2223 |
| 3. cut away from speaker changes | 0.2237 |

**Primary contrast, arm 2 − arm 3: −0.0015, CI [−0.0125, +0.0100].** The frozen verdict is
**inconclusive** — the interval covers neither the 1.0-point improvement that would mean
continue nor a clean exclusion of it.

By turn density, terciles frozen from the diarization:

| stratum | arm 2 − arm 3 |
|---|---|
| low (< 0.8 changes/min) | −0.0022 [−0.0220, +0.0174] |
| mid | +0.0176 [−0.0039, +0.0398] |
| **high (≥ 2.8 changes/min)** | **−0.0200 [−0.0414, +0.0023]** |

The high-density stratum moves 2.0 points in the predicted direction and its interval
misses zero by 0.2 points. That is the prespecified stratum and the prespecified sign, so
it is the one honest reason not to close the question — but it is not a pass, and the
rule was frozen precisely so that a near miss could not be talked into one.

## The 6-point gap is mostly ours, not Whisper's

Arm 1 beats both chunked arms by about 6 WER points. It would be easy and wrong to report
that as "chunking hurts Whisper".

The error decomposition says otherwise. Insertions go from 0.023 per reference word in arm
1 to 0.055 in arm 2 and 0.075 in arm 3, while total hypothesis length stays sane (0.97,
0.97 and 1.01 times the reference). Something is emitting extra words at a rate that
tracks the number of seams, and the obvious suspect is the seam handling: 0.75 s of
padding on both sides of every boundary means every chunk pair transcribes the same audio
twice, and the de-duplicator only removes a repeat when the tail of one chunk and the head
of the next are an **exact** word-sequence match. Whisper does not transcribe the same 0.75
seconds identically twice, so most seams leak.

With about six chunks per window, that is five leaky seams per window, and it is entirely
sufficient to explain the gap. So arm 1 versus the chunked arms is **confounded by our own
stitching** and is reported here as a pipeline artifact, not a finding about Whisper.

This is exactly why the control arm exists. Arms 2 and 3 share the same padding, the same
de-duplicator and the same chunk count, so whatever the seams cost, they cost it equally.
The 2 − 3 contrast is unaffected by the bug, which is the only reason this run produced a
usable number at all.

## What this changes

The cheap version of the idea does not work. Handing Whisper the same audio cut at speaker
changes instead of arbitrary points is worth **nothing measurable** overall, and the one
place it might be worth something — the busiest third of windows — needs a bigger sample
to say so.

Before anyone spends that sample, the seam handling has to be fixed, because a pipeline
that loses 6 points to its own plumbing cannot demonstrate a 2-point gain in a stratum.
The obvious repair is to cut without padding and let the chunk boundaries be the seams, or
to align the overlapping regions acoustically rather than by exact string match.

What it does **not** change: turn density remains the better error predictor. This
experiment tested one cheap way of acting on that signal, and that way did not pay. It says
nothing about conditioning a model on speaker activity, which is the DiCoW question and
remains open.

## Caveats

whisper-large-v3, not our fine-tune — the fine-tune's CTranslate2 build is on the mini-PC
and the pod's uplink measured 180 kB/s. The 2 − 3 contrast is within one system, so this
changes what the number generalises to, not whether it is identified.

227 of 232 windows; five meetings have no public audio URL for the pod to fetch.

These are the same 227 windows that produced the turn-density hypothesis, so even a pass
would have been a confirmatory pilot and not external validation.
