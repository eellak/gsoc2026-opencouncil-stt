# Preregistration: does cutting the audio at speaker changes help?

Frozen 2026-08-03, before any decode. Design reviewed by Codex at high effort; the arm
that exists only because of that review is marked `[codex]` and it is the arm that makes
the experiment mean anything.

Scripts: `eval/controlled_eval/exp_segmentation.py` (build + run), `_analyze.py`.

## Why this and not the others

Four things were on the table: diarization-aware segmentation, source separation, ASR
fusion, and who-said-what mapping. This one goes first because it is the only one that
acts directly on the **strongest measured signal** and can be judged with the metric we
actually have.

The [corpus analysis](../reports/2026-08-03-overlap-screen.md#7) found speaker-turn
density beats overlap as a predictor of window error, and that overlap adds nothing once
turn density is known. Turn density is available at inference time for free. So the
question is no longer "is overlap harmful" — that is the synthetic experiment's job — but
**"does knowing where the speakers change let us decode better".**

The other three are deferred for stated reasons. Separation with
`speech-separation-ami-1.0` is trained on English office meetings at a distant mic; a
negative result would say nothing about separation in general. Two-system fusion has no
tie-breaker and its real baseline is the three-way vote that already wins 1.1 points.
Who-said-what cannot be scored at all until a human speaker-attributed reference exists,
because the published transcript's speaker boundaries are themselves machine output — the
text was human-corrected, the speaker assignment was not.

## Arms

Same audio, same checkpoint, same decoder settings, same normalisation. 232 windows.

| arm | segmentation |
|---|---|
| **1. baseline** | the whole two-minute window, one decode, as today |
| **2. turn-aware** | cut near precision-2 speaker changes, ±0.75 s context padding, overlapping text de-duplicated at the seams |
| **3. shifted control** | the same **number** of chunks as arm 2, boundaries placed as far from every speaker change as the length limits allow | `[codex]`

Arm 3 is not optional. `[codex]` Without it, any gain in arm 2 is equally explained by
"the chunks got shorter", which changes Whisper's context window, its temperature fallback
behaviour and its hallucination rate. The identifying contrast is **2 − 3**, not 2 − 1.

Chunk construction, frozen: speaker changes from precision-2's turns, minimum turn 0.25 s,
adjacent changes closer than 2 s collapsed to one boundary, no chunk shorter than 5 s or
longer than 30 s. Arm 3 keeps arm 2's chunk **count** and the same 5-30 s legal range, and places its
boundaries to maximise the minimum distance to any speaker change (binary search on that
margin, feasibility by dynamic programming on a 0.25 s grid). Same padding, same
de-duplication.

The first design held arm 2's chunk lengths exactly and searched a global offset. Measured
result: 182 of 232 windows ended with a margin of zero, because six boundaries in two
minutes cannot all dodge the changes. A control that also cuts at speaker changes controls
nothing, so the constraint was moved from lengths to count. Achieved margin is now a
median of **9.2 s**, with 231 of 232 windows clear of 1.0 s, and the per-window margin is
recorded so a weak control is visible rather than hidden. Median chunk length is 21.7 s in
arm 2 against 24.2 s in arm 3 — close but not identical, and that residual difference is
the honest cost of the change.

De-duplication at seams: overlapping padded regions produce repeated words. The seam
merge takes the longest common word sequence between the tail of one chunk and the head of
the next and drops one copy. Identical in arms 2 and 3.

## Systems

Our fine-tuned council model as primary, whisper-large-v3 as a check. faster-whisper on
one RunPod community GPU. Greedy, `beam_size=1`, `temperature=0`,
`condition_on_previous_text=False`, VAD off, language `el`. Decode order shuffled with a
fixed seed so runtime drift cannot align with an arm.

## Endpoints

**Primary:** reference-word-weighted WER over all 232 windows, meeting-clustered
bootstrap, on the **2 − 3** contrast for the fine-tune.

**Prespecified stratum:** the top tercile of turn density, which is where a routed
deployment would actually apply this. Reported with the low-density stratum alongside, so
a routing regression cannot hide.

**Secondary:** 2 − 1 and 3 − 1, to say how much of any gain is chunking rather than
placement. Per-city and leave-one-city-out.

## Decision rule

Frozen. `[codex]`

- **Continue** if 2 − 3 improves by ≥ **1.0** absolute WER point overall, or ≥ **2.0**
  points in the top turn-density tercile, with no material regression in the bottom
  tercile.
- **Stop** if the confidence interval rules out both thresholds.
- An interval that excludes neither is **inconclusive**, which is a sample-size result and
  not a success. It does not license shipping.

The 232 windows are the same ones that produced the turn-density discovery, so this is a
confirmatory pilot on the data that generated the hypothesis, not external validation. If
it passes, it has to be repeated on meetings this project has never scored.

## What a positive result would not license

That turn density is causal. That the same gain survives on unseen meetings. That
speaker attribution improved — this experiment never assigns a word to a speaker and
cannot say anything about who-said-what. That overlap-targeted training is justified;
that remains the synthetic experiment's separate question.
