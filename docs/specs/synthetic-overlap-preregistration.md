# Preregistration: paired synthetic-overlap intervention

Frozen 2026-08-03, **before any mixture was transcribed**. Codex reviewed the draft at
high effort and the corrections it forced are marked `[codex]`. Nothing below may be
changed once the first result is looked at; changes go in a dated amendment section.

Scripts: `eval/controlled_eval/synth_overlap_build.py`, `_run.py`, `_analyze.py`.

## Why

The [overlap screen](../reports/2026-08-03-overlap-screen.md) established that
pyannote-estimated overlap is a strong error *marker* for all seven ASR systems, and was
explicit that a marker is not a burden: overlap is not randomly assigned, and crosstalk
windows are also noisier, faster and busier. This experiment manipulates overlap directly
so the contrast is causal for the manipulation performed.

It exists to gate an expensive decision: whether to train a diarization-conditioned
Whisper (DiCoW). That gate is written down here, in advance, with numbers.

## Estimand

For a fixed target window, donor waveform, placement and SIR, the paired difference

> WER(mixed) − WER(clean)

where both arms are scored against the **same unchanged human main-speaker reference**.

This is the effect of *this additive-speech intervention*, not the effect of natural
crosstalk. `[codex]` The write-up must use that phrasing.

Aggregation is the word-weighted paired error-count difference over total reference
words (`scoring.cluster_bootstrap`), not the mean of per-window WERs, which would
overweight short windows. `[codex]`

## Items

Benchmark windows from `2026-06-10-oc-benchmark` that have local audio **and** zero
pyannote-detected overlap (the "none" bucket, 100 of 232). Zero *detected* overlap is not
verified-zero overlap; the blinded listening audit now with the user includes 20 controls
drawn from exactly this bucket, and its result bounds the contamination. `[codex]`

## Intervention

**One** event per window, not three. `[codex]` Three 1.5–3 s events in a two-minute window
would impose 4–8% overlap, roughly triple the 2.2% actually observed, and repeating one
donor phrase can trigger Whisper repetition behaviour that natural crosstalk would not.
One event of ~2.5 s is ~2% of the window, which means the measured burden reads directly
as the burden *at natural prevalence* — which is the number the deployment decision needs.

Donors: 1.5–3.0 s single-speaker excerpts mined from zero-overlap windows in a
**different city** from the target. Assigned by SHA-256 of `(item_id, arm)` — a stable
documented hash, not a runtime hash. `[codex]` No donor is reused within an item, and
donor-meeting is carried into the analysis so dependence through donor reuse is visible.

Placement: inside a target-speech-active region (so it genuinely overlaps rather than
filling silence), located by an RMS VAD with hysteresis at a documented threshold
relative to the window's own active level. Not pyannote — the diarization output does not
retain segment boundaries, and this only has to answer "is the main speaker talking here".

## Arms

| arm | content |
|---|---|
| A | clean, round-tripped through the identical decode/write path |
| B | donor at SIR +15 dB (interjector well below target) |
| **C** | **donor at SIR +5 dB — the single primary condition** `[codex]` |
| D | donor at SIR 0 dB (equal level) |
| E | envelope-modulated speech-shaped noise from the same donor, +5 dB |
| F | the same donor reversed, +5 dB |
| G, H | clean at ±3 dB, fine-tune only — pipeline gain-sensitivity control |

B/C/D minus A would be three primary hypotheses. C is the primary; B and D are a
prespecified dose-response check. `[codex]`

E is the speech-specific control. A time-reversed voice is **not** a non-speech control —
it keeps pitch, formants and speech modulation — so it is carried as F, a separate
"speech-like acoustics without linguistic content" arm, and E (envelope-matched
speech-shaped noise) is the energy control. `[codex]` Neither isolates "voice"
metaphysically; both are described operationally.

## Level handling

`[codex]` No per-arm normalisation of any kind. Per-arm loudness matching would change
the target's own level between arms and confound the contrast.

1. decode every source to the same 16 kHz mono float PCM;
2. measure the target's **local active-speech** RMS at the placement, not whole-file;
3. scale the donor to the requested local active-speech SIR;
4. sum in float64;
5. take the maximum headroom needed across **all arms of that item**;
6. apply that one common attenuation to every arm of the item, clean included;
7. write all arms identically, no limiter.

QC gates, enforced in code, item fails and is dropped if violated:

- no clipped samples, true peak below −1 dBFS;
- achieved active-speech SIR within ±0.25 dB of requested;
- target samples bit-identical across arms up to the common item gain.

## Endpoints

- **Primary (policy):** full WER against the unchanged main-speaker reference. Under a
  target-only output policy, a correctly recognised interjector word *is* unwanted
  output, so penalising it as an insertion is the right accounting.
- **Secondary (donor-aware bound):** hypothesis tokens matching the donor's own
  transcript (obtained by transcribing the donor clip in isolation) discounted before
  scoring. This is an optimistic bound, not ground truth — simultaneous speech has no
  unique serialisation and both speakers can say the same word.
- **Secondary:** S/D/I decomposition, described only as an alignment decomposition.
  `[codex]` Plain deletions+substitutions is **not** a source-specific measure and must
  never be called "main-speaker damage": a donor word can be absorbed as a substitution,
  and a donor word matching an omitted target word can erase a deletion.
- **Regional:** WER in a fixed window around the event, and in the region *after* it, to
  separate recognition-under-overlap from Whisper segmentation/context spillover.

## Systems

whisper-large-v3 base and our fine-tuned council model, faster-whisper/CTranslate2, one
RunPod community GPU. Both are Whisper-family, which is the family DiCoW would be built
on. Decoder settings frozen and identical across arms: greedy, `beam_size=1`,
`temperature=0`, `condition_on_previous_text=False`, VAD off, language `el`. Processing
order randomised; no state carried between files.

Commercial providers are excluded on cost. The result therefore does not generalise
beyond these two systems. `[codex]`

## Decision gate — DiCoW

Frozen before results. `[codex]`

- **Go** requires total burden `C − A` ≥ **2.0** absolute WER points on the fine-tuned
  model, **and** speech-specific excess `C − E` ≥ **1.0** point with a one-sided 90%
  lower bound above zero.
- No serious qualitative contradiction at 0 dB. Strict monotonicity is not required.
- Even on a pass, DiCoW is not trained until a cheap frozen pretrained separation or
  target-speaker front end recovers ≥25% of the added errors on held-out mixtures.

A large `C − A` with a negligible `C − E` means generic energetic masking, for which
enhancement is the cheaper answer than diarization conditioning.

These are engineering thresholds, not constants. The economic form is
`P(R·B > M) > 0.8` with `B` the deployment-weighted recoverable burden, `R` the fraction
DiCoW recovers and `M` the minimum production gain worth the training and maintenance —
and `B` cannot be estimated credibly until natural overlap prevalence and SIR are
human-validated, which pyannote's 2.2% does not do.

## What this will not license

`[codex]` Even a clean positive result does not say that natural crosstalk explains the
observational association; that pyannote is accurate on Greek council audio; that DiCoW
will recover the measured burden; that Soniox's insertions are hallucination or correct
transcription; that anything generalises past two Whisper-family systems; that a short
additive interjection represents a prolonged argument or reverberant multi-speaker
overlap; or that the intervention improves production WER at natural exposure rates.

## Execution order

1. QC/variance batch of 25 items, blinded to the endpoints — pipeline correctness and
   variance only. Generation or scoring bugs fixed only through the QC rules above; the
   gate does not move. `[codex]`
2. Full paired run.
3. Recoverability test with a frozen separator, only if the gate passes.
4. DiCoW pilot, only if 3 shows headroom.

## Known limits accepted in advance

Cluster count, not window count, is what the CIs rest on; the number of independent
target meetings is reported alongside every interval, and if it falls under ~20 the
bootstrap is reported as unstable with a meeting-level aggregate as the fallback. `[codex]`
Donors carry their own room, microphone and codec, so "same recording conditions" is an
approximation; donor city is balanced but channel quality is not measured.

## Amendment 2026-08-03: short-event sensitivity arm

Added **after** the precision-2 corpus pass and **before** any mixture was transcribed.
It changes no primary endpoint and does not move the gate.

precision-2's event geometry over 232 windows: 1,123 events, median duration 0.52 s, p75
1.0 s, p95 2.2 s. The preregistered dose of uniform(1.5, 3.0) s therefore sits above the
90th percentile of real events. It stays the primary condition — the frozen plan
explicitly refuses to re-derive a causal dose from a detector that has a known blind spot
— but it is now understood as an upper extreme, not a typical case.

New arm **C_short**: identical to C (+5 dB SIR) with donor duration drawn from
uniform(0.4, 0.7) s, the empirical interquartile region around the median. Secondary,
descriptive, reported as dose-response alongside B/C/D. If the burden at the real median
event length is negligible while C's is large, then the measured burden is a statement
about rare long interjections, and the deployment-weighted number is much smaller than C
alone would suggest.
