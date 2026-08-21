# Clean-pack 300-step screen

Frozen before creating the GPU pod and before any arm is trained. Data supply and
selection criteria: [`2026-08-19-overlap-clean-selection.md`](2026-08-19-overlap-clean-selection.md).

## Arms

- **A (control):** the incumbent recipe, unchanged — the current single-utterance
  training parquet that produced `artifact-adapter-fixed`.
- **B (treatment):** overlap-free contiguous packs. One continuous span of a single
  dominant speaker per example, boundaries placed acoustically, ~22 s of speech in a
  <=29 s span, target text is the meeting's own utterances in order.

Seeds and order: A13, B13, A29, B29, A47, B47, sequentially on one pod. Same base
model, same LoRA recipe and LR, effective batch 8, 300 optimizer steps, identical
decode stack. `PACK_ARM=pn` (no timestamp tokens), matching the previous screen.

## What this screen does and does not isolate

It answers **"is this better than what we train on today?"**. It does **not** isolate
cleanliness from density: B differs from A in overlap filtering, in window occupancy
(~22 s of speech per 30 s window against 3.55 s), and in example construction at the
same time. Any positive result is a result about the whole recipe, and the write-up must
say so. A separate arm would be needed to attribute the effect.

At 300 steps and effective batch 8 the run consumes 2,400 examples. B holds 2,476 packs,
so it makes roughly one pass with no example repeated; A is sampled as it always was.

## Endpoints and gates

Primary: paired WER on the 39 frozen validation windows, all six adapters decoded on one
GPU float16 faster-whisper stack with the frozen served config.

Promotion requires every gate in `docs/decisions/training-evidence.md`, unchanged and
frozen here: mean ΔWER < 0, at least 2/3 seeds negative, mean deletion delta <= 0,
insertion delta < +0.0005, no leave-one-window-out sign reversal, and no single window
above 25% of net gain. The insertion gate is not negotiable against a better total WER;
it is the gate the previous dense screen failed.

Diagnostic, never promoting: training WER on the frozen 300-row sample, reported for
both arms.

## Known gap: the overlap slice

An overlap-heavy non-inferiority slice is the right guard for an arm that trains only on
clean speech, and it cannot be built at full coverage now: pyannoteAI credits ran out on
2026-08-19. 21 of the 39 validation windows already carry diarization from
`exclusive_phase1`, so the slice is computed on those 21 and reported as **diagnostic
only**. A promotion to shipping needs the missing 18, which needs credits.

## Standing risks

- Training only on clean single-speaker speech may weaken the model exactly where
  inference is hardest. The 21-window slice is a partial check, not a guarantee.
- The supply covers 73 of 265 meetings, taken in download order rather than stratified
  by city. City balance is reported with the result; the surviving fraction estimates
  nothing about the corpus.
- Diarization is not ground truth. A human listened to 12 packs (12/12 single speaker,
  12/12 text matching audio, 2/12 with a slightly clipped first syllable, judged
  acceptable), which is a sanity check and not a validation.

## Cost and stop rule

RTX A4000 at $0.250/h. The previous 300-step paired screen took roughly 6-7 GPU-hours
for six runs, about $3 including setup. Hard billing deadline armed before upload, pod
ID recorded, pod terminated as soon as results are retrieved. If the same failure
happens twice, stop and write it down rather than retrying.

The seven temporal holdout windows stay sealed. No medium or full stage follows
automatically from a passing screen.
