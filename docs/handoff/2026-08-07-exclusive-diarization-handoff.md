# Handoff: enable pyannoteAI exclusive diarization in opencouncil-tasks

> **ΞΕΠΕΡΑΣΜΕΝΟ 2026-08-21.** Το `exclusive: true` βρίσκεται **ήδη** στο σώμα
> του αιτήματος `/identify` στο `opencouncil-tasks/src/lib/PyannoteDiarize.ts`
> upstream, με σχόλιο που παραπέμπει στη σύσταση της pyannote για συμφιλίωση με
> την έξοδο STT. Το issue και το PR που προτείνει αυτό το έγγραφο **δεν
> χρειάζονται**. Το `model` εξακολουθεί να μην περνιέται (άρα precision-2 by
> default). Επαληθεύτηκε απευθείας στην πηγή, όχι μόνο μέσω ευρετηρίου.

For an agent that will open an issue and a PR on `schemalabz/opencouncil-tasks`.
Written 2026-08-07 from the OpenCouncil fine-tuning project's measurements.

> **Gated (2026-08-07):** do not open the issue/PR until the validation
> experiment in
> [2026-08-07-exclusive-diarization-experiment-handoff.md](2026-08-07-exclusive-diarization-experiment-handoff.md)
> has run and **both its gates passed**. The experiment's report supplies the
> numbers the issue must cite.

## The change, in one sentence

Add `exclusive: true` to the pyannoteAI `/identify` request and use the returned
`exclusiveDiarization` timeline for word→speaker assignment, keeping the existing
`diarization` timeline as an overlap signal — so overlapping speech stops forcing
the drift-cost guess and stops dropping utterances.

## Current production behaviour (verified via DeepWiki, 2026-08-07)

- `src/lib/PyannoteDiarize.ts` posts to the pyannoteAI cloud API `/identify`
  endpoint with body `{ url, voiceprints, webhook }` — no `model`, no `exclusive`.
- The API's `model` default is `precision-2` (the only option on `/identify`), so
  production already runs Precision-2 implicitly. `exclusive` defaults to `false`.
- ASR (ElevenLabs Scribe) runs with `diarize: "false"`; speaker assignment happens
  downstream in `src/tasks/applyDiarization.ts` →
  `DiarizationManager.findBestSpeakerForUtterance` (`src/lib/DiarizationManager.ts`):
  - exactly one overlapping diarization segment → direct assignment, drift 0;
  - multiple overlapping segments (overlap or fragmentation) → per-word "drift
    cost" heuristic picks one speaker for the whole utterance;
  - no speaker covers any word → the utterance is **dropped** (logged "SKIPPING",
    collected in `skippedUtterances`).

The failure mode this handoff targets: wherever two diarization segments cover the
same time span, the code must guess or drop. That is exactly where speaker
attribution and utterance boundaries go wrong in the published transcripts.

## What `exclusive: true` does (docs.pyannote.ai, /identify reference)

- Adds an **additional** response key `exclusiveDiarization`: same timeline but
  with overlaps resolved so that at any instant exactly one speaker is active.
- The normal `diarization` key (with overlaps) is **still returned** — nothing is
  lost; you get both views from the same call.
- Precision-2 only (which production already uses). pyannoteAI documents the mode
  as intended for STT reconciliation, i.e. precisely this word→speaker merge.
- Default `false`, so today production receives only the overlapping timeline.

Unknown / to note in the PR: the docs do not specify *how* overlap is resolved
(dominant speaker per frame is the likely behaviour). In this corpus the second
voice is usually a ~0.5 s off-mic interjection, so absorption into the dominant
speaker is the desired outcome anyway.

## Evidence from the fine-tuning project (cite in the issue)

All from `angelospk`'s GSoC fine-tuning repo measurements (aggregate numbers only —
do not copy utterance text anywhere public):

- Overlap prevalence over 232 two-minute benchmark windows (10 cities): **2.2% of
  speech time mean, 0.3% median**, 43% of windows zero. Median overlap event
  **0.52 s**; mostly off-mic room voices, not two miked speakers.
  (report: `docs/reports/2026-08-03-overlap-screen.md`)
- Blinded human audit of the overlap detector: **40/40 detected events were real**,
  zero false positives — the detection side is trustworthy.
- Causal cost of overlap on WER: **0.16 points** at natural prevalence (synthetic
  paired experiment, `docs/reports/2026-08-03-synthetic-overlap.md`). So this
  change is a transcript-quality / speaker-attribution fix, **not** a WER fix —
  the issue should not promise accuracy gains.
- In a 13-case human-adjudicated disagreement audit, Precision-2 matched the human
  in 10/13 vs community-1's 3/13 (suggestive, p≈0.09) — supports relying on the
  Precision-2 timeline more, not less.

## Scope of the PR (keep it small)

1. `src/lib/PyannoteDiarize.ts`: add `exclusive: true` to the `/identify` request
   body. Consider also passing `model: "precision-2"` explicitly so the behaviour
   is pinned rather than default-dependent.
2. Plumb `exclusiveDiarization` through the webhook/response handling next to the
   existing `diarization` (check `combineDiarizations()` — segment-offset logic
   must apply to both timelines identically).
3. `src/lib/DiarizationManager.ts`: prefer the exclusive timeline for
   `findBestSpeakerForUtterance`. With non-overlapping segments the "multiple
   overlapping segments" branch and most "SKIPPING" drops should become rare;
   keep the drift heuristic as fallback (fragmentation can still occur).
4. Keep the original `diarization` timeline available downstream as an overlap
   marker (future UI flag "two speakers here"); do not remove it.
5. Fallback: if `exclusiveDiarization` is absent in a response (older jobs, API
   hiccup), behave exactly as today.

Non-goals: no speech separation, no transcribing the interjector, no utterance
splitting redesign, no change to Scribe parameters.

## Suggested validation

- Unit: fixture response containing both keys with a synthetic overlap; assert
  assignment uses the exclusive timeline and that a formerly-skipped utterance now
  gets a speaker.
- Integration/manual: re-run `applyDiarization` on one meeting known to have
  crosstalk; compare `skippedUtterances` count and speaker assignments before vs
  after. Expect fewer skips and no regressions on non-overlapping stretches.
- Confirm API cost is unchanged (same call; response is slightly larger).

## Deliverables for the agent

1. **Issue** on `schemalabz/opencouncil-tasks`: describe the current guess/drop
   behaviour in `DiarizationManager`, what `exclusive: true` provides, the
   evidence above (aggregates only), and the proposed scope. Honest framing:
   quality-of-attribution fix affecting ~2% of speech time, near-zero cost.
2. **PR** implementing the scope above, linked to the issue, with the validation.
3. Both texts are public-facing → run the `humanizer` skill on the drafts before
   posting (per the user's global instruction).

## Verification the agent should redo before coding

The code quotes above came from DeepWiki (2026-08-07), not from reading the repo
directly. Before writing the PR: clone/read the actual current files
(`PyannoteDiarize.ts`, `DiarizationManager.ts`, `applyDiarization.ts`,
`ScribeTranscribe.ts`) and confirm endpoint, body shape, and merge logic. Also
re-check the `/identify` docs page (https://docs.pyannote.ai/api-reference/identify)
for the exact response schema of `exclusiveDiarization`.
