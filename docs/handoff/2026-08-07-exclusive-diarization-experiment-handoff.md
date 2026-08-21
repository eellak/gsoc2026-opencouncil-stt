# Handoff: validate pyannoteAI `exclusive` diarization before proposing it to production

> **ΞΕΠΕΡΑΣΜΕΝΟ 2026-08-21.** Το `exclusive: true` βρίσκεται **ήδη** στο σώμα
> του αιτήματος `/identify` στο `opencouncil-tasks/src/lib/PyannoteDiarize.ts`
> upstream, με σχόλιο που παραπέμπει στη σύσταση της pyannote για συμφιλίωση με
> την έξοδο STT. Το issue και το PR που προτείνει αυτό το έγγραφο **δεν
> χρειάζονται**. Το `model` εξακολουθεί να μην περνιέται (άρα precision-2 by
> default). Επαληθεύτηκε απευθείας στην πηγή, όχι μόνο μέσω ευρετηρίου.

For an agent executing the full plan end-to-end. Written 2026-08-07.

**Objective.** Measure, offline, whether `exclusive: true` on the pyannoteAI API
would improve speaker attribution and utterance splitting in the OpenCouncil
production pipeline (`schemalabz/opencouncil-tasks`). Only if the frozen gate
below passes, open the issue + PR described in
[2026-08-07-exclusive-diarization-handoff.md](2026-08-07-exclusive-diarization-handoff.md).
If the gate fails, write the negative report and stop — that is a successful
outcome too.

**Project culture you must follow.** This repo preregisters experiments: metrics
and gates are frozen in a spec *before* anything runs, and the report says what
the prereg promised it would say. Read two examples first:
`docs/specs/synthetic-overlap-preregistration.md` and
`docs/reports/2026-08-03-overlap-screen.md`. Your first deliverable is
`docs/specs/exclusive-diarization-preregistration.md` containing the design below
(adjusted only where reality forces it, with the change noted).

---

## Background you need (all already measured)

- Production calls pyannoteAI `/identify` with `{url, voiceprints, webhook}` —
  no `model` (so it gets the default, precision-2) and no `exclusive` (default
  false). Speaker assignment happens downstream in
  `src/lib/DiarizationManager.ts` `findBestSpeakerForUtterance`: one overlapping
  segment → direct assignment; several → per-word "drift cost" guess; none
  covering any word → the utterance is **dropped** ("SKIPPING"). Verified via
  DeepWiki 2026-08-07 — **re-verify by reading the actual repo before Phase 2.**
- `exclusive: true` adds an extra response key `exclusiveDiarization`: the same
  timeline with overlaps resolved so exactly one speaker is active at any
  instant. The regular `diarization` key is still returned. Precision-2 only.
  Docs: https://docs.pyannote.ai/api-reference/identify (and `/diarize`, which
  also accepts it and needs no voiceprints — use `/diarize` for this experiment).
- Overlap in this corpus: 2.2% of speech time mean (median 0.3%), median event
  0.52 s, mostly off-mic room voices. Causal WER cost 0.16 points. So this is a
  transcript-quality/attribution experiment, **not** a WER experiment — do not
  promise or measure WER gains.
- How overlap resolution *works* inside exclusive mode is undocumented. Finding
  that out empirically is Phase 0.

## Assets to reuse (all exist, do not rebuild)

| asset | path | what it gives you |
|---|---|---|
| API key | `~/.cache/oc-overlap/pyannote_api_key` (or env `PYANNOTE_API_KEY`) | pyannoteAI cloud access |
| API call pattern | `eval/controlled_eval/precision2_compare.py` | `upload()` (presigned PUT to `media://oc-overlap/...`), `diarize()` (POST `/diarize`), `wait_job()` (poll `/jobs/{id}`). **Use curl via subprocess, not urllib** — Cloudflare-fronted API, documented lesson. |
| 95 synthetic mixtures, ground truth by construction | `~/.cache/oc-overlap/mixtures/win_*__{A..F}.wav` + `~/.cache/oc-overlap/synth_overlap_manifest.json` | per item: `event_start_sec`, `event_dur_sec`, `achieved_sir_db` per arm (A=clean, C=+5 dB interjector, D=0 dB, …), window duration. You know exactly when the interjector speaks. |
| 232 real benchmark windows (~2 min each) | `~/.cache/oc-overlap/winwav/win_*.wav` | Phase 2 material |
| per-window overlap/turn features | `~/.cache/oc-overlap/overlap_features.json`, `precision2_corpus.json` | select high-turn-density windows for Phase 2 (inspect schema before use) |
| existing precision-2 turns for the 232 windows | `~/.cache/oc-overlap/precision2_corpus.json` | the **non-exclusive** baseline may already be here — check before paying for re-runs |
| production-mirroring utterance builder | `eval/oc_inference_harness.py` `_words_to_utterances()` (pause ≥ 1.0 s / sentence punctuation / 30 s cap) | Phase 2 replay |
| blinded audit page pattern | `eval/controlled_eval/build_overlap_audit.py`, `build_disagreement_audit.py` | Phase 2 human adjudication UI |

## Privacy constraints (hard rules)

- Council **audio** may go to the pyannoteAI cloud — the benchmark already
  accepts identical exposure for Scribe/Soniox/Gladia. The **corrections corpus
  must not** be used or uploaded anywhere. Nothing from `~/oc-approve-audit`.
- Nothing with utterance text enters git. Raw API outputs go under
  `~/.cache/oc-overlap/`; only scripts, the prereg spec, and aggregate reports
  are committed. Follow the existing `.gitignore` conventions.

---

## Phase 0 — what does exclusive mode actually return (minutes, ~$0)

Pick 2–3 arm-C mixtures. For each, submit **two** `/diarize` jobs: baseline
`{url, model: "precision-2"}` and `{url, model: "precision-2", exclusive: true}`.

Record: exact response schema of `exclusiveDiarization`; and around the known
`event_start_sec .. +event_dur_sec` interval, how the overlap was resolved —
(a) interjector absorbed into the main speaker, (b) main speaker's segment cut
with an interjector segment inserted, (c) something else.

**Stop condition:** if the exclusive timeline is pathological — the main
speaker's timeline fragments into materially more segments than baseline outside
the event, or the schema is unusable — stop, write it up, no further spend.

## Phase 1 — synthetic mixtures, objective scoring (no human time)

Run all 95 items, arms **A** (clean control) and **C** (+5 dB, the natural-
prevalence arm), both with `exclusive: true` (one call returns both timelines).
Optionally arm D (0 dB) if C is ambiguous. ~190–285 jobs on ~2-min windows;
estimate cost from the first ten before committing (the earlier 9.3 h corpus
pass was cheap; if projected cost exceeds ~$20, stop and report).

Frozen metrics (compute from timelines vs manifest ground truth):

1. **Dominant-speaker correctness:** fraction of injected events where the
   exclusive timeline attributes the event interval's majority to the main
   speaker (the desired absorption behaviour, given the reference transcribes
   only the main speaker).
2. **Fragmentation ratio:** count of main-speaker segments in the exclusive
   timeline ÷ count in the regular timeline, on arm C; and arm C ÷ arm A within
   exclusive. Both should be ≈ 1.
3. **Merge simulation:** implement the production assignment (Phase 2's port,
   below) and run it over both timelines: on arm C, how many synthetic
   "utterances" (build them with `_words_to_utterances()` over whisper word
   timestamps, or simpler fixed 5 s tiling — state the choice in the prereg)
   fall into the multi-segment/guess branch or the skip branch under the regular
   timeline vs the exclusive one.

**Phase 1 gate (frozen):** dominant-speaker correctness ≥ 0.80 AND fragmentation
ratios ≤ 1.2. Below that, exclusive mode does not behave as needed on exactly
the overlap type this corpus has → stop, negative report.

## Phase 2 — real windows, production-logic replay, bounded human adjudication

This phase produces the numbers for the issue. It is the decision-maker.

1. **Select windows:** top-25 by speaker-turn density from
   `overlap_features.json` (turn density, not overlap, is this corpus's marker
   of contested audio — see `docs/reports/2026-08-03-overlap-screen.md` §M3).
2. **Port the real merge logic:** clone `schemalabz/opencouncil-tasks`, read
   `src/lib/DiarizationManager.ts` and `src/tasks/applyDiarization.ts`, and port
   `findBestSpeakerForUtterance` (including the drift cost and the skip rule)
   faithfully to a local script. Unit-test the port against 2–3 hand-constructed
   cases so the replay is their algorithm, not an approximation.
3. **ASR words:** faster-whisper large-v3 with word timestamps on each window
   (CPU is fine at this scale), utterances built with `_words_to_utterances()`.
   This substitutes whisper words for Scribe words — an accepted deviation;
   state it in the prereg and the issue.
4. **Replay both variants:** assignment using the regular `diarization` timeline
   (status quo) vs the `exclusiveDiarization` timeline (proposed). Same
   utterances, same port.
5. **Count:** skipped utterances per variant; utterances landing in the
   guess branch; utterances whose assigned speaker **differs** between variants.
6. **Blinded human adjudication of the differences, capped:** if > 50 differing
   utterances, sample 50 (fixed seed 20260807). Cut a clip ±2 s around each,
   build a blinded page in the style of `build_disagreement_audit.py` — the
   human hears the clip and the two candidate segmentations/attributions with
   variant labels hidden and order randomized per item. The user (Greek speaker)
   judges "A right / B right / can't tell". Target ≤ 30 minutes of their time.
   You prepare everything; notify the user when it is ready; do not proceed to
   the verdict until the answers file exists.

**Phase 2 gate (frozen):** among determinate human judgments, exclusive is right
in ≥ 2/3 AND skipped-utterance count does not increase. Auxiliary (report,
don't gate): guess-branch reduction, per-window breakdown.

## Deliverables, in order

1. `docs/specs/exclusive-diarization-preregistration.md` — the above, frozen,
   **before** Phase 1 runs. Per the global workflow, send the spec through the
   codex bridge for plan review (`model_reasoning_effort=high`) and incorporate
   or rebut its notes; skip with a note if the bridge is down.
2. Scripts under `eval/controlled_eval/` (e.g. `exclusive_diar_run.py`,
   `exclusive_diar_analyze.py`, `build_exclusive_audit.py`), raw outputs under
   `~/.cache/oc-overlap/`.
3. `docs/reports/2026-08-XX-exclusive-diarization.md` — verdict against the
   gates, honest about what Phase 2's whisper-for-Scribe substitution does and
   does not show. Update `CURRENT.md`'s next-steps list with the outcome.
4. **Only if both gates pass:** execute
   [2026-08-07-exclusive-diarization-handoff.md](2026-08-07-exclusive-diarization-handoff.md)
   (issue + PR on `schemalabz/opencouncil-tasks`), now armed with these numbers
   in the issue text. Public texts go through the `humanizer` skill first.

## Budget and stop rules

- API spend: hard cap ~$20; project from the first ten jobs.
- Human time: ≤ 30 min, Phase 2 step 6 only, prepared so it is one sitting.
- Any stop condition firing → the report is the deliverable; no issue, no PR.
- Never modify production, never push to `opencouncil-tasks` main, never upload
  corrections-corpus audio.
