# Preregistration: does pyannoteAI `exclusive` diarization improve speaker attribution

Frozen 2026-08-07, **before any Phase 1 job was submitted**. Phase 0 (a three-item
schema probe — no metrics, no gates) ran first; its findings are in § Phase 0 and
they forced three changes to the design, marked `[phase0]`. Codex reviewed the draft
at high effort and the corrections it forced are marked `[codex]`; where its
recommendation was not taken, the rebuttal is written next to it. Nothing below may
change once a Phase 1 result is looked at; changes go in a dated amendment section.

Scripts: `eval/controlled_eval/exclusive_diar_api.py`, `_probe.py`, `_run.py`,
`_analyze.py`, `oc_merge_port.py`, `build_exclusive_audit.py`.
Execution handoff: [`docs/handoff/2026-08-07-exclusive-diarization-experiment-handoff.md`](../handoff/2026-08-07-exclusive-diarization-experiment-handoff.md).

## Why

Production (`schemalabz/opencouncil-tasks`) calls pyannoteAI without `exclusive`, so
it receives a timeline where two speakers can be active at the same instant. Speaker
assignment happens downstream in `src/lib/DiarizationManager.ts`
`findBestSpeakerForUtterance`: exactly one diarization segment overlapping the
utterance envelope → direct assignment, drift 0; otherwise a per-word "drift cost"
picks among the speakers that fully cover at least one word; if no speaker fully
covers any word → `null`, and `applyDiarization.ts` **drops the utterance**
("SKIPPING").

`exclusive: true` returns an extra `exclusiveDiarization` timeline with exactly one
speaker active at any instant. If that timeline is well-formed, the multi-segment
guess branch and the drop branch should both get rarer.

This is an **attribution and utterance-integrity** experiment. It is not a WER
experiment. The [overlap screen](../reports/2026-08-03-overlap-screen.md) measured
overlap in this corpus at 2.2% of speech time (median 0.3%, median event 0.52 s) and
the causal WER cost at 0.16 points. No WER gain is predicted, promised, or measured
here, and the report may not claim one.

## Scope limits accepted in advance

- **Endpoint.** `[codex]` Production calls `/identify` with voiceprints; this
  experiment calls `/diarize`. Voiceprints for these speakers do not exist outside
  production and building them is out of scope, so the result is limited to
  `/diarize` with `model: "precision-2"` and must be reported that way. The two
  endpoints share the diarization backbone and both accept `exclusive`, which makes
  the transfer plausible, not demonstrated. Any issue text must say so.
- **ASR.** Phase 2 substitutes faster-whisper words for ElevenLabs Scribe words
  (below). Absolute drop and guess counts are therefore not production's counts.
- Both limits mean this experiment can only support a *paired, internal* comparison
  of two timelines under one fixed set of utterances. It cannot estimate the
  production-level effect size.

## Estimands

**Phase 1 (synthetic, ground truth by construction) — a feasibility screen.**
`[codex]` Phase 1 cannot license a production conclusion and no Phase 1 number may
appear in an issue as evidence of improvement. It exists to answer: does the
exclusive timeline behave sanely on exactly the overlap type this corpus has, and is
it worth spending Phase 2's money and the user's 30 minutes.

**Phase 2 (real windows) — the decision.** For the 25 highest-turn-density benchmark
windows, the paired difference between replaying production's own assignment over the
regular timeline versus the exclusive one, on identical utterances: dropped
utterances, guess-branch utterances, and utterances whose assigned speaker differs —
with the differing ones adjudicated blind by a human listener.

## Phase 0 findings (descriptive, ran before the freeze)

Three arm-C mixtures, each submitted twice (`{url, model}` and
`{url, model, exclusive: true}`), `eval/controlled_eval/exclusive_diar_probe.py`,
raw output in `~/.cache/oc-overlap/exclusive_probe.json`.

1. **Schema.** `output.exclusiveDiarization` is a list of `{speaker, start, end}` —
   the same shape as `output.diarization`. Both keys are returned by one call when
   `exclusive: true`. Nothing else changes.
2. **The regular timeline is unaffected by the flag** in 3/3 items: the
   `diarization` list from the exclusive call was equal, element for element, to the
   `diarization` list from the baseline call. Three items do not establish a
   guarantee `[codex]`, so Phase 1 still pays for paired calls (below) and reports
   the invariance rate rather than assuming it.
3. **Resolution behaviour is (a), absorption by deletion.** Where the regular
   timeline had a short interjector segment inside a longer main-speaker segment,
   the exclusive timeline **dropped the interjector segment entirely and left the
   main-speaker segment uncut**. Example, `win_argos_dec23_2025_581922`: regular has
   `SPEAKER_01 72.905–76.185`, `SPEAKER_02 73.025–74.525`, `SPEAKER_00
   74.525–74.545`, `SPEAKER_02 74.545–74.665`; exclusive has only `SPEAKER_01
   72.905–76.185`. Segment counts never increased in any item (46→46, 42→41, 52→45).
4. **No fragmentation outside the event** in 3/3: main-speaker segment counts
   outside the event ±1 s were identical (35/35, 18/18, 21/21). The Phase 0 stop
   condition therefore did not fire.
5. `[phase0]` **The "main speaker" cannot be defined as the globally
   longest-speaking speaker.** In `win_argos_aug14_2025_203808` the event interval
   was owned by `SPEAKER_02` in *both* timelines while the globally dominant speaker
   was `SPEAKER_01` — the windows contain several real speakers and the injection
   point is not always inside the globally dominant one's speech. The definition
   below is local to the event and taken from the clean arm.
6. `[phase0]` **Operational note, not a finding.** One job sat in `running` for over
   half an hour with `updatedAt` frozen 2 s after creation, when two jobs were
   submitted back to back against the same uploaded media. Jobs are therefore
   submitted sequentially, one media object each, with a 300 s poll timeout and one
   retry. A job that fails twice is a **failed item**, handled under § Failures.

## Phase 1 — synthetic mixtures, feasibility screen

### Items and calls

All 95 built items from `~/.cache/oc-overlap/synth_overlap_manifest.json`, arms **A**
(clean control) and **C** (donor interjection at +5 dB SIR — the primary arm of the
[synthetic-overlap prereg](synthetic-overlap-preregistration.md)).

`[codex]` Arm **D** is **omitted**. The original plan ran it "if C is ambiguous",
which is a researcher degree of freedom with no decision rule attached.

Calls per item: arm A with `exclusive: true` (1 call), arm C with `exclusive: true`
(1 call), arm C without the flag (1 call, the paired status-quo comparator). 285
jobs. Submission order is randomized with `random.Random(20260807)` over the flat
list of (item, arm, flag) triples `[codex]`, sequentially, one media object per job.

**Cost rule, frozen.** Cost is projected from the first ten completed jobs. If the
projection for 285 jobs exceeds $20, the arm-C unflagged comparator is restricted to
the **first 25 item ids in sorted order** (deterministic, no selection freedom) and
the invariance rate is reported over those 25. If the projection still exceeds $20,
Phase 1 stops and the report says so.

### Speaker-label mapping `[codex]`

Labels are per-job and arbitrary. Arm-A labels are mapped to arm-C labels by greedy
maximum time overlap computed **outside** the event interval widened by a ±0.5 s
guard, on the two *regular* timelines, largest overlap first, one-to-one. A mapping
pair with less than 1.0 s of overlap outside the guard is not made. If the arm-A
local event owner (below) has no mapped counterpart in arm C, the item is a
**mapping failure**: it stays in the denominator of metric 1 and is scored as a miss
(worst case) `[codex]`.

Within one job the exclusive and regular timelines carry the same labels, so no
mapping is needed between them — Phase 0 item 3 shows exclusive segments are a
subset of regular ones with identical labels. This assumption is checked in code per
item (every exclusive label must appear in that job's regular timeline); a violation
makes the item a mapping failure.

### Metrics (frozen)

1. **Main-speaker absorption rate.** `[codex]` Renamed from "dominant-speaker
   correctness", which was circular: the injected interval really does contain a
   second speaker, so attributing it to the main speaker is not acoustic
   ground-truth correctness. It is *alignment with the target-only transcription
   policy*: the human reference for these windows transcribes only the main speaker,
   so absorption is what keeps production from dropping or misattributing real
   main-speaker words.

   Definition: the **local main speaker** of an item is the speaker owning the most
   time inside `[event_start, event_start + event_dur]` in the **arm-A regular**
   timeline `[phase0]`. Absorption rate = the fraction of items where the arm-C
   *exclusive* timeline gives the majority of that same interval to the mapped local
   main speaker. Items where arm A has **no** speech at all in the event interval
   are excluded from this metric (the injection landed in silence, so there is no
   main speaker to absorb into) and their count is reported; all other items,
   including mapping failures and failed items, stay in the denominator.

   Reported alongside, as the paired comparator: the same rate for the arm-C
   *regular* timeline.

2. **Fragmentation.** Per item, on arm C: (a) local-main-speaker segment count in the
   exclusive timeline ÷ in the regular timeline; (b) the exclusive-timeline
   local-main-speaker segment count on arm C ÷ on arm A. Same-speaker segments
   separated by < 10 ms are merged first, so a pure boundary artefact is not counted.
   A zero denominator (no main-speaker segments at all) makes the item a failed item
   for this metric, counted and reported, not silently dropped `[codex]`. Reported:
   the median **and** the fraction of items above 1.2 `[codex]` — a median alone
   would let half the corpus degrade badly and still pass.

3. **Merge simulation.** Pseudo-utterances are a **fixed 5 s tiling** of the window
   (`[0,5), [5,10), …`, final partial tile kept if ≥ 1 s), each tile's "words" being
   five 1 s spans. Frozen choice: it needs no ASR, is identical across arms and
   variants by construction, and isolates timeline geometry from Whisper variance.
   It is **not** a model of real utterance boundaries — that is Phase 2's job, and no
   Phase 1 tiling number may be reported as a production count. Each tile goes
   through the ported `findBestSpeakerForUtterance` (§ Port) over the regular and
   over the exclusive timeline; counted per variant: direct branch, guess branch,
   drop branch.

### Phase 1 gate (frozen)

All four, on arm C:

- absorption rate ≥ **0.80**;
- absorption rate under exclusive ≥ absorption rate under regular `[codex]` — a
  comparative requirement, so Phase 1 cannot pass while being no better than the
  status quo;
- both fragmentation medians ≤ **1.2** **and** the fraction of items above 1.2 ≤
  **0.10** `[codex]`;
- merge simulation: guess-branch + drop-branch tiles under exclusive ≤ under regular.

Anything else → stop, negative report, no Phase 2, no issue, no PR. Failed items and
mapping failures count against the gate; they cannot be excluded after the fact
`[codex]`.

Uncertainty is reported as a cluster bootstrap over **target meeting**
(`meeting_id`), not item, using `eval/controlled_eval/scoring.py::cluster_bootstrap`
`[codex]` — items share target meetings and donors, so item-level intervals would be
too narrow. The gate is on the point estimates; the intervals are reported for
honesty about precision.

### Phase 1 stop conditions

- Projected API cost over ~$20 after the fallback above → stop.
- Failed items over 5% of jobs → stop; a broken run is not a negative result.

## Phase 2 — real windows, production-logic replay

### Windows

Top **25** by speaker-turn density (segments per minute of the precision-2 regular
timeline in `~/.cache/oc-overlap/precision2_corpus.json`) among the 232 benchmark
windows. Turn density, not overlap, because the overlap screen's M3 showed turn
density strictly dominates overlap as a marker of contested audio:
`U(turns | overlap) = +0.00127 [+0.00014, +0.00248]`, while `U(overlap | turns)` is
*negative*. Ties broken by item id ascending. The 25 ids are computed and written to
`~/.cache/oc-overlap/exclusive_phase2_windows.json` **before** any Phase 2 job is
submitted, and the file is hashed into the freeze record.

### Port

`findBestSpeakerForUtterance`, the drift cost and the skip rule are ported to
`eval/controlled_eval/oc_merge_port.py` from `schemalabz/opencouncil-tasks` at
commit **`5ff16a3c20968d6a5610d3584322b9a0059ad482`** `[codex]` — a pinned SHA, not
"the current version", so the oracle cannot move after results.

Parity is established **differentially against the real TypeScript**, not against
hand-computed expectations: a small `tsx` harness runs the pinned
`DiarizationManager` over the same fixtures and the port must match its branch,
selected speaker and drift value exactly. Fixtures: `[codex]`

1. **Boundary/direct** — words exactly touching segment start and end, one eligible
   segment; checks interval inclusivity and the drift-0 direct path.
2. **Competing/tie** — two overlapping speaker segments with asymmetric word
   timestamps, plus a second fixture engineered to produce equal drift; checks the
   winner, the drift value and deterministic tie behaviour.
3. **False coverage / drop** — utterance `[0,2]`, words `[0,0.4]` and `[1.6,2]`,
   one segment `[0.8,1.2]`: it overlaps the utterance envelope but covers no word,
   so the port must reproduce the TypeScript drop.
4. **Randomized differential** — 2,000 random (utterance, timeline) fixtures from
   `random.Random(20260807)`, port versus `tsx`, requiring exact agreement on branch
   and speaker and agreement on drift to 1e-9.

The replay does not run until all four pass. Any ambiguity in the TypeScript is
resolved in favour of the status-quo variant and listed in the report.

### ASR words — the accepted deviation

faster-whisper large-v3, word timestamps, greedy (`beam_size=1`, `temperature=0`,
`condition_on_previous_text=False`, VAD off, language `el`), CPU. Utterances built
with `eval/oc_inference_harness.py::_words_to_utterances` (pause ≥ 1.0 s / sentence
punctuation / 30 s cap), the builder the rest of this repo uses. One transcription
per window, shared by both variants, so the utterance set is identical by
construction.

### Counts (frozen)

Per variant, over the same utterances and the same port, reported as **paired**
outcomes `[codex]` rather than two marginal totals:

- regular drops / exclusive keeps (recoveries);
- regular keeps / exclusive drops (regressions);
- both drop; both keep;
- guess-branch membership, same 2×2 pairing;
- utterances whose assigned speaker differs between variants;
- the same breakdown per window.

### Human adjudication

Differing-speaker utterances only, plus every recovery and every regression. If more
than 50, sample 50 with `random.Random(20260807)` after sorting by
`(window_id, utterance_start)` — seed and sort order frozen here. Sampling is
stratified so recoveries and regressions are never dropped in favour of plain
disagreements.

Each item: the utterance clip ±2 s; the two candidate attributions with variant
labels hidden and A/B order randomized per item by the same seeded RNG; and
**reference anchors** `[codex]` — for each candidate speaker, two short clips drawn
from that speaker's segments elsewhere in the window, so "speaker 3" means something
to the listener. Answer options, frozen `[codex]`:

- **A is right** / **B is right**
- **both name the same voice** (no audible difference)
- **neither is right**
- **can't tell**

Determinate judgments = A-right + B-right. "Neither" and "both same" and "can't tell"
are reported and enter the frozen limits below; they are never silently dropped.

Answers land in a JSON file; the analysis script refuses to run before it exists. The
blinding key lives in a separate file the analysis reads only after the answers
exist, and no aggregate is computed or looked at before then. Single rater (the
user), which limits inference to one listener's judgment `[codex]`; a second rater is
out of budget and the report says so.

### Phase 2 gate (frozen)

`[codex]` A production-positive conclusion — i.e. opening the PR — requires **all**
of:

1. **Powered.** ≥ **40** determinate judgments. Below that the phase is a pilot: the
   report is written, no PR is opened, and no "improves attribution" claim is made.
   For context, distinguishing 2/3 from 1/2 at one-sided 5% with 80% power needs
   ~53 independent judgments, and window clustering pushes that higher; 40 with a
   clustered bound is an explicit, stated compromise with the 30-minute budget, not
   a claim of adequate power.
2. **Attribution.** Exclusive right in ≥ 2/3 of determinate judgments **and** the
   window-clustered one-sided 95% lower bound on that proportion above 0.5.
3. **Safety on drops.** Non-inferiority: the window-clustered one-sided 95% upper
   bound on (regressions − recoveries) per 100 utterances below **+0.5**. Recoveries
   are a hoped-for benefit and are reported, but the gate on drops is safety, not
   benefit — the change is motivated by attribution.
4. **Quality of the judgments.** "Neither is right" ≤ 20% and "can't tell" ≤ 40% of
   all adjudicated items. A high "can't tell" rate is itself a finding: it would mean
   the difference is inaudible and not worth production risk.
5. **Parity.** All four port-parity tests green against the pinned TypeScript.

Anything short of all five → report only. `[codex]` Auxiliary numbers — guess-branch
reduction, per-window breakdowns, Phase 1 tiling results — are descriptive and
**cannot rescue a failed gate**; the report must say which single number licenses
each claim: absorption rate licenses nothing about production, the adjudicated
proportion licenses "improves attribution", the paired drop bound licenses "does not
increase drops", and nothing here licenses any WER claim.

## Failures and exclusions `[codex]`

Defined before execution:

- **Failed item**: a job that does not reach `succeeded` after one retry, or a
  `succeeded` job whose `exclusiveDiarization` key is missing, empty, or contains a
  label absent from that job's `diarization`.
- Failed items stay in every denominator and are scored as the **worst case for the
  proposal** (no absorption, fragmentation counted as > 1.2, tiles counted as
  drops). They can never improve a result by disappearing.
- The only pre-declared exclusion is metric 1's "arm A has no speech in the event
  interval", which is a property of the stimulus, not of the outcome.

## Freeze record

Written to `~/.cache/oc-overlap/exclusive_freeze.json` before the first Phase 1
submission, and its hash quoted in the report `[codex]`: `opencouncil-tasks` commit
SHA; sha256 of every script named above; sha256 of the manifest and of the selected
window list; the exact API payloads; pyannoteAI model string; faster-whisper and
model versions; the RNG seeds; and this file's own sha256.

## What this will not license

Even a clean pass does not establish: that `exclusive` improves WER (not measured);
that it helps on `/identify` with voiceprints (not tested); that it improves
attribution on Scribe words (Whisper stood in); that it helps on the ~98% of speech
time with no overlap; that pyannoteAI's overlap detection is accurate on Greek
council audio (the overlap screen showed it is a weak marker); that the 25
highest-turn-density windows represent the corpus (they are chosen to be the worst
case, and the report says so); or that Phase 1's absorption behaviour on a
synthetic +5 dB interjection is how the model behaves on natural crosstalk, which has
different geometry, level and room acoustics.

## Amendment 2026-08-07 (a): adjudication covers disagreements only

Added while Phase 1 was still submitting jobs and **before any Phase 1 metric or any
Phase 2 output was computed**. It changes no gate.

The frozen text sent "every recovery and every regression" to the human alongside
the differing-speaker utterances. That cannot be asked blind: a recovery is a case
where one variant assigns a speaker and the other assigns none, so the two candidate
answers are not the same kind of object, the A/B presentation stops being symmetric,
and the item silently reveals which side is the proposal.

Adjudication is therefore **differing-speaker utterances only** — both variants
assign, they disagree about who. Recoveries and regressions need no listener: they
are counted objectively and gate 3 (drop non-inferiority) already rests on exactly
those counts. Gate 2 (attribution) rests on the adjudicated disagreements, as before.
Gate thresholds, seeds, sample cap and answer options are unchanged.

## Amendment 2026-08-07 (b): where the cost rule is evaluated

`GET /v1/jobs/{id}` does not return the `quantity` (billed seconds) field; only the
`GET /v2/jobs` listing does. The frozen ~$20 rule is therefore evaluated by
`eval/controlled_eval/exclusive_cost_check.py` against the listing rather than inside
the run loop. Threshold, arithmetic and consequence unchanged. Measured before the
Phase 1 run passed 25 jobs: mean 121 s/job, 285 jobs projected at **$1.18**, so the
comparator-subset fallback does not fire and the full 285-job design runs.

## Privacy and budget

- Council audio may go to the pyannoteAI cloud — the same exposure the benchmark
  already accepts for Scribe, Soniox and Gladia. The corrections corpus may not:
  nothing from `~/oc-approve-audit` is touched.
- No utterance text enters git. Raw API output stays under `~/.cache/oc-overlap/`;
  only scripts, this spec and aggregate reports are committed.
- API spend: hard cap ~$20, projected from the first ten jobs.
- Human time: ≤ 30 minutes, once, Phase 2 only.
- Production is never modified and nothing is pushed to `opencouncil-tasks` main.
