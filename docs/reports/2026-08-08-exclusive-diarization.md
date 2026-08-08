# pyannoteAI `exclusive` diarization: passes the screen, fails the production gate

Preregistered in [`docs/specs/exclusive-diarization-preregistration.md`](../specs/exclusive-diarization-preregistration.md),
frozen 2026-08-07 before any Phase 1 job was submitted. Freeze record
`~/.cache/oc-overlap/exclusive_freeze.json`, sha256
`27e5c2667516a41779ed9cabe4624dde44db2868ed1ca67f0b60217cb7ed6733`.

**Verdict: no issue, no PR.** Phase 1 passed all five conditions. Phase 2 failed the
drop-safety condition, decisively and on the objective counts alone — the one-sided
95% upper bound on net lost utterances is **+1.94 per 100**, where the frozen gate
required below **+0.5**. Under the preregistration that ends it: `exclusive: true`
is not proposed to `schemalabz/opencouncil-tasks`.

That is not the whole story, and the interesting part is *why* it fails.

## What exclusive mode actually does

Phase 0, three arm-C synthetic mixtures, each submitted with and without the flag.

`output.exclusiveDiarization` is a list of `{speaker, start, end}` — the same shape
as `output.diarization`, returned alongside it by a single call. Overlaps are
resolved by **deleting the shorter speaker's segment**, leaving the longer speaker's
segment uncut. In `win_argos_dec23_2025_581922` the regular timeline has
`SPEAKER_01 72.905–76.185` with `SPEAKER_02 73.025–74.525` inside it; the exclusive
timeline has only the `SPEAKER_01` span. Segment counts never rose (46→46, 42→41,
52→45) and main-speaker segments outside the event were untouched (35/35, 18/18,
21/21).

Deletion, not splicing. Everything below follows from that one fact.

## Phase 1 — synthetic screen: pass

95 items × {arm A exclusive, arm C exclusive, arm C unflagged} = 285 jobs, zero
failures, $1.18 of API spend against a $20 cap.

| condition | frozen threshold | result | |
|---|---|---|---|
| main-speaker absorption (arm C, exclusive) | ≥ 0.80 | **0.990** [0.968, 1.000] | pass |
| absorption beats the status quo | ≥ regular | 0.990 vs 0.958 | pass |
| fragmentation medians | ≤ 1.2 | 1.00 and 1.00 | pass |
| fraction of items above 1.2 | ≤ 0.10 | 0.011 and 0.011 | pass |
| merge simulation, guess + drop | ≤ regular | 211 vs 238 | pass |

CI is a cluster bootstrap over the 72 target meetings. One item failed on label
mapping and was scored worst-case, as preregistered; none were excluded.

**The regular timeline was byte-identical with and without the flag in 95/95 items.**
Phase 0 could only suggest that over three; it now holds across the corpus, which is
what makes the one-call-per-item design and the paired Phase 2 comparison legitimate.

The merge simulation already showed the shape of the problem, and it was visible
before any Phase 2 number existed: guesses fell **73 → 17** while drops rose
**165 → 194**. Net better, so the gate passed — but the trade was there from the
start.

## Phase 2 — real windows, production's own logic: fail

25 highest-turn-density benchmark windows, 718 Whisper utterances, replayed twice
through a port of `findBestSpeakerForUtterance` pinned to `opencouncil-tasks` commit
`5ff16a3c`. The port is checked **differentially against the real TypeScript** under
`tsx` — 2,009 fixtures, exact agreement on branch, speaker and drift, including the
`reduce`-without-initial-value behaviour and the `Set`-ordering tie-break. This is
their algorithm, not a description of it.

### The good number

**Guess-branch utterances: 190 → 44.** Three quarters of the cases where production
currently picks a speaker by drift heuristic instead of reading it off the timeline
simply stop being ambiguous. On the corpus's most contested audio, that is a large
effect and it is not in dispute — it is a deterministic property of the two
timelines, no human judgment involved.

### The number that ends it

| paired outcome | count |
|---|---|
| both variants keep the utterance | 696 |
| both drop | 11 |
| regular drops, exclusive keeps (recovery) | **1** |
| regular keeps, exclusive drops (regression) | **10** |

Net **+1.25 lost utterances per 100**, window-clustered one-sided 95% upper bound
**+1.94**. The gate required below +0.5. It fails by a factor of four, and the point
estimate alone is already over the threshold, so this is not a precision problem that
more windows would fix.

The mechanism is the one Phase 0 identified. `findBestSpeakerForUtterance` drops an
utterance when **no speaker fully covers any of its words**. Deleting the overlapped
speaker's segments removes coverage. Most of the time the survivor covers the words
anyway — 696 of 718 — but when the deleted segment was the one covering an
utterance's words, that utterance stops being attributable and
`applyDiarization.ts` discards it. Exclusive mode buys attribution certainty by
throwing away timeline, and production's skip rule turns the missing timeline into
missing transcript.

### Attribution quality

51 utterances got a different speaker under the two timelines; 50 were sampled per
the frozen seed, 49 built (one had no usable voice anchor). The blinded listening
package is prepared and served, and the result will be appended here.

**It cannot change the verdict** — gate 3 already failed on counts that need no
listener. What it decides is whether the guess-branch reduction is a real quality
gain, i.e. whether a *modified* proposal is worth anyone's time or whether this is a
dead end. That is worth 30 minutes, and the instrument is run exactly as
preregistered rather than re-cut now that the gate is known to fail.

## What this does and does not say

Licensed by these numbers: exclusive mode resolves overlap by deletion; it does not
fragment the timeline; it removes three quarters of the drift-heuristic guesses; and
**as a drop-in flag it costs about one utterance per hundred on contested audio.**

Not licensed, and not claimed anywhere above: any WER effect (not measured, and the
overlap screen put the causal WER cost of overlap at 0.16 points, so there was never
much to find); any behaviour on `/identify` with voiceprints, which is what
production actually calls — this ran on `/diarize`; any statement about Scribe words,
since faster-whisper stood in for them and the absolute drop count is therefore not
production's count; and anything about the ~98% of speech time with no overlap. The
25 windows were selected to be the worst case in the corpus, deliberately.

The Whisper-for-Scribe substitution deserves one more sentence, because it is the
main threat to the headline number: both variants saw the identical utterance set, so
the *paired* comparison is internally valid, but a different segmenter would produce
a different absolute rate of lost utterances. The direction — deletion costs
coverage, coverage loss becomes drops — is a property of production's skip rule and
does not depend on which ASR produced the words.

## If anyone wants to revisit this

The failure is not in exclusive mode; it is in the interaction between deletion and a
skip rule that treats "no covering segment" as "no utterance". Two changes would
decouple them, neither of which this experiment tested:

- consume **both** timelines — exclusive for attribution, regular as the coverage
  fallback when exclusive yields nothing;
- or relax the skip rule so an uncovered utterance keeps its nearest-speaker guess
  instead of being discarded.

The first is a small change in `applyDiarization.ts` and would, on these numbers,
keep the 190 → 44 guess reduction while removing the drop regression by
construction. It is a hypothesis, not a result: it has not been run.

## Artefacts

Scripts under `eval/controlled_eval/`: `exclusive_diar_api.py`, `_probe.py`,
`_run.py`, `_analyze.py`, `exclusive_cost_check.py`, `oc_merge_port.py`,
`oc_merge_oracle.mts`, `test_oc_merge_port.py`, `exclusive_phase2_asr.py`,
`_run.py`, `_replay.py`, `_analyze.py`, `build_exclusive_audit.py`,
`exclusive_freeze.py`. Aggregates in `results_exclusive_phase1.json` and
`results_exclusive_phase2.json`. Raw API output, timelines and transcripts stay under
`~/.cache/oc-overlap/` and never enter git.
