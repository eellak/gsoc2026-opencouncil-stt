# Does shorter context or a speaker-count hint stop pyannote merging people?

Status: **CLOSED, negative, 2026-09-15.** Answer: no — re-diarizing topic slices
is worse than cutting the whole-meeting output at the same boundaries (mean paired
Δ = −0.0074, better in 0 of 4 meetings), and cutting alone already costs recall
0.9156 → 0.5705. Findings and limits:
[report](../../docs/reports/2026-09-15-diarization-clustering.md).
Reviewed by Codex Astra (`6d9715190ad149778f958c62ddc0f8f0`) before any job was
submitted; arms `a1`/`a2` were built and deliberately not run.

## The question

Upstream measured that switching to the exclusive timeline fixes only ~13% of
speaker misattribution, and that the dominant residual cause is **pyannote
merging several real people into one voice** — one meeting had a single voice
covering 324 utterances from 33 people
([opencouncil-tasks#15](https://github.com/schemalabz/opencouncil-tasks/issues/15)).

Harold's hypothesis: there is a practical ceiling near 30 speakers, so long
meetings force merging, and diarizing shorter pieces — cut per agenda topic —
would beat it.

## What the hypothesis can and cannot become

**"The ceiling is real" is not an available conclusion.** pyannote's own docs
state automatic detection has *no upper limit*, and no number of runs below a
threshold establishes one. Only falsification is available:

> Under automatic speaker counting, no A0 job returns ≥ 35 distinct clusters.

One job returning 35 kills it. Nothing confirms it.

Two documented mechanisms already explain the merging without any ceiling:
`maxSpeakers` set too low **forces** merges regardless of embedding distance,
and a minimum-cluster-size rule reassigns small clusters into the nearest large
one — which is exactly how a rare speaker disappears into the chair.

## Arms

| arm | request | information at inference | cost |
|---|---|---|---|
| `a0` | whole meeting, no hint | audio only — the deployable baseline | 6.1 h |
| `a1` | `numSpeakers` = reviewed people | audio + **privileged** count | 6.1 h |
| `a2` | `minSpeakers` = reviewed people | audio + **privileged** floor | 6.1 h |
| `a3` | each topic slice diarized alone | audio + slice rule | 6.1 h |
| `a0_local` | `a0` cut at the `a3` boundaries | none | **free** |
| `c3` | `a3` boundaries relabelled by `a0` identity | none | **free** |
| `b0` | `/identify` on the whole meeting with enrolled voiceprints | audio + enrolment | 6.1 h |
| `b3` | `/identify` per topic slice, same voiceprints | audio + enrolment + slice rule | 6.1 h |

**The b-arms exist because the a-arms could not separate two things.** Slicing
reduces how many people a piece contains, but it also reduces how much voice
each of them has in it — and pyannote clusters by embedding, so less voice means
a worse embedding. Voiceprints break that trade: identity comes from enrolment,
not from how much someone speaks inside the piece, and the same personId comes
back from every separate call, so **sliced arms need no relinking at all**.

`c3` answers the same question from the other side, for free: take the sliced
boundaries and relabel them using the whole-meeting run as the identity map.

**Trap, measured:** `/identify` does **not** return null for unmatched speech —
it puts the diarization label (`SPEAKER_00`) in the same `speaker` field as real
personIds. chania returned 13 distinct labels for 2 voiceprints sent. Anything
not among the voiceprints actually sent must be namespaced per slice, or two
slices' anonymous speakers silently merge into one person.

`a1` and `a2` are **reference-derived count hints, never deployable results**.
The reviewed-people count is not the acoustic speaker count: an attendee who
never speaks inflates it, an unnamed interjector deflates it. They measure
sensitivity to privileged information — if they help, the follow-up is a count
estimate obtainable without the answer key, e.g. from the roster.

`a0_local` is not optional. Independent slices give each slice its own label
space, so two people who never share a slice cannot contaminate one cluster:
impurity drops for a mechanical reason. `a3` is read **only** against
`a0_local`, never against `a0`.

## The threshold arm (`b3t`), declared before it ran

`matching.threshold` is a **minimum confidence score, 0–100, default 0**. At 0 the
call cannot abstain: every segment is handed an enrolled label however poor the match.
Below threshold, pyannote returns the segment with its diarization label and
`match: null` — which this code namespaces per slice, so rejections keep
*distinguishable* anonymous identities rather than collapsing into one "unknown"
that would manufacture merging.

**Declared value: `threshold = 50`.** It is the value pyannote's own documentation
uses in both of its identification examples ("higher values (50-70) more strict"), not
a value picked after seeing a result. One meeting, `athens/may18_2_2026` — the
worst-case arm, 43.2% of its speech from people with no voiceprint.

**Any threshold can also be replayed offline, for nothing.** The `voiceprints[]` block
of an `/identify` response carries, for every diarization speaker, its match and its
confidence against each enrolled print — so `simulate_threshold` logic needs no new
call. That is how the whole-meeting figures in the report were produced: 50 changes
nothing there (enrolled speakers already match at 77+), 70 cuts wrong names 26% at no
cost in correct ones, 85 starts costing. The simulation assumes the rejection rule is
exactly "best confidence below threshold", which a live call has not confirmed.

The output carries its threshold in the filename — `diarization_b3t50.json` — so a
later run at another value cannot silently reuse it. Score it by that name:

```bash
python3 eval/diarization_clustering/run_arms.py --arms b3t \
  --only athens/may18_2_2026 --threshold 50
python3 eval/diarization_clustering/score_naming.py \
  --meeting athens__may18_2_2026 --arms b0 b3 b3t50
```

Scored on four numbers **read together** (`score_naming.py`), because any one of them
can be gamed:

| number | why |
|---|---|
| meeting-wide B³ F | clustering quality |
| correct names over **all** turns | not over *named* turns — abstaining raises that for free |
| wrong names on unenrolled speakers | the specific damage a zero threshold causes |
| abstention rate | the price paid for the other two |

## Frozen parameters

Set before any number was looked at:

- **Slice rule**: contiguous runs of utterances sharing `discussionSubjectId`,
  merged forward until ≥ **300 s**; a trailing remainder is absorbed into the
  previous slice, never shipped as a runt.
- **Falsification threshold**: ≥ 35 clusters in one `a0` job.
- **Audio**: 16 kHz mono WAV for every slice, re-encoded (not `-c copy`, which
  shifts the start by up to one mp3 frame and desyncs the offsets).
- **Transcript**: human utterance boundaries, identical across all arms.
- **Cohort**: 6 meetings, all `humanReview: true`, all absent from
  `exclusion-manifest-2026-08`, all cached locally.

## What the slicing arm actually tests

Topic slices are **not** uniform. Measured on the cohort:

| meeting | duration | topic runs | slices at 300 s |
|---|---:|---:|---:|
| chania/jul27_2026 | 20.6 min | 33 | 3 |
| chalandri/may25_2026 | 15.8 min | 4 | **1 — null** |
| athens/may18_2026 | 42.7 min | 6 | **1 — null** |
| athens/may18_2_2026 | 72.0 min | 35 | 7 |
| athens/may28_2026 | 82.1 min | 45 | 9 |
| vrilissia/may27_2_2026 | 134.4 min | 47 | 10 |

**Two of six meetings produce a single slice**, because one agenda topic covers
almost the whole session (40.7 of 42.7 minutes in `athens/may18_2026`). For
those, `a3` is byte-identical to `a0` and is not run. The sliced arm therefore
rests on **four** meetings, which is what its confidence interval has to say.

## Gates

Merging and splitting are scored as a pair. A system that gives every utterance
its own voice scores perfect purity and is useless, so purity alone cannot be a
gate.

- **G1 (ceiling)**: falsified by one `a0` job at ≥ 35 clusters. Otherwise
  "unfalsified", never "established".
- **G2 (hint helps)**: `a1` must reduce merge error **and** keep the splitting
  increase inside the pre-set tolerance. Both, on the same run.
  **NOT EVALUATED** — `a1`/`a2` were dropped before running (see status above), so
  this gate has no result and nothing here says whether count hints help.
- **G3 (slicing helps)**: `a3` vs `a0_local` on the same local mapping and the
  same reference windows. Any gain is **within-slice attribution only** — it
  says nothing about meeting-wide attribution until a linking stage exists and
  is evaluated end to end.
- **Unit of analysis is the meeting**, not the utterance. Report all paired
  effects and whether removing any one meeting reverses the direction.

## Limits that travel with every number here

- Human utterance boundaries are **favourable** — production runs on Scribe
  boundaries. These numbers measure diarization quality, not end-to-end
  attribution.
- `humanReview: true` is not exhaustive acoustic ground truth. Rare speakers,
  overlap and boundaries need a sampled audit before any headline claim.
- Six selected meetings support a pilot. A deployable configuration needs a
  held-out confirmation on uncontaminated meetings.
- Single run per arm: service variability is unmeasured. Repeat `a0` on at
  least one meeting before trusting a small effect.

## Running it

```bash
# 1. Tests first — they guard the offset arithmetic the sliced arm rests on.
python3 -m pytest -q eval/diarization_clustering

# 2. Build frozen inputs. No API calls, no cost.
python3 eval/diarization_clustering/build_inputs.py

# 3. See what would be submitted, and what it would cost.
python3 eval/diarization_clustering/run_arms.py --arms a0 a3 --dry-run

# 4. Baseline only. ~6 audio-hours of a 150-hour trial.
export PYANNOTE_API_KEY=$(grep '^PYANNOTE_API_KEY=' .env | cut -d= -f2)
python3 eval/diarization_clustering/run_arms.py --arms a0

# 5. The free control, then the sliced arm. a3 is read ONLY against a0_local.
python3 eval/diarization_clustering/run_arms.py --arms a0_local
python3 eval/diarization_clustering/run_arms.py --arms a3

# Arms a1/a2 (speaker-count hints) are implemented but were NOT run, by user
# decision: the reviewed-people count is not the acoustic speaker count, so the
# hint would have forced the merging under study. Do not run them without first
# deciding what number is legitimate to send.
```

Raw responses are persisted under `<run-dir>/<meeting>/raw/` as they arrive:
**pyannote deletes job output 24 hours after completion.** Re-running an arm
reuses whatever is already on disk instead of resubmitting.

Scoring uses the upstream harness on branch `feat/exclusive-diarization-eval`
(worktree at `~/projects/opencouncil/oc-diar-eval`), which already reports
per-voice purity, mixed voices and people-split-across-voices:

```bash
npm run cli -- compare-diarization-modes \
  -D <run-dir>/<city>__<meeting>/diarization_a0.json \
  -T <run-dir>/<city>__<meeting>/transcript.json \
  -m <city>/<meeting> -O report_a0.json
```

## The 15-minute arm (`b3L`)

There is no `b3L` arm name. It is `b3` run against a second input directory built
with a larger minimum slice, which is why the report calls it `b3L` but the CLI
never does:

```bash
python3 eval/diarization_clustering/build_inputs.py \
  --out ~/.cache/oc-public/diarization-clustering-2026-09-long --min-slice 900
ln -s ~/.cache/oc-public/diarization-clustering-2026-09/voiceprints \
      ~/.cache/oc-public/diarization-clustering-2026-09-long/voiceprints
ln -s ~/.cache/oc-public/diarization-clustering-2026-09/reference-snapshot \
      ~/.cache/oc-public/diarization-clustering-2026-09-long/reference-snapshot
python3 eval/diarization_clustering/run_arms.py --arms b3 \
  --run-dir ~/.cache/oc-public/diarization-clustering-2026-09-long
```

The symlinks exist so the voiceprints and the frozen reference are stored once.
At 900 s only three meetings yield more than one slice, against four at 300 s.

## Budget, as spent

**89 jobs, 30.86 audio-hours** of a 150-hour trial expiring around 2026-10-15:
`a0` and `a3` (35 jobs, 12.25 h), `b0`, `b3` and `b3L` (47 jobs, 17.41 h), and the
threshold arm `b3t` on one meeting (7 jobs, 1.20 h). `a0_local` and `c3` are free.
