# What 299 seconds look like when you stop aggregating

2026-08-18. Harness [`eval/tsfusion/`](../../eval/tsfusion). Experiment:
`exp-2026-08-18-timestamp-diagnostic`.

One two-window slice of `vrilissia/apr1_2_2026`, 299.29 s, laid out token by token with the
audio, the pyannote turns, the three ASR systems, the fusion vote W and the published
reference on one page. 835 columns, 818 published tokens, 62 diarization turns, 8 speaker
changes. The bundle carries verbatim speech and audio and lives only under
`~/.cache/oc-public/tsfusion-2026-08/`; only the code is in git.

This is a **diagnostic**, not an arm. No gate, no bootstrap, no preregistration. It exists
because every other number in this project is an aggregate over 247 windows, and an
aggregate cannot show you an error whose cause is visible only in situ.

## Errors, one by one

Each system re-scored against the published reference on this slice, using the same frozen
alignment the project uses everywhere:

| system | S | D | I | (S+D)/N | WER |
|---|---|---|---|---|---|
| W (3-way composition) | 10 | 16 | 11 | 0.0318 | 0.0452 |
| soniox | 24 | 10 | 14 | 0.0416 | 0.0587 |
| scribe-v2-clean | 27 | 21 | 7 | 0.0587 | 0.0672 |
| our adapter | 31 | 32 | 15 | 0.0770 | 0.0954 |

218 individual errors across the four systems, each with its reference context, its position
in the audio, and which systems had the word right. **16 of W's 37 errors were correct in
some other system** — the selection-loss class, and the only class a better arbiter can reach.

## Two things no aggregate showed

**Alignment drift.** Rows positioned by the midpoint of their uncertainty interval run
*backwards* where the three systems conflict over a run of columns: rows 86/87/88 read
32.31 / 32.62 / 31.89 s. Correct words are then charged as errors because they were aligned
against the wrong reference position. This is the visible surface of the failure that
`exp-2026-08-18-anchored-realignment` now preregisters. Anchoring on observed,
non-conflicted rows and interpolating between them makes the timeline monotone and fixes
those three rows to 30.684 / 31.152 / 31.620 s. 19 of 835 columns are interpolated.

**Reference conventions.** An audit of all 21 errors W was charged on this slice found
roughly 7 that are impossible conventions of the published text and 10 that are reference
omissions, leaving 4 possibly genuine. This motivated the open question of what our metric
should be once we know what the reference actually measures.

## The page's own warnings were mostly noise

Worth recording, because a diagnostic that cries wolf is worse than none:

- **Overlap** was painted on 40 columns. Requiring both ≥ 0.30 s of overlap **and** ≥ 30% of
  the word's duration confirms **3**.
- **`overlap_fraction`** is `top_ov / width`, the share of the word covered by the *assigned*
  speaker. It is coverage, not overlap. 88 rows sit below 1.0 and none of them is a fault.
- **Turn cards** were grouped by `speaker_state` rather than by speaker, so one handover
  shattered into 11 cards of 1–4 words, several labelled as overlap with no reason attached.
  Grouping by speaker gives 8 cards over the same audio.
- **`time_uncertainty`** has median 0.36 s and p90 0.72 s. Marking any nonzero value marks
  everything; the threshold now sits at ≥ 1.0 s, which selects 35 words.

The rare signals that *are* real stayed visible: 8 time conflicts, 1 unresolved column,
2 straddling words, 11 bracketed timings.

## Which clock to anchor on

Two systems here carry real word timestamps. The comparison is split:

| | soniox (stt-rt-v4) | ours (word_timestamps) |
|---|---|---|
| column coverage | 96.8% | 90.5% |
| median word duration | 0.18 s | 0.28 s |
| words straddling a turn edge | 61 (7.5%) | 44 (5.8%) |
| speaker changes inside a word | 3 of 8 | 0 of 8 |
| median turn-edge → nearest word edge | 0.058 s | 0.165 s |

On the 740 columns timed by both, the median start gap is 0.14 s and **15.8% differ by more
than 0.30 s**. Soniox wins coverage and turn-edge explanation; ours wins boundary
cleanliness. On 299 seconds and 8 speaker changes this is an indication, not a decision.

**The provenance trap:** these Soniox timestamps come from `stt-rt-v4`, not the
`stt-async-v5` text the benchmark W votes on. Anchoring on this clock is a separate question
from merging the two Soniox models, and merging them stops W being W.

## Caveats

- One slice. Every number is indicative; none carries a confidence interval or a gate.
- The `ref_omission_suspect` rule (occupancy ≥ 2) flags 11 of 11 insertions here. A filter
  that selects its whole population does not discriminate. It is reported beside the primary
  figure and never inside it.
- Agreement-with-OpenCouncil, as everywhere on this substrate.
