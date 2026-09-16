# Scribe's per-word confidence, measured against a human who listened

`exp-2026-09-09-scribe-confidence-fidelity`. Split out of
[the routing probe](2026-09-09-scribe-confidence-routing.md), which closed negative on
routing and deliberately left this unrun. The two questions are independent: a signal can
fail to say where the fusion is worth paying for and still say where a human should listen.

Scribe v2 is the system the fusion falls back to, the system R2 hands every three-way
disagreement, and the only one of the three whose confidence had never been measured.

## The answer

**Mean within-meeting AUROC 0.7974**, against a pre-declared GO threshold of 0.60. The
signal is real, it is stable, and it is the weakest of the three — while having the
smallest blind spot.

| | Scribe v2 | Soniox | our adapter |
|---|---:|---:|---:|
| mean within-meeting AUROC | **0.7974** | 0.8167 | 0.8151 |
| deletions as a share of edit operations | **20.7%** | 22.8% | 41.1% |
| emitted words carrying a confidence | **100%** | — | 91.1% |
| bottom-decile lift | 3.94× | 3.51× | 3.88× |

**These three numbers are not strictly like-for-like and the table must not be read as a
ranking.** The methodology is identical by construction — every statistic is imported from
`eval/soniox_confidence_probe.py` rather than reimplemented — but the three systems emit
different words and therefore face different error populations. A system that abstains on
hard audio is graded on an easier set. What the table does support is that all three
confidences carry comparable information, and that they differ far more in *what they
cannot see* than in how well they rank what they can.

## The measurement

30 frozen gold cells transcribed with `scribe_v2` at the production `SCRIBE_PARAMS`
(`eval/scribe_gold_decode.py`), 30/30 succeeded, 0 failures, 17.5 minutes of audio,
about $0.07. 27 scored cells over 6 meetings in 6 cities, `core_envelope` region.
1,019 emitted words, 146 errors, prevalence 0.1433.

Frozen before any outcome was computed, and unchanged from the two prior probes:

- Primary statistic: mean of the within-meeting AUROCs of `1 − exp(logprob)` for
  predicting that an **emitted** word is an error.
- An error is `S` or `I` under one deterministic Levenshtein alignment, priority
  `(S, D, I)`, `wtoks` normalization. Deletions have no emitted word and are excluded
  by construction, not by choice.
- A meeting with no errors has an undefined AUROC and is dropped from the mean.
  All 6 were informative, so the rule did not fire.
- Words without a logprob are excluded and counted. **There were none.**

Robustness:

- All six meetings between 0.7137 and 0.8965. Leave-one-meeting-out 0.7776–0.8141 —
  no meeting carries the result.
- Meeting-cluster 95% interval [0.752, 0.847], six clusters. Descriptive, not a
  significance test and not an estimate for an unseen meeting.
- Within-meeting permutation null over 2,000 permutations: mean 0.5005, 95%
  [0.4169, 0.5783]. The observed value sits outside the whole band, one-sided p = 0.0005.
- Pooled AUROC 0.8346, average precision 0.4911 against a prevalence null of 0.1433.
- Bottom confidence decile: error rate 0.5644 against prevalence 0.1433, **3.94× lift**.

## The part that decides how it can be used

**The blind spot is the smallest of the three, and it is still a fifth of the damage.**
38 of 184 edit operations in the scored region are deletions — words Scribe never emitted,
which therefore have no confidence to be low. Confidence can attach to 79.4% of Scribe's
errors and to none of the remaining 20.6%.

Broken down by error type, the same pattern the other two probes found:

- substitutions against matches: 0.8473
- insertions against matches: **0.7155** — insertions are the weakest, again

Overlapping speech scores 0.836 (n=204) against 0.799 outside it (n=815), so the signal
does not collapse in crosstalk. That is a subset comparison on a conditioned population,
not a controlled contrast.

## What this changes

For the review-triage design, Scribe's confidence is the most usable of the three per-word
signals despite ranking lowest: it covers every emitted word, its blind spot is half the
adapter's, and it is attached to the system that supplies the fused transcript's timings
and its fallback. It is a **ranking** signal — no calibrated error probability is claimed,
and calibrating one is a separate piece of work.

It does not touch the deletion problem. Nothing per-word can.

## Caveats

- Fidelity-to-audio on 27 cells in 6 meetings. Meeting and city are fully confounded:
  one meeting per city.
- One model, one decode, one day. Scribe is a hosted model and may move.
- The gold set's human answers were frozen 2026-08-16 under the listening protocol of
  2026-08-09; this measurement inherits every limit of that set.
