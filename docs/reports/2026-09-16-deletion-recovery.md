# Putting back what the clean read drops: the words are real, the policy is not

`exp-2026-09-16-deletion-recovery` ran in one day for **$0.07**, on data that already
existed. Two measurements: a 391-window screen on agreement-with-OpenCouncil, and a
30-cell score against a human who listened.

The idea: Scribe v2 with `no_verbatim: true` suppresses disfluency *at the source* and
deletes 0.02118 of the reference against the verbatim arm's 0.00955. The same vendor in
the same request shape emits those words. Take clean as the base, put back the words only
verbatim said, skip the hesitations.

## Screen, 391 windows, agreement-with-OpenCouncil

`eval/deletion_recovery/screen.py`. Scored with the project's local scorer, so all arms
here are comparable to each other and **to nothing published** — clean reads 0.09056 here
against its benchmark figure of 0.09294.

| arm | WER | sub | del | ins |
|---|---:|---:|---:|---:|
| clean (base) | 0.09056 | 0.04591 | 0.02118 | 0.02346 |
| restore everything | 0.12908 | 0.04716 | **0.00757** | 0.07435 |
| oracle: additions that land on a reference word | 0.08009 | 0.04428 | 0.01016 | 0.02565 |

7,592 donor-only tokens, of which only **452 (6.0%)** are hesitations by the frozen
`FILLER_RE`. The mechanism works — restoring everything removes nearly the whole deletion
gap — and costs three insertions for each. Only **1,462 of 7,140** non-filler candidates
(20.5%) land on a reference word, and that share is **flat across token length** (11.5% at
one character, 19.4–22.4% above it), so length is not a selector.

> **Correction, same day.** The first version of this report said 1,543 / 21.6% and an
> oracle of 0.08418. That oracle credited any candidate whose token appeared *anywhere* in
> the reference's unspent multiset, which counts a restoration that makes the transcript
> worse: reference «α α β», base «α β», donor adding a trailing «α» scored as supported
> while raising the edit distance from 1 to 2. Found by CodeRabbit, fixed in
> `screen.restore`, and the oracle is now position-aligned. The correction moves the
> support rate down by about one point and does not change any conclusion.

## Gold, 27 cells / 6 meetings / 999 words, fidelity-to-audio

`eval/deletion_recovery/gold_score.py`, region `core_envelope` (primary; all three regions
are in the json and agree in sign). Thresholds are quantiles of the donor's own emitted
logprobs, frozen as a grid before scoring; every arm is reported and none is selected.

| arm | WER | sub | del | ins | Δ vs clean |
|---|---:|---:|---:|---:|---:|
| clean | 0.20420 | 92 | 85 | 27 | — |
| restore all (+80 words) | 0.21221 | 109 | 40 | 63 | +0.00801 |
| restore ≥ p25 (+51) | 0.20120 | 104 | 52 | 45 | −0.00300 |
| restore ≥ median (+29) | 0.20521 | 92 | 71 | 42 | +0.00100 |
| restore ≥ p75 (+7) | 0.20721 | 94 | 82 | 31 | +0.00300 |
| **verbatim, plain** | **0.18418** | 88 | 38 | 58 | **−0.02002** |

Absolute counts over 999 gold words; WER columns are rates.

**The words are real.** Restoring all 80 candidates removes 45 deletions: **56.2% of them
were words the human actually heard**, against 20.5% support on the published reference.
(Counting instead how many of the 80 align to a gold word gives 49, or 61.3%; the two
estimators differ because a restored word can also re-align its neighbours. 56.2% is the
conservative one and is the figure quoted.) The clean read drops real speech, and the
OpenCouncil reference does not contain it either — because that reference is itself a
clean read.

## What the dropped words actually are

`eval/deletion_recovery/what_is_dropped.py`, over the 49 candidates that align to a gold
word. Single-token categorisation with frozen closed lists and no context, so every count
is an upper bound on its category and a pointer for a human, not an adjudication.

| category | heard by the human | not heard |
|---|---:|---:|
| repetition of a neighbouring word | **29** | 13 |
| content word | **12** | 12 |
| function word | 4 | 5 |
| single character | 3 | 1 |
| negation or vote (`δεν`, `ναι`, `μη`…) | **1** | 0 |

**Three fifths of what the clean read drops is a word the speaker had just said** — the
mode doing its job. The remaining two fifths are not: 12 content words and one negation,
out of 49, in 999 words of audio. Scaled naively that is on the order of one dropped
content word per 75 reference words, and the published reference cannot see any of it.

Whether that rate is acceptable is the product question, not a measurement one. What the
measurement establishes is that it is **not zero**, and that no agreement-with-OpenCouncil
number will ever reveal it.

**The policy still fails its gate.** Pre-declared: ≥80% of recovered words judged real
speech **and** net gold edits fall. Measured 56.2%, and no arm falls below the plain
verbatim donor. The best restore arm (p25, −0.00300) does not approach simply *using the
donor* (−0.02002). Patching clean towards verbatim is a worse way to get verbatim than
asking for verbatim.

## The finding underneath, which is not about patching

`clean` wins on the OpenCouncil reference by 0.00619. `verbatim` wins on the human
reference by 0.02002. **Each reference rewards the mode that produced it**: the published
transcript comes from the clean-read pipeline, and the gold set was collected under the
2026-08-09 listening protocol, which keeps false starts in. Neither number is evidence
that one mode hears better.

So the mode is a **product choice about what a transcript is for**, not a measurement
result — and today's evidence says the two goals point in opposite directions. That
question is `exp-2026-09-16-scribe-mode-product-choice`.

## The clean arm's confidence ranks as well — and is blind to twice as much

Because both arms now exist on the gold cells, the confidence probe of
`exp-2026-09-09-scribe-confidence-fidelity` was run on each, same protocol, same
statistic, separate output files. The verbatim arm reproduces its published figure to
sixteen decimal places (0.7974037728145765), so the refactor that made the arm selectable
changed nothing.

| | verbatim | clean |
|---|---:|---:|
| mean within-meeting AUROC | 0.7974 | **0.7993** |
| meeting-cluster 95% CI | 0.752–0.847 | 0.704–0.879 |
| bottom-decile error rate / prevalence | 0.5644 / 0.1433 | 0.5426 / 0.1265 |
| bottom-decile lift | 3.94× | **4.29×** |
| emitted words without a logprob | 0 | 0 |
| edit ops S / D / I | 88 / 38 / 58 | 92 / **85** / 27 |
| **deletions as a share of all edits** | **20.7%** | **41.7%** |

**The ranking survives the mode switch; the coverage does not.** Clean Scribe's logprob is
just as good at ordering the words it emits — the AUROC difference of 0.0019 is nothing
against a cluster CI that spans 0.18 — and the bottom decile is, if anything, slightly
richer in errors. But a per-word confidence cannot see a word that was never emitted, and
on the clean arm **41.7% of the damage is exactly that**, against 20.7% on verbatim.

So any review-triage design built on Scribe's confidence reaches **58% of the clean arm's
errors and 79% of the verbatim arm's**. Choosing clean for the published transcript also
halves what the triage tooling can ever find. That is a second, independent cost of the
mode choice, and it points the same way as the deletion count.

Both arms pass their pre-declared 0.60 gate with a permutation p of 0.0005 or better, on 6
informative meetings. Neither is a calibrated probability, and the two are not strictly
like-for-like: they emit different words and therefore face different error populations.

## What would overturn this

- **27 cells, 6 meetings.** Verbatim is better in 4 of 6 and worse in 2, and one reversal
  is large (jun18: 0.1842 → 0.2895). This set can falsify; it cannot certify. Both
  reviewers of the 2026-09-15 result said independently that every surviving path runs
  through the gold set and the gold set has to grow.
- The 391 windows are now **partially spent** for this question: the blind and oracle
  numbers have been seen, so a selector tuned to beat them there would turn the benchmark
  into a development set.
- The gold cells' donor row is verbatim **by defect** — `eval/scribe_gold_decode.py`
  omitted `no_verbatim` exactly as the fusion harness did. That is why a donor existed to
  measure at all, and it is also why `exp-2026-09-09-scribe-confidence-fidelity` is a
  statement about the verbatim arm rather than the shipped one.
