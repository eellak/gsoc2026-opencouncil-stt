# The benchmark's "human reference" is the published transcript

2026-08-03. Measured, not inferred. No GPU, no cost.

## The question that was asked

Are the no-edit utterances in the training set actually correct? The reasoning was that
they come from meetings a human reviewed, so an utterance nobody changed is an utterance a
human confirmed.

That reasoning is **half right**, and checking it turned up something larger.

## The half that is right

The no-edit backbone is not drawn at random. `docs/decisions/data.md` records the gate:
meetings with `humanReview=true` **or** `frac_user ≥ 15%`, with 13 meetings under 5%
edit-fraction explicitly denylisted because "a no-edit utterance there means *nobody
looked*, not *ASR was right*". So the meetings were review-exposed.

## The half that is wrong, and it was already written down

The same decision says the gate is "a minimum-viability gate, **not** verified-correct — the
open «trust non-corrected as ground truth?» question stands". A reviewer working through a
meeting does not read every utterance with equal care, and leaving text unchanged is not
the same act as confirming it.

## What the measurement adds

The benchmark samples windows from meetings above a `reviewedThreshold` of 75 and calls
the text for those windows the reference. Comparing that reference against the published
OpenCouncil transcript for the same time spans:

| | |
|---|---|
| WER, published transcript vs benchmark reference | **0.0008** |
| windows essentially identical (WER < 0.02) | **223 of 227** |
| windows with WER > 0.20 | 0 |

They are the same text. **The benchmark reference *is* the published transcript.**

For scale, on those same 227 windows the ASR systems score 0.131 to 0.169.

## Why this matters more than the training question

Three things follow, and none of them were visible before today.

**1. Every WER in this project measures distance from the published transcript**, not
distance from what was said. That is a legitimate product metric — it is what OpenCouncil
ships — but it is not accuracy, and it has been read as accuracy.

**2. A model that hears better than the published transcript is scored worse for it.** This
is not hypothetical. The [listening audit](2026-08-03-overlap-screen.md#8-the-words-the-reference-does-not-have) had a human
transcribe second speakers from blinded clips; of the words they heard that the reference
does not contain, Soniox's transcript held 43% and Scribe's 41%, against a 20% and 17%
chance baseline (p = 0.0004, Bonferroni-corrected). The reference omits speech that is
audibly there, and the two systems at the top of the leaderboard are penalised for
catching it.

How much speech? In the audited clips the missing words are about 11% of the reference
words — but those clips were selected for containing detected overlap **and** intelligible
second-speaker speech, so that is not a corpus rate and must not be quoted as one. The
corpus rate is unmeasured.

**3. Training on no-edit rows is training to imitate the published transcript**, including
wherever it is wrong. The [benchmark diagnosis](2026-08-02-benchmark-diagnosis.md) already
concluded the fine-tune's damage on clean audio comes from 48.1% corrections teaching an
edit bias and 51.9% no-edit rows teaching imitation. This says what the imitation target
actually is: not truth, but the production pipeline's output after partial review.

## What this means for the mixture question

The proposal was to sweep the ratio of clean to corrected data in short training runs and
follow whatever improves. That sweep is worth running — but it can only optimise
**agreement with a target that is itself imperfect in a known, systematic direction**.
Every ratio will be scored by how closely it reproduces the published transcript, and the
best-scoring ratio will be the one that best imitates it, omissions included.

So the sweep answers "which mixture best reproduces our current output", not "which
mixture makes a better model". Both are useful questions. Only the first is answerable
with what exists today.

## What would make accuracy measurable

A reference produced **independently of the pipeline**: a small set of windows transcribed
from audio by a person who has not seen any system output, including speech from anyone
audible rather than only the miked speaker. A few hours of audio would be enough to
calibrate how far the published transcript sits from what was said, and every existing WER
in this project could then be corrected by that offset instead of being quietly
misread.

That is review hours, not GPU hours, and it is the highest-value thing on the list.

## Caveats

227 of 260 windows matched a locally cached meeting JSON. The 0.0008 residual is
tokenisation and window-boundary trimming, not disagreement.

This does not say the published transcript is bad. It says it is the thing being measured
against, so it cannot also be the evidence that the measurement is right.
