# Why the fine-tune loses to base whisper, read off the benchmark

2026-08-02. Source: the `2026-06-10-oc-benchmark` run (260 windows, 10h34m of council
audio, 10 cities, human-corrected references), report generated 2026-07-28.

The headline was already known from the [July postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md):
our adapter sits at **15.33% WER against 15.02% for the un-adapted whisper-large-v3**,
with CER 9.54 against 8.75. We are behind the model we started from, and behind Scribe v2
(13.35), soniox (14.26) and Gladia (14.70).

What the per-city breakdown adds is the reason, and it is not the one we had been assuming.

## The adapter damages clean audio and helps on hard audio

Sorted by how hard each city is for the base model:

| city | base WER | ours | Δ |
|---|---|---|---|
| Χαλάνδρι | 8.27 | 10.26 | **+1.99** |
| Ζωγράφου | 10.50 | 12.98 | **+2.48** |
| Βριλήσσια | 11.48 | 12.49 | +1.01 |
| Ξυλόκαστρο | 12.69 | 13.95 | +1.26 |
| Χανιά | 13.04 | 13.16 | +0.12 |
| Ορεστιάδα | 14.74 | 13.37 | **−1.37** |
| Σπάρτη | 14.85 | 13.71 | **−1.14** |
| Αθήνα | 18.70 | 18.74 | +0.04 |
| Άργος | 21.41 | 20.13 | **−1.28** |
| Σαμοθράκη | 23.51 | 24.70 | +1.19 |

In all four cities where base whisper is already under 13% WER, our adapter is worse, by
1.7 points on average. Where the audio is hard it is roughly neutral, and it helps in three
of the six. Correlation between base difficulty and our delta: −0.47.

The model did learn something. It just learned a habit that only pays off when the
recording is bad, and charges for itself when the recording is good. Since most council
audio in this corpus is closer to the good end, the average comes out negative.

## Half the training data teaches it to copy the previous system

Composition of the 28,967 training clips: **13,929 corrections (48.1%) and 15,038 no-edit
rows (51.9%)**. The no-edit rows are utterances no human ever touched, so their training
target is whatever the old pipeline produced, errors included.

Put plainly, we taught the model two things and both of them were the wrong thing to teach:

On the corrections half, "change what you hear". Nearly every one of those 13,929 examples
is an utterance the ASR got wrong paired with the fix, so the model acquires a standing
bias toward editing. That bias is free when the input is wrong and expensive when it is
already right.

On the no-edit half, "reproduce the previous system's output". Those labels were never
verified by a person. Where the old pipeline made a mistake, the mistake was the target.

## Why the old evaluation said the opposite

Both of our validation subsets measured exactly the two populations we had trained on.

`val_corr` contains only utterances a human corrected. That is the regime where the model
helps by construction. It is the good quarter of the distribution, measured on its own.

`val_reg` contains no-edit utterances scored against the old pipeline's text. The
24.8 → 10.4 improvement we reported there is not better transcription. It is the model
learning the function that produced the labels, and succeeding at it. We were grading it on
imitation and reading the score as accuracy.

The benchmark is the first evaluation we have with human-corrected references over ordinary
speech rather than over the corrections. Nothing was hidden. We had simply never asked the
question in a form that could return a bad answer.

## The same failure showed up somewhere else this week

The [LLM post-editor probe](2026-08-01-postedit-gate.md) found that feeding an editor text
that is already correct damages roughly one utterance in six, which put break-even at 20% to
24% of utterances needing correction against a real rate of 24.6%.

That is a completely different mechanism reaching the same place. Two systems, one trained
and one prompted, both built from the population of things that were wrong, both applied to
everything, both paying a tax on the majority of inputs that needed nothing. It is worth
treating as a property of the problem rather than a coincidence of two experiments.

## Caveats

Our model returned **10 errors** in the benchmark run (HTTP 524 from the mini-PC behind
Cloudflare). Those items are missing from our average while present in everyone else's, so
15.33 against 15.02 is not strictly like for like. The gap is small enough that this matters.
A re-run through the RunPod serverless endpoint would settle it.

Ten cities is a thin basis for a correlation. The "worse in four of four clean cities" claim
is the stronger one; the −0.47 is a description, not a finding.

The benchmark sample is frozen and shared across providers, which is what makes it usable at
all here. It is still a stratified sample of two-minute windows, not the full corpus.

## What this changes

Retraining the same recipe on the same data will not fix this, and the
[corrected run](2026-08-02-fulltrain-corrected.md) is evidence: with the label bug gone, the
numbers moved by fractions of a point. The objective was one problem and it is fixed. The
data is the other one and it is not.

Two directions follow, and they cost review hours rather than GPU hours:

Drop the 15,038 unverified no-edit rows, or verify them. Dropping leaves 13,929 corrections
and makes the edit bias worse, so on its own it trades one failure for the other. The
backbone was the right idea. Its labels need to be human-checked rather than inherited from
the system we are trying to beat.

Apply selectively. Both the adapter and the post-editor are net positive where an error is
likely and net negative where it is not, so the deployment question is not "which model" but
"where do we let it run". That question is answerable with the confidence signals we already
have, and it does not need another training run to explore.
