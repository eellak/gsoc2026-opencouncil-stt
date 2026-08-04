# Combining ASR systems beats every one of them, and the LLM is not what does it

2026-08-02. Scripts: `eval/controlled_eval/exp_fusion_headroom.py` (free, no LLM) and
`exp_fusion.py` (500 sonnet calls) · raw: `results_fusion_headroom.json`,
`results_fusion.json` · 250 benchmark windows, 147 meetings, 10 cities, human-corrected
references, no GPU.

Every number below is scored with `eval/controlled_eval/scoring.py` over the frozen
`2026-06-10-oc-benchmark` sample, so it is internally consistent and NOT identical to the
benchmark app's published leaderboard, which trims window-boundary words.

## The substrate nobody had used

`bench.opencouncil.gr` publishes a public `report.json` per run that contains, for each of
the 260 two-minute windows, the human reference **and the verbatim transcript of every
provider**. Seven systems, identical audio, one trustworthy answer key. We had been
reading the leaderboard off it and ignoring the rest.

That is the only place in this project where independent ASR systems can be compared and
combined at zero transcription cost. It is what made everything here cost 500 LLM calls
instead of a training run.

## 1. The systems fail on different windows

| | WER |
|---|---|
| Scribe v2 (best single) | 0.1319 |
| Soniox | 0.1404 |
| Gladia | 0.1425 |
| whisper-large-v3 | 0.1456 |
| our fine-tune | 0.1497 |
| gpt-4o-transcribe | 0.1677 |
| **oracle: per window, take the best of all 7** | **0.1016** |

The oracle needs the reference, so it is not a system. It is the ceiling on any method
that picks whole windows, and at 3 points below the best single provider it says the
errors are largely disjoint. Had it come out at 0.13 this report would have ended here.

## 2. A vote captures half of that, with no model at all

Per window, keep the hypothesis most similar to the other two. No reference, no LLM, no
training, no tuning.

| | WER | vs Scribe, clustered over 147 meetings |
|---|---|---|
| Scribe alone | 0.1319 | |
| **consensus vote, Scribe + Soniox + ours** | **0.1211** | **−0.0108 [−0.0142, −0.0074]** |

Two things about the trio. Our own fine-tune, second-worst on its own, is the most useful
third voter: swapping it for Gladia or base whisper captures less of the headroom. A model
that loses head to head can still be worth having, because what a voter contributes is
independence, not accuracy.

And the gain is **larger** on the 103 windows whose meetings are not in the fine-tune's
training data (−0.0135) than on the 147 that are (−0.0089), so training contamination is
not what is producing it.

Pairs do not work. With two hypotheses there is no majority to be in, and the vote lands
at 0.1349, worse than Scribe alone. Three is the minimum.

## 3. The question this experiment was built to answer

Hand an LLM the three hypotheses and let it write one merged transcript, and it beats
Scribe. That result on its own means nothing: an LLM handed a **single** transcript also
rewrites it, so "fusion beat Scribe" is equally consistent with the combination
contributing nothing.

So the two LLM arms share one system prompt, one output format, one failure policy and one
model. The only difference is how many transcripts arrive in the user message. They ran
interleaved, window by window, so service drift could not line up with an arm.

| arm | WER |
|---|---|
| A. Scribe, verbatim | 0.1319 |
| B. consensus vote, no LLM | 0.1211 |
| C. the LLM, given **one** transcript | 0.1354 |
| D. the same LLM, given **three** | **0.1174** |

**D − C = −0.0180, CI [−0.0228, −0.0132].** Better on 183 windows, worse on 47. Negative in
all ten cities, and leave-one-city-out never brings it above −0.0168, so no single city is
carrying it. On the uncontaminated subset alone it is −0.0155 [−0.0248, −0.0063].

The combination is doing the work. That was the thing worth knowing and it is now settled
on this benchmark.

## 4. The part that changes what we should build

**C is worse than doing nothing.** The LLM editing a single transcript scores 0.1354
against Scribe's 0.1319: +0.0035, CI [+0.0015, +0.0060]. It made 34 windows better and 47
worse, and tied on 169.

This is the third time the same tax has shown up. The
[fine-tune](2026-08-02-benchmark-diagnosis.md) damages clean audio because it was trained
on corrections. The [post-editor](2026-08-01-postedit-gate.md) damages one already-correct
utterance in six. Now an LLM over a two-minute window is net negative for the same reason:
most of what it reads is already right, and it edits anyway.

What changes when it sees three transcripts is not that it becomes a better editor. It
gets **evidence about where the uncertainty is**. Agreement between independent systems
marks the safe text, disagreement marks the places worth touching. That is a targeting
signal, and it is exactly what every single-input arm in this project has lacked.

**And the LLM adds little over the free vote.** D vs B is −0.0037, CI [−0.0080, +0.0008],
which includes zero. With the output gate applied to both, D reaches 0.1157 against the
vote's 0.1211, −0.0054 [−0.0090, −0.0017], significant but small.

So of the 1.45 points that fusion wins over Scribe, the vote accounts for 1.1 and the LLM
for the rest. The cheap half of the idea is most of the idea.

## What this is worth in practice

The vote is deployable now. It needs no training, no GPU and no prompt: run three
providers, compare, keep the middle one. The cost is that you pay three ASR bills instead
of one, which is the actual decision to put in front of OpenCouncil, not the WER.

If that is too expensive, the same logic suggests the cheaper shape: run the second and
third systems only where the first is uncertain, and let the LLM see the disagreements
only where they exist. Every result in this file points at selective application, and so
does the post-editor's break-even arithmetic.

## Caveats

The trio was chosen by trying every subset against these references, which is selection on
the test set. The split-half check in `exp_fusion_headroom.py` fits the choice on half the
meetings and scores it on the other half, and it holds (0.1110 vs a 0.1233 baseline), so
the **procedure** generalizes. The specific trio is unconfirmed on untouched audio and D's
absolute number should be read as exploratory. D − C does not depend on this, since both
arms see the same windows and the same model.

Six calls in arm C and seven in arm D failed on a session limit and fell back to Scribe.
The policy is identical in both arms, but falling back to Scribe costs C nothing while
dragging D toward the baseline, so this is conservative against the effect being reported.

The output gate rejects 11% of arm C and 8% of arm D here, mostly for splitting the text
into paragraphs. Its thresholds were tuned on 15-word utterances, and paragraph breaks in a
two-minute window are formatting rather than misbehaviour. The gated numbers are reported
as the deployable pipeline, not as the measurement.

Window-level selection is coarse. Word-level ROVER over a confusion network should sit
below the 0.1016 oracle, and it is the obvious next non-LLM step if this direction is
pursued.

Ten cities. As with the benchmark diagnosis, that is enough to see the effect is not
city-specific and not enough to describe Greek council audio in general.

---

## The chain

The August reports read in order. The one you are on is marked.

- [Why the fine-tune loses to base](2026-08-02-benchmark-diagnosis.md)
- **Combining ASR systems, and what does the work** (you are here)
- [Overlapping speech as an error marker](2026-08-03-overlap-screen.md)
- [The causal test: overlap costs 0.16 points](2026-08-03-synthetic-overlap.md)
- [Cutting the audio at speaker changes](2026-08-03-segmentation.md)
- [The reference is our own transcript](2026-08-03-the-reference-problem.md)
- [Seven in ten missing words were really said](2026-08-04-reference-omissions.md)
- [200 hours of meetings we have never seen](2026-08-04-public-meetings.md)
- [17.5% disagreement with what a human hears](2026-08-04-audio-faithful-reference.md)
- [The ranking flips](2026-08-04-the-ranking-flips.md)
- [Synthesis: what we did and why it matters](2026-08-04-what-we-learned.md)
- [The GSoC delivery plan](../specs/gsoc-delivery-plan.md)
