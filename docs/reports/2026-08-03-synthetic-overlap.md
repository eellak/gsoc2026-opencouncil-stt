# Adding a real interjector costs almost nothing, and that closes the DiCoW question

2026-08-03. Scripts: `eval/controlled_eval/synth_overlap_build.py`, `_run.py`,
`_analyze.py` · raw: `results_synth_overlap.json` · 95 items, 72 target meetings, 30 donor
meetings, 855 decodes, whisper-large-v3 on one RTX A4500, ~95 minutes, $0.30.

Preregistered in
[synthetic-overlap-preregistration.md](../specs/synthetic-overlap-preregistration.md),
reviewed by Codex at high effort, gate and thresholds frozen before a single mixture was
transcribed.

## The design in one paragraph

Take a benchmark window with no detected overlap. Mix in a **real** 1.5–3.0 s excerpt of
another council speaker from a different city, once, placed where the main speaker is
actually talking. Score against the **unchanged** human reference, so the contrast is the
same target speech with and without a competing voice. No arm is normalised on its own —
one common attenuation is applied to every arm of an item, clean included, so "mixed is
worse" can never mean "the target got quieter".

## Result

| arm | what it is | WER | vs clean |
|---|---|---|---|
| A | clean | 0.1314 | — |
| B | interjector at +15 dB | 0.1293 | −0.0021 [−0.0063, +0.0003] |
| **C** | **interjector at +5 dB — primary** | **0.1330** | **+0.0016 [−0.0049, +0.0076]** |
| D | interjector at 0 dB, equally loud | 0.1398 | +0.0084 [+0.0003, +0.0168] |
| E | envelope-matched noise, +5 dB | 0.1279 | −0.0035 [−0.0110, +0.0024] |
| F | the same voice reversed, +5 dB | 0.1323 | +0.0009 [−0.0038, +0.0046] |
| G / H | clean at ±3 dB | 0.1325 / 0.1305 | ±0.001 |

**The gate fails, and not narrowly.** It required a burden of 2.0 WER points at +5 dB; the
measured burden is **0.16 points**, with a confidence interval that excludes anything above
0.8. Even at 0 dB — an interjector exactly as loud as the person at the microphone, which
the corpus does not contain — the cost is 0.84 points.

The speech-specific excess `C − E` is +0.0051, one-sided 90% lower bound +0.0005. There is
a real component that comes from the competing signal being *speech* rather than matched
noise, but it is a twentieth of what the gate asked for.

G and H confirm the pipeline is not gain-sensitive: shifting the whole clean file by ±3 dB
moves WER by a thousandth. Spillover is small too — 1.8% of the words in audio that is
bit-identical between arms changed, so the damage stays roughly where the event is.

And the dose here is **generous**. precision-2's event geometry over the corpus puts the
median real overlap event at 0.52 s; the preregistered 1.5–3.0 s sits above the 90th
percentile. Real interjections are shorter than the ones we paid for, so the true burden
is smaller than 0.16 points, not larger.

## Why this matters more than it looks

The [observational screen](2026-08-03-overlap-screen.md) found that windows in the top
overlap quartile score roughly **10 WER points worse** for all seven systems. That number
was always labelled an association. This experiment adds overlap directly, at natural
prevalence, and recovers **0.16 points**.

The gap is the whole story. Overlapping speech does not cause the errors that occur in
overlapping windows. Something else about those windows does, and overlap travels with it.

That is the same conclusion the corpus analysis reached by a completely different route,
when speaker-turn density beat overlap as a predictor and overlap added nothing on top of
it. Two independent methods, one observational and one causal, agree: **overlap is a marker
of contested passages, not the mechanism.**

## What this decides

**Diarization-conditioned Whisper is not worth training for this corpus.** The gate was
written before the data existed precisely so this could be settled rather than argued, and
it fails by more than a factor of ten. Recovering all of the burden this experiment can
measure would buy about 0.2 WER points, against a training run, a maintenance burden and a
runtime dependency on a diarizer.

For comparison, the [free consensus vote](2026-08-02-asr-fusion.md) already delivers 1.1
points and needs no model at all.

The [separation experiment](../specs/segmentation-experiment-preregistration.md) is also
moot as an overlap remedy. It was gated on this result and this result did not pass.

## What this does not decide

It does not say natural crosstalk is harmless — it says *adding one interjector to
otherwise clean audio* is nearly harmless, which is a claim about the manipulation
performed. Prolonged argument, reverberant multi-speaker rooms and off-mic chatter that
was present all along are outside it.

It does not explain what *does* make the high-overlap windows hard. Turn density is the
best available marker and the [segmentation experiment](2026-08-03-segmentation.md) showed
that acting on it naively does not pay either. That question is open.

It does not generalise past whisper-large-v3. Our fine-tune could not be moved to the pod
at 180 kB/s, and no commercial provider was included, on cost.

95 items in 72 meetings, so the clustering has room; the intervals above are meeting-
clustered and wide enough to be believed at this size.

---

## The chain

The August reports read in order. The one you are on is marked.

- [Why the fine-tune loses to base](2026-08-02-benchmark-diagnosis.md)
- [Combining ASR systems, and what does the work](2026-08-02-asr-fusion.md)
- [Overlapping speech as an error marker](2026-08-03-overlap-screen.md)
- **The causal test: overlap costs 0.16 points** (you are here)
- [Cutting the audio at speaker changes](2026-08-03-segmentation.md)
- [The reference is our own transcript](2026-08-03-the-reference-problem.md)
- [Seven in ten missing words were really said](2026-08-04-reference-omissions.md)
- [200 hours of meetings we have never seen](2026-08-04-public-meetings.md)
- [17.5% disagreement with what a human hears](2026-08-04-audio-faithful-reference.md)
- [The ranking flips](2026-08-04-the-ranking-flips.md)
- [Synthesis: what we did and why it matters](2026-08-04-what-we-learned.md)
- [The GSoC delivery plan](../specs/gsoc-delivery-plan.md)
