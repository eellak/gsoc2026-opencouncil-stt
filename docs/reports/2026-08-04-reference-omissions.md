# Seven in ten of the "hallucinations" were really said

2026-08-04. Script: `eval/controlled_eval/build_omission_audit.py` · answers:
`~/.cache/oc-overlap/omission_audit_answers.json` · 120 clips, 90 meetings, 10 cities, one
human listener, no GPU, no cost.

Follows [the reference problem](2026-08-03-the-reference-problem.md), which established
that the benchmark's reference text *is* the published OpenCouncil transcript, and left the
error rate of that transcript unmeasured.

## The design

Transcribing hours of audio independently is the right way to measure this and it is not a
task that gets done. So the question was made cheaper: where **Soniox and Scribe
independently emit the same word** that the reference does not contain, that word is a
candidate omission. Two systems inventing the same Greek word in the same place is much
less likely than both hearing it.

1,804 such words exist across the corpus — 2.4% of all reference words, in 225 of 250
windows. 120 were sampled, at most one per window so no busy window dominates, each cut
into a 10 s clip with the word named above it and one question: **was it said?**

## Result

| | |
|---|---|
| yes, audible | **79** |
| no | 33 |
| could not tell | 8 |

**Confirmed real: 70.5%, CI [62.3%, 78.6%]** (meeting-clustered, "could not tell" dropped).
Counting every uncertain case as a failure instead gives 65.8% [57.6%, 73.9%], so the
conclusion does not depend on that choice.

Applied to the corpus, that is **1,272 words, or 1.68% of the reference**, that were spoken
and are missing from the text every system is scored against.

That figure is a **lower bound in three separate ways**. It only counts words *both* top
systems caught, so omissions that only one system heard, or that none did, are invisible to
it. It counts a word once even where the reference drops a whole phrase. And confirmation
was *higher* in windows with many candidates (73% at six or more, 60% at two or fewer),
while the extrapolation applies the flat rate — the corpus weights those busy windows more
heavily than the sample did.

## The penalty is not shared equally

Each of those words, when a system transcribes it, is scored as an insertion error. So the
charge lands hardest on the systems that hear the most:

| system | unfair penalty | WER as measured | roughly corrected |
|---|---|---|---|
| **scribe-v2-clean** | **1.68** | 0.1319 | 0.115 |
| **soniox** | **1.68** | 0.1404 | 0.124 |
| our fine-tune | 1.00 | 0.1497 | 0.140 |
| gpt-4o-transcribe | 0.97 | 0.1677 | 0.158 |
| whisper-large-v3 | 0.94 | 0.1456 | 0.136 |
| gladia | 0.84 | 0.1425 | 0.134 |
| greek-whisper-v3-turbo | 0.53 | 0.4712 | 0.466 |

The order does not change, but the distances do. Scribe's lead over Gladia goes from 1.1
WER points to about 1.9. The benchmark has been systematically flattering the systems that
hear less.

The corrected column is an estimate, not a rescoring: "the system emits this word somewhere
in the window" is not identical to "the alignment charged it as an insertion there". It is
the right order of magnitude and the right direction, and a true rescoring needs word
timings the stored hypotheses do not carry.

## What this settles

The [insertion question from the overlap screen](2026-08-03-overlap-screen.md#4) is closed.
Soniox's insertion rate quintupling in high-overlap windows was never evidence of
hallucination; it was largely evidence of the reference being incomplete exactly where two
people speak. Two independent audits now say the same thing — the blinded second-speaker
transcription, and this one.

## What it changes about training

The [benchmark diagnosis](2026-08-02-benchmark-diagnosis.md) found the fine-tune damages
clean audio, and attributed it to 48.1% corrections teaching an edit bias plus 51.9%
no-edit rows teaching imitation. This puts a number on the second half: the imitation
target omits **at least** 1.7% of the words that were spoken, concentrated where more than
one person is talking.

So training on no-edit rows teaches the model, among other things, **to leave out the
interjector**. That is a learned behaviour, not noise, and more data of the same kind
deepens it.

A mixture sweep over clean-versus-corrected ratios remains worth running, and it now has a
clearly stated ceiling: every ratio is scored by agreement with a target that is missing
1.7% of the words, so the sweep finds the best imitation of the current pipeline and
cannot find anything better than it.

## What to do instead

The candidate-word method that produced this measurement is also a **repair** method. Where
two independent systems agree on a word the transcript lacks, that word is 70% likely to be
real. Feeding those back into the published transcripts fixes the training targets and the
benchmark reference at the same time, at roughly one human second per word.

The obvious next measurement is the same audit with the *rejected* third: of the 33 that
were not said, is there a pattern that predicts them cheaply, so the repair can be applied
without a listener at all.

## Caveats

One annotator, who is the project owner. They were blinded to which system proposed the
word and to every hypothesis, but the question itself ("two systems say this word is here")
is a leading one, and a second listener would be worth having before any repair is applied
in bulk.

Word position was estimated from the token's index in the Soniox output at an assumed even
speaking rate. It only had to land inside a 10-second clip with the word named, but a
mislocated word could be judged "no" while being present a few seconds outside the clip,
which biases the confirmed rate **down**.

120 clips in 90 meetings. The interval above is meeting-clustered and honest at that size.
