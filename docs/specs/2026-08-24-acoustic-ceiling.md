# Where the acoustic ceiling is, and what lies past it

Written 2026-08-24, after the GSoC report was frozen. This is a direction, not a
preregistration. Nothing here is measured that is not cited; every number below
comes from an existing record and is named with its source.

## The claim being tested

Harold's reading: Scribe and Soniox beat us because they have a better decoder,
probably a language model correcting text before it is emitted. If so, our
approach has an **acoustic ceiling** it can reach and no further, and everything
past that ceiling is decoding.

The evidence splits cleanly, and it does not fully agree with the claim.

## What supports it

**We misspell homophones 2.5 times more often than Scribe.** Greek collapses ω and
ο to one sound, η ι υ ει οι to /i/, αι to /e/. A substitution whose two spellings
fold to the same skeleton was **not misheard**. It was misspelled, and no acoustic
model can be blamed for it.

| system | substitutions | homophone misspellings | share |
|---|---:|---:|---:|
| ours (clean-pack v2) | 11,708 | 855 | 7.3% |
| Soniox | 6,648 | 296 | 4.5% |
| Scribe v2 | 5,803 | 168 | 2.9% |

Source: [`2026-08-23-decoder-only-screen.md`](../reports/2026-08-23-decoder-only-screen.md).

The same bucket is the **most recoverable** of any error kind: Scribe gets 86.8% of
our homophone errors right, the highest rate in the table
([`2026-08-23-gap-to-scribe.md`](../reports/2026-08-23-gap-to-scribe.md)). Something
in their pipeline fixes exactly the errors an acoustic model cannot make an argument
about. That is language-model shaped.

**And where three systems disagree, the right answer is usually already present.**
In 58.8% of three-way disagreement columns the near pair contains the reference
word, but a frozen spelling rule picks the right member only 33.8% of the time,
against the current vote's 30.5%
([`exp-2026-08-24-near-miss-vote`](../reports/2026-08-24-near-miss-vote.md)). The
information is there; what is missing is something that can read the sentence and
choose. That is also language-model shaped.

## What undercuts it

**The shape of our errors is the same as theirs. Only the volume differs.** Share
of substitutions on 4+ character words that sit within two characters of the right
word: ours 44.6%, Scribe 41.1%, Soniox 38.8%. If their advantage were a correction
stage, their residual errors should look different from ours. They do not. We make
11,708 substitutions where Scribe makes 5,803, and they are the same kind of
substitutions.

**The decoder-shaped part is bounded, and it is small.** An oracle that fixed every
homophone misspelling we make moves us from 0.1795 to 0.1718. The gap to Scribe is
4.18 points. So the oracle is worth **0.77 points, 18.4% of the gap**, and the
subset Scribe actually recovers is **0.66 points, 15.8%**. Those are two different
numbers and the reports have conflated them; keep them apart. Either way, at least
four fifths of the gap sits outside this error class.

**"Scribe got it right" does not make our error acoustic.** This is the weakest
joint in the whole argument. Scribe's gain on a token could come from its encoder,
its cross-attention, its segmentation, its search, its context window, its training
data or its language prior. End-to-end ASR has no clean boundary between hearing and
decoding, and nothing measured here locates one. Read every "recoverable" figure as
"another system produced this token", never as "we misheard it".

**Neither vendor documents a correction stage.** ElevenLabs and Soniox describe
context handling, long-form stability, speaker separation and formatting. Neither
publishes an architecture with an LLM correction pass. Inferring one from output
quality is an unsupported step, and the argument here does not need it.

**The largest bucket is words, not spellings.** Far substitutions, wrong by four or
more characters, are 6,346 errors worth 0.0386, and Scribe gets 67.3% of them right.
Those are different words, not misspellings. A decoder cannot invent the right word
without acoustic evidence it never received. That bucket is hearing.

**The deletions are architectural and inherited.** 36.3% of our deleted tokens sit
in runs of five or more consecutive words, against 18.6% for Scribe. But base
`whisper-large-v3`, with none of our training, is at 32.0%, and the earlier adapter
at 42.3% ([`2026-08-24-deletion-runs.md`](../reports/2026-08-24-deletion-runs.md)).
The block-deletion behaviour comes with whisper. The unit that vanishes is a whole
speaker turn of about 2.3 seconds. No correction stage recovers words that were
never emitted.

## The honest split

Of the 4.18-point gap to Scribe, on the evidence we have:

- **0.77 points, 18.4%**, is decoder-shaped and provably not acoustic (homophones,
  full oracle); 0.66 of those are recovered by Scribe in practice.
- The rest sits in far substitutions and whole-turn deletions. Those are content,
  not spelling, and a text-only stage cannot reconstruct speech that left no textual
  trace.
- The buckets overlap, so the shares do not sum to the gap. They bound it from
  different sides.

Harold's intuition is right about the *existence* of a decoder-shaped defect and
right that we cannot spell our way to Scribe. It is wrong about the *proportion*.
But note what the evidence does NOT say: it does not say we hear worse. It says the
missing material is content rather than orthography, and where that content is lost
remains unlocated.

## Where the real headroom is, and it is not where we were looking

On the same 391 windows, three-system composition reaches 0.1202 while the
alignment-conditional per-column oracle over those same three hypotheses is 0.0611.
That is **5.91 WER points** left on the table by the chooser alone, larger than the
entire gap to Scribe.

0.0611 is a reference-conditioned oracle over three fixed hypotheses. It is not an
achievable operating point and it is emphatically not an acoustic ceiling; most of
that gap will not be learnable. What it does establish is an upper bound on how much
complementary correct material the candidates already contain, and that bound is
large.

So the strongest measured statement available is this: **a language model's job here
is choosing, not correcting.** Correcting our single hypothesis is bounded at 0.77
points. Choosing correctly among hypotheses we already have is worth up to 5.91, and
we know the right answer is present in the candidate set 58.8% of the time in
exactly the columns where the current vote fails.

## Inside whisper or beside it

Whisper's decoder already is an autoregressive language model, which is why the
decoder-only fine-tuning proposal was screened and bounded at 0.0066 of the gap.
Three placements are worth separating before any of them is built:

1. **Decoder-only LoRA.** Screened, bounded, not promising on its own.
2. **Shallow fusion or beam rescoring** with an external Greek LM, inside the
   decode. Touches only what whisper already emitted as alternatives, so it inherits
   the same ceiling as any single-hypothesis correction. If this is tried, naive
   shallow fusion double-counts the language prior an end-to-end model already
   carries; density-ratio style subtraction of the internal LM is the version worth
   testing, and whisper ships no external-LM fusion interface, so it is real work.
3. **Rescoring across systems**, over the confusion network of several hypotheses.
   This is the only one of the three whose ceiling is 0.0611 rather than 0.1718.

The first two are corrections. The third is a chooser. The numbers point at the
third.

## The ceiling is not identified

Nothing above locates it. 0.1795 is an operating point, 0.1377 is proof another
complete system does better, 0.1202 is an achievable ensemble, 0.0611 is an oracle.
None of the four is a ceiling for `whisper-large-v3` plus LoRA on this domain.

Two measurements would locate one, in ascending cost:

1. **Same-model N-best oracle.** This is also the precondition for every rescoring
   idea above, so it comes first regardless of which direction wins.
    Generate progressively richer candidate sets from
   our own model, wide beam, sampling, alternative chunking, timestamp and
   no-speech ablations, and plot oracle WER against candidate budget. Where it
   saturates separates candidate generation from ranking, and it is the precondition
   for any rescoring work: fusion cannot select a turn that was never emitted.
2. **Frozen-encoder probe.** Train an alternate decoder on frozen whisper encoder
   states, sweep capacity and data to saturation, then compare against a matched
   unfrozen encoder. The frozen plateau is the defensible empirical ceiling for that
   encoder. No output-only analysis can establish it.

## The measurement that would settle the framing

Everything above is agreement-with-OpenCouncil. The claim "we hear worse" deserves
fidelity-to-audio evidence, and this project has been burned before by merging the
two metrics.

The cheapest discriminator needs no audio at all: **re-score all four transcripts,
reference plus three systems, in phoneme space** through one frozen Greek lexicon and
grapheme-to-phoneme, with the same meeting-clustered bootstrap. If the 4.18-point gap
largely dissolves once orthographic homophones collapse, we spell worse. If it
survives as phoneme substitutions and deletions, we lose spoken content. Given that
the all-homophone oracle is only 18.4% of the gap, the prediction is that most of it
survives. No retraining, no re-decoding, no API.

That separates spelling from content. It does not separate encoder weakness from
segmentation or search, and it says nothing about fidelity to audio. For that,
the test is still a stratified human listen over the columns where Scribe is right
and we are wrong, split by bucket.

Open questions, none preregistered:

- Where does base `whisper-large-v3` sit on the same buckets? It bounds how much of
  each bucket is ours to fix at all.
- Is the 5.91-point chooser gap reachable without per-word confidence from the same
  decode pass? `exp-2026-08-18-conf-substrate` measured that confidence does not
  attach after the fact: 0 of 133 windows reproduce.
- Does an LLM chooser beat the hierarchical vote on the 667 near-miss columns, where
  we know the answer is present 58.8% of the time and the current vote finds it 30.5%?
