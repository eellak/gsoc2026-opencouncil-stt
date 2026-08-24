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
4.18 WER points. Homophone spelling is about **0.6 of them, roughly 15%**.

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

- about **0.6 points** is decoder-shaped and provably not acoustic (homophones),
- about **3.9 points** sits in far substitutions and deletions, which are hearing
  and architecture,
- and those overlap, so the shares do not sum to the gap. They bound it from
  different sides.

Harold's intuition is right about the *existence* of a decoder-shaped defect and
right that we cannot spell our way to Scribe. It is wrong about the *proportion*:
most of the gap is that Scribe hears passages we do not.

## Where the real headroom is, and it is not where we were looking

On the same 391 windows, three-system composition reaches 0.1202 while the
alignment-conditional per-column oracle over those same three hypotheses is 0.0611.
That is **5.91 WER points** left on the table by the chooser alone, larger than the
entire gap to Scribe.

So the strongest measured statement available is this: **a language model's job here
is choosing, not correcting.** Correcting our single hypothesis is bounded at 0.6
points. Choosing correctly among hypotheses we already have is worth up to 5.91, and
we know the right answer is present in the candidate set 58.8% of the time in
exactly the columns where the current vote fails.

## Inside whisper or beside it

Whisper's decoder already is an autoregressive language model, which is why the
decoder-only fine-tuning proposal was screened and bounded at 0.0066 of the gap.
Three placements are worth separating before any of them is built:

1. **Decoder-only LoRA.** Screened, bounded, not promising on its own.
2. **Rescoring the beam** with an external Greek LM, inside the decode. Touches only
   what whisper already emitted as alternatives, so it inherits the same ceiling as
   any single-hypothesis correction.
3. **Rescoring across systems**, over the confusion network of several hypotheses.
   This is the only one of the three whose ceiling is 0.0611 rather than 0.1718.

The first two are corrections. The third is a chooser. The numbers point at the
third.

## The measurement that would settle the framing

Everything above is agreement-with-OpenCouncil. The claim "we hear worse" deserves
fidelity-to-audio evidence, and this project has been burned before by merging the
two metrics.

The cheapest decisive test: take a stratified sample of the columns where Scribe is
right and we are wrong, split by bucket (far substitution, homophone, deletion run),
and have a human listen. If our far substitutions turn out to be audible on the
recording, the ceiling is not where this document places it.

Open questions, none preregistered:

- Where does base `whisper-large-v3` sit on the same buckets? It bounds how much of
  each bucket is ours to fix at all.
- Is the 5.91-point chooser gap reachable without per-word confidence from the same
  decode pass? `exp-2026-08-18-conf-substrate` measured that confidence does not
  attach after the fact: 0 of 133 windows reproduce.
- Does an LLM chooser beat the hierarchical vote on the 667 near-miss columns, where
  we know the answer is present 58.8% of the time and the current vote finds it 30.5%?
