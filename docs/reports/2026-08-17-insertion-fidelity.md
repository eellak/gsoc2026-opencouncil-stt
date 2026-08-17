# What our "insertions" actually are

2026-08-17. Record: `exp-2026-08-17-insertion-fidelity`.
Script: [`eval/insertion_fidelity.py`](../../eval/insertion_fidelity.py).
Tests: [`eval/tests/test_insertion_fidelity.py`](../../eval/tests/test_insertion_fidelity.py) (17).
Aggregates: [`eval/results_insertion_fidelity.json`](../../eval/results_insertion_fidelity.json).
Substrate: the frozen gold set of
[`exp-2026-08-16-gold-set`](2026-08-16-gold-set-findings.md), `answers
fa7a5dec…f063`, `cells 7f734db8…1b46a`. Zero GPU, zero paid API, no new audio,
no confirmation spent.

## The answer, in one sentence

On this frozen, deliberately overlap-enriched 27-cell audit, **18 of 76 (23.7%)
of our adapter's scored insertions and 53 of 130 (40.8%) of Soniox's are
supported by the single annotator's certain transcript and matched to gold
occurrences the published text fails to match under *every* minimum-cost
alignment**; a further 36.8% and 12.3% respectively are **undecidable** under the
coverage rule, and only 31.6% and 28.5% are unsupported. The classes are mutually
exclusive: a further **7.9% (6 of 76) and 14.6% (19 of 130)** are supported by the
annotator but sit on occurrences the published text *does* match in every optimal
alignment — duplications and alignment artefacts, not reference omission. For
Soniox the remaining **3.8% (5 of 130)** are supported and PUB-unmatched in some
but not every optimal alignment; that gap is the forced/possible band, and it is
empty for the adapter.

Intervals, meeting-clustered over **6 clusters** and descriptive, not
significance: 0.2368 [0.0417, 0.5714] and 0.4077 [0.2895, 0.5424]. One
annotator, one pass, six meetings, one meeting per city, and a cell mix chosen to
be overlap-rich rather than typical. **No gate, no ranking, no population
claim.**

Every class definition and threshold was frozen before a number existed; the
evidence is Codex design review job `847c449feb004c4abbf9b5243cdcd75c`, which
refused the first design and forced four structural changes to it. A second
review, job `cea10e780f034f7f91a09897f3000bec`, ran on the findings before any
claim was written and found an arithmetic error that is corrected here.

## What it covers, and what it cannot

**W was not run and cannot be run here.** W is defined over ElevenLabs Scribe v2
+ Soniox + the adapter, and there is no ElevenLabs credential in this
environment — the same wall `exp-2026-08-16-gold-set` hit. Covered:

| | |
|---|---|
| **ADP** | `artifact-adapter-fixed`, CTranslate2 int8 on the local CPU endpoint, per-word timestamps |
| **SNX** | Soniox `stt-rt-v4`, the **free realtime** path, per-token timestamps — **not** the paid `stt-async-v5` |
| **PUB** | the published OpenCouncil transcript. It is the **reference** here, so it has no insertions against itself |

Not covered: W, Scribe v2, `stt-async-v5`.

## Why this needed three alignments

Every WER this project ships is *agreement-with-OpenCouncil*: the reference is
our own published text. The gold set already showed that text omits real speech.
So when a system emits a word the published text lacks, the metric charges an
insertion — and some of those charges are for being right.

The trap Codex named as fatal: "the annotator also has this word" is **not** the
same as "the published text is missing it". PUB may be representing the same
spoken occurrence somewhere else, and the insertion is then an artefact of where
the alignment put things. Hence three alignments, not two:

| | reference | hypothesis | what it decides |
|---|---|---|---|
| **A** | PUB view | system tokens | which tokens the shipped metric charges as insertions — the population |
| **B** | gold certain tokens | system tokens | a conservative, order-sensitive support label — **sensitivity only** |
| **C** | gold certain tokens | PUB view | which gold occurrences the published text can be reading |

Support for the primary comes from neither: it is an occurrence-level,
temporally local, **injective, maximum-cardinality** matching between system
tokens and individual certain gold occurrences, whose objective never sees an
alignment-A label. All of a system's tokens compete for gold occurrences, not
only the ones the metric called insertions — otherwise a duplicated word could
claim an occurrence another token already represents. Maximum cardinality rather
than greedy matters: greedy can strand a token whose only candidate was taken by
a token that had two, which would have made the supported count depend on
processing order.

Alignment C is not read from one backtrace either. A single backtrace attaches a
repeated word to one occurrence arbitrarily, so the verdict is computed as a
**band over all minimum-cost alignments**: forward and backward optimal costs
mark every lattice cell on some optimal path, and each gold occurrence is
labelled matched-in-every, matched-in-some, or matched-in-none.

## The classes

- **`gold_supported`** — matched to a certain gold occurrence in a temporally
  admissible block. Split by what PUB does with that occurrence:
  - **`pub_unmatched_forced`** — PUB matches it in **no** minimum-cost alignment.
    Reference-omission-**consistent**.
  - **`pub_unmatched_possible`** — PUB fails to match it in at least one.
  - **matched in every optimal alignment** — a duplication or alignment artefact,
    **not** reference omission.
- **`undecidable`** — no match, and the token's plausible window is not wholly
  inside exhaustively transcribed, text-certain gold coverage.
- **`not_supported`** — no match, and the whole window *is* inside certain
  coverage.

The evidence rule is deliberately **asymmetric**, on Codex's insistence. Support
needs a match. Non-support needs the annotator to have covered the whole time the
word could have occupied. Proximity to a block never licenses a negative verdict:
the annotator transcribed blocks, not the gaps between them, and treating a
dilated block as annotated territory would have manufactured "hallucinations" out
of unannotated silence. The class is `not_supported` and never "not said" — one
anchored listener cannot establish that a word did not occur. For the same
reason nothing here is called "omission-attributable": that would sound causal,
and alignment C identifies a matching, not a cause.

Frozen with the classes: primary region `core_envelope` (the parent study's
declared primary), τ = 0.5 s dilating the word interval, τ grid {0.25, 0.5, 1.0},
the frozen S>D>I alignment tie-break with all 6 op priorities × forward/reversed
as an envelope over alignments **A and B** (alignment C is computed over all
minimum-cost paths and is tie-break-invariant by construction; the envelope
reaches it only through which tokens A calls insertions), τ_w grid
{0.75, 1.5, 3.0}, and the frozen scorer's **midpoint**
PUB-view rule, because the target is literally the metric we ship.

## Results

`core_envelope`, τ = 0.5 s, 26 cells, 999 certain gold tokens, 905 PUB tokens.

| | ADP | SNX |
|---|---|---|
| insertions | 76 | 130 |
| insertion rate vs PUB | 0.0840 [0.0178, 0.2086] | 0.1436 [0.0550, 0.2978] |
| **`gold_supported`** | **0.3158** [0.2000, 0.6667] | **0.5923** [0.5385, 0.7458] |
| — PUB-unmatched, **forced** | **0.2368** [0.0417, 0.5714] | **0.4077** [0.2895, 0.5424] |
| — PUB-unmatched, **possible** | 0.2368 [0.0417, 0.5714] | 0.4462 [0.3971, 0.5424] |
| — PUB matches it in every optimal alignment | 0.0789 (6) | 0.1462 (19) |
| **`undecidable`** | **0.3684** [0.2500, 0.8000] | **0.1231** [0.0810, 0.2381] |
| **`not_supported`** | **0.3158** [0.0000, 0.4046] | **0.2846** [0.1515, 0.3508] |
| alignment-B match (sensitivity) | 0.2763 | 0.5231 |

The forced/possible band is narrow — 18 vs 18 for ADP, 53 vs 58 for SNX — so the
repeated-word ambiguity alignment C could have introduced turns out to be small
here. That is a measured fact about this set, not a general property.

As a rate rather than a share, pooled over the whole corpus:

| | ADP | SNX |
|---|---|---|
| measured insertion rate | 0.0840 | 0.1436 |
| PUB-unmatched component (forced) | **0.0199** [0.0009, 0.0491] | **0.0586** [0.0181, 0.1247] |
| … possible | 0.0199 | 0.0641 [0.0236, 0.1320] |
| … upper bound, counting every undecidable as omission | 0.0508 [0.0157, 0.1118] | 0.0818 [0.0301, 0.1645] |

> **One cell has no published tokens in its scored region at all** — the
> production pipeline's coverage hole in the flesh — and contributes 7 ADP and 5
> SNX insertions with a zero denominator. It is **kept** in the pooled ratio,
> contributing to the numerator and nothing to the denominator. The first version
> of this analysis dropped such cells and reported 69/905 for a system with 76
> insertions; Codex caught it. Dropping them would have removed precisely the
> case most suggestive of reference omission.

Separately, and never merged into the same number: PUB's own gold-unsupported
tokens are 27 of 999 = 0.0270 [0.0041, 0.0519]. That is *fidelity-to-audio* for
the published text; everything above is *agreement-with-OpenCouncil*. The project
rule is that these never become one quantity, and this report obeys it.

## The class the cut F3 family would have deleted

[`docs/specs/2026-08-17-llm-composer-draft.md`](../specs/2026-08-17-llm-composer-draft.md)
§1 cut family F3 — "ask a text-only model to drop a word two systems both heard"
— arguing it would be rewarded for reproducing editorial omission. That class is
directly measurable here: a system's insertions **echoed** by the other system
within τ_w seconds. The counterpart may be any of the other system's region
tokens, not only its insertions, so the relation is deliberately asymmetric and
the two directions have different counts.

| τ_w | ADP insertions echoed by SNX | SNX insertions echoed by ADP |
|---|---|---|
| 0.75 s | n=18, supported 0.833, forced 0.556 | n=43, supported 0.860, forced 0.395 |
| **1.5 s** | **n=25, supported 0.720, forced 0.480** [0.111, 0.824] | **n=54, supported 0.796, forced 0.407** [0.146, 0.722] |
| 3.0 s | n=26, supported 0.692, forced 0.462 | n=59, supported 0.797, forced 0.441 |

Echoing raises the supported share from 0.32 → 0.72 (ADP) and 0.59 → 0.80 (SNX)
at every τ_w. On this set, roughly **four in five** of the words both systems
emit and the published text lacks are words the annotator also has, and about
**four in ten** are matched to occurrences PUB cannot be reading. F3 would have
deleted them and the WER would have improved. The draft's argument was right, and
it now has a number instead of an intuition.

This is **enrichment evidence, not independent truth**. ADP and SNX are both
large speech models trained on overlapping web-scale material; correlated errors
are exactly what a corroboration count cannot rule out. And the CIs at n=25 are
very wide.

## Does the published text lose speech in overlap? The direct test

The insertion split cannot answer this — it conditions on a system having
produced an insertion, and in genuine simultaneous speech ADP produces **0** and
SNX **7**. The direct test takes **gold occurrences** as the denominator and asks
how often PUB fails to match them:

| gold occurrences whose block… | PUB fails to match |
|---|---|
| lies wholly inside a strict overlap intersection | **0 occurrences** — empty by construction: gold times are block-level and a block is longer than its intersection |
| **touches** simultaneous speech | **25 / 74 = 0.338** [0.189, 0.486] — but see below |
| does not | **140 / 925 = 0.151** [0.089, 0.243], 6 meetings |

Occurrences in blocks touching simultaneous speech go unmatched by the published
text at about **twice** the rate of those that do not. That is the parent study's
finding recapitulated on a different denominator, and it is the analysis the
insertion split should never have been asked to do.

**Its interval is far weaker than it looks.** Only **2** of the 6 meetings
contribute any overlap-touching block — samothraki 18/37 and xylokastro 7/37 —
so [0.189, 0.486] is a two-cluster bootstrap whose endpoints are simply the two
per-meeting values. Read it as "two meetings, both above the non-overlap rate",
not as an interval.

For completeness, the exploratory adjacency diagnostic on insertions — a token
sitting in or beside a block that participates in an overlap somewhere — runs the
other way (ADP forced share 0.158 inside vs 0.263 outside, 3/19 vs 15/57; SNX
0.375 vs 0.427, 18/48 vs 35/82). **That is adjacency, not simultaneity, and it is
not an overlap effect.** It is reported so nobody rediscovers it and calls it one.

## Where this is fragile

**One meeting dominates the insertion supply.** Samothraki (`jul6_2026`) carries
54 of ADP's 76 insertions and 74 of SNX's 130 — the same meeting that was worst
for both systems in the parent study. Leave-one-meeting-out on the forced
PUB-unmatched share:

- SNX 0.387 – 0.446. Stable.
- ADP 0.179 – 0.318. **Not stable**: removing samothraki *raises* ADP's share to
  0.318, and 23 of ADP's 24 `not_supported` insertions live in that one meeting
  (ADP's `not_supported` share falls to 0.046 without it). Three of ADP's six
  meetings contribute **zero** forced-unmatched insertions.

No single **cell** dominates (largest is 21.1% of ADP insertions and 16.9% of
SNX's; 22.2% / 20.8% of the forced counts). Non-overlapping cells (minimum gap
60 s, clips 35 s) prevent a physical occurrence being counted twice, but they do
not make the cells independent.

**The region changes the level, not the ordering.** Three separate estimands;
never summed, never averaged, and none estimates the others:

| region | ADP rate / forced share | SNX rate / forced share |
|---|---|---|
| core_strict | 0.3120 / 0.527 | 0.4123 / 0.689 |
| **core_envelope (primary)** | **0.0840 / 0.237** | **0.1436 / 0.408** |
| clip | 0.1095 / 0.383 | 0.1750 / 0.480 |

What may be said: the rate and the class composition are **materially
region-dependent**; SNX is descriptively higher than ADP on both quantities in
all three regions; and ADP's share is the more region-sensitive of the two. This
**recapitulates** the region sensitivity the parent study found between
core-strict and core-envelope. It does not confirm a common cause, and no
cross-region percentage exists.

**The primary is conditional on the shipped boundary rule and is not invariant to
it.** The published text has utterance-level timestamps, so which utterances fall
in the scored region depends on a rule:

| rule | PUB tokens | ADP insertions (forced) | SNX insertions (forced) |
|---|---|---|---|
| **midpoint (frozen, primary)** | 905 | 76 (18) | 130 (53) |
| any_overlap | 1052 | 50 (10) | 95 (31) |
| wholly_contained | 714 | 229 (137) | 301 (200) |

These are partly **different estimands**, not error bars around one. Midpoint is
primary because it is the rule the shipped metric uses — that is enough to choose
a primary and not enough to claim boundary robustness. `wholly_contained` triples
the insertion count and its forced share because dropping a straddling utterance
removes *reference* words; that is the rule's artefact, and it is stated here
rather than in an appendix precisely because it is so different.

**τ moves only the undecidable boundary.** Over τ ∈ {0.25, 0.5, 1.0} the
supported counts barely move (ADP 24/24/26, SNX 77/77/77) while `undecidable`
grows (ADP 20/28/34, SNX 13/16/29) at the expense of `not_supported`. The
supported and PUB-unmatched numbers are robust to τ; `not_supported` is not, and
should be read as "at most this many". The class is measuring **ascertainability
under timestamp geometry and annotation coverage**, not semantic doubt.

**The alignment tie-break is small but not free:** over all 6 op priorities ×
forward/reversed, ADP's supported share spans 0.303–0.346 and its forced share
0.185–0.259; SNX's span 0.561–0.611 and 0.408–0.443. This is an alignment
sensitivity envelope, **not** the forced/possible identification band — those are
different objects and the table above keeps them apart.

**ADP's undecidable share is three times SNX's (0.368 vs 0.123), and the cause is
not identified.** Two checks against the obvious "Whisper timestamps drift"
explanation: the fraction of *all* region tokens whose dilated window lies wholly
inside certain gold coverage is **0.695 [0.635, 0.752] for ADP and 0.721 [0.669,
0.774] for SNX** — near parity; and ADP's words are longer (mean 0.395 s vs
0.265 s), which the containment check already absorbs since it uses the same
dilated window. Neither test supports the clock-artefact reading, and neither
identifies an alternative. **This is a property of the scoring pipeline, and it
is not evidence that ADP is less reliable or less intelligible.**

## What this does not license

**No transport to the 247-window benchmark, and none is attempted.** The
benchmark's insertion rates are 0.0227 (adapter) and 0.0750 (Soniox), with W at
0.03743 and the alignment-conditional column oracle at 0.01312. Here the same two
systems measure 0.0840 and 0.1436, on 15-second overlap-enriched cores with
utterance-boundary effects two-minute windows do not have. Two statements are
permitted and no more:

1. Soniox's PUB-relative insertion rate is descriptively higher than the
   adapter's in both datasets, though the ratio differs substantially (1.71 here,
   3.3 there) and the datasets are not exchangeable.
2. PUB-relative insertions **can** include words supported by an anchored
   transcript and unmatched by the published text; therefore **benchmark
   insertion headroom must not be equated with hallucination headroom**, and the
   2.43-point gap between W's 0.03743 and the oracle's 0.01312 is not wholly a
   catalogue of things the systems invented.

Explicitly forbidden, and not done anywhere in this report: applying 23.7% or
40.8% to the benchmark, producing corrected benchmark rates, revising the
2.43-point figure numerically, or inferring anything about W's or Scribe's
sensitivity to reference omission. The *direction* of this one mechanism is
identifiable — genuinely spoken words missing from PUB push PUB-relative
insertion counts up for systems that recover them — but the **net** bias in
benchmark WER is not, because other reference errors and alignment changes act
in ways this study does not measure.

**No system ranking.** SNX's supported share is higher in every region, but 6
confounded meeting/city clusters cannot rank systems, the parent study's ranking
already reversed between regions, and SNX also emits 1.7× as many insertions here
to begin with.

**No promotion, no gate, no confirmation spent.**

**The parent study's 80.3% is the mirror question, not this one.** It counted
occurrences **PUB has and ADP lacks** — adapter deletions. This report counts
occurrences **a system has and PUB lacks**. Same audio, opposite direction, not
two measurements of one quantity.

## Honest limits

- 27 cells, 6 meetings, **one meeting per city**: meeting and city are fully
  confounded and nothing here is a population estimate.
- **One annotator, one pass, no second annotator.** The intervals cover
  sampling and between-meeting variation **only** — the annotator's own error is
  in none of them. The reference is also prefilled from a Whisper-family system
  the human corrected, anchoring it toward Whisper vocabulary and segmentation,
  and ADP is Whisper-family.
- The 17 spans the annotator judged as lost speech in the parent study were
  **never written into the frozen reference**, so the gold knowingly
  under-covers and every `not_supported` count here over-counts by an unmeasured
  amount.
- Cell selection is stream-stratified and deliberately **overlap-enriched**
  (P=11 pyannote overlap, I=4 published interjection, H=4 speaker change, M=3
  single speaker, R=4 random). It is *not* error-driven, so it does not select
  for the systems' mistakes, but the mix is not the natural one and every
  pooled number is conditional on it.
- A meeting-clustered interval over 6 clusters that excludes zero **is not
  statistical significance**. No p-value is quoted anywhere in this report.
- The support matching maximises cardinality, so the supported *count* does not
  depend on processing order; **which** of two equally admissible gold
  occurrences a token receives still can, and that residual is not inside the
  forced/possible band, which holds the local assignment fixed.
- Cross-system corroboration is enrichment evidence, not independent
  confirmation.

## Review

- **Codex job `847c449feb004c4abbf9b5243cdcd75c`, on the written design, before
  any code produced a number.** It refused the first design and forced: the
  asymmetric undecidability rule (the symmetric one would have turned unannotated
  silence into hallucinations); the occurrence-level injective local matching as
  primary in place of an alignment-B match; alignment C, without which no
  omission claim was earned at all; the rename of `not_said` to `not_supported`;
  global capacity over gold occurrences with all system tokens competing; the
  strict-intersection definition of overlap; the removal of the position-free
  region-multiset definition from the headline; and a flat refusal of any
  transport to the 247-window benchmark.
- **Codex job `cea10e780f034f7f91a09897f3000bec`, on the findings, before any
  claim was written.** It found the rate-denominator error (zero-PUB cells were
  being dropped, deflating both numerator and rate in a self-flattering
  direction), demanded maximum-cardinality matching in place of greedy, demanded
  the forced/possible band over all minimum-cost alignment-C paths, renamed
  `omission_attributable`, required the direct gold-denominator overlap test in
  place of the insertion-conditioned one, ruled the adjacency comparison
  exploratory, and fixed exactly what may and may not be said about the
  benchmark. Every one of those is implemented above.
