# Composition over selection: word-level fusion beats every selector we have

Preregistered in [`docs/specs/2026-08-16-composition-prereg.md`](../specs/2026-08-16-composition-prereg.md),
design frozen on [wayfinder #22](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/22)
before any number on the full sample was computed. Two Codex reviews ran, one before
the numbers and one before this text.

**Verdict.** Composing the three hypotheses word by word — no LLM, no audio, no
speaker information — takes WER from **0.1201 to 0.1005**, −0.01966 [−0.02292,
−0.01665], and lowers **all three** error components at once: deletions
0.0247 → 0.0203, insertions 0.0443 → 0.0374, substitutions 0.0512 → 0.0427. Every
CI excludes zero. Both preregistered rate gates pass. 208 of 247 windows improve, 29
worsen. Leave-one-out over windows, meetings and cities produces no sign flip
anywhere.

That is the largest single move this project has measured on this benchmark, and it
is free.

Everything bolted on top of it failed. The LLM arbiter changes nothing detectable
(−0.00035 [−0.00093, +0.00023]). The length guard and the speaker-grounded
restoration both **fail the insertion gate** and both make WER worse than plain W.

## Why the ceiling moved

Selection has an oracle: pick, in hindsight, the best whole hypothesis per window.
On this substrate that oracle is **0.1064** for the trio and **0.0995** for all nine
systems. Every selector this project has built — the consensus vote, the roster
selector, the LLM selector — lives under 0.1064 by construction.

W is not a selector. Its output is a text none of the three systems produced, so the
whole-window oracle is not its bound. It lands at 0.1005: **below** the trio's
whole-window oracle, and within 0.001 of the nine-system one.

| | WER | del | ins | sub |
|---|---|---|---|---|
| scribe-v2-clean | 0.1322 | 0.0380 | 0.0374 | 0.0568 |
| soniox | 0.1409 | 0.0142 | 0.0750 | 0.0518 |
| `artifact-adapter-fixed` | 0.1386 | 0.0459 | 0.0227 | 0.0700 |
| **V** — whole-window vote | 0.1201 | 0.0247 | 0.0443 | 0.0512 |
| oracle, whole-window trio | 0.1064 | 0.0257 | 0.0295 | 0.0511 |
| oracle, whole-window all 9 | 0.0995 | 0.0233 | 0.0229 | 0.0532 |
| **W** — per-column vote | **0.1005** | **0.0203** | **0.0374** | **0.0427** |
| **W+L** — plus LLM arbiter | 0.1001 | 0.0203 | 0.0374 | 0.0424 |
| W+len — plus length guard | 0.1108 | 0.0177 | 0.0467 | 0.0464 |
| W+D — plus speaker restoration | 0.1087 | 0.0187 | 0.0478 | 0.0421 |
| W+L+D | 0.1084 | 0.0187 | 0.0478 | 0.0419 |
| *alignment-conditional column oracle* | *0.0475* | *0.0157* | *0.0131* | *0.0187* |

Read the percentages carefully. W recovers **143%** of the gap between V and the
whole-window trio oracle and **95%** of the gap to the nine-system one. Neither
number means saturation: they compare a search space to a different, smaller search
space. Codex was explicit that "composition is not bounded by selection" is too
broad a reading. What is fair: **whole-window selection is not an upper bound on
composition**, and the trio oracle stops being the number to chase.

## The new ceiling, stated honestly

The per-column oracle — best entry per column, in hindsight — is **0.0475**, less
than half the whole-window trio oracle. Its own contrast against that oracle is
−0.05891 [−0.06441, −0.05394].

It is **alignment-conditional** and must not be quoted as an attainable or
alignment-free ceiling. Across the seven alignments measured it ranges
**0.0461–0.0479**, and the pattern inside that range is the informative part:

| alignment | W | column oracle |
|---|---|---|
| exact 3-way DP | **0.1005** | 0.0475 |
| progressive 012 | 0.1047 | 0.0461 |
| progressive 021 | 0.1038 | 0.0479 |
| progressive 102 | 0.1039 | 0.0461 |
| progressive 120 | 0.1031 | 0.0462 |
| progressive 201 | 0.1027 | 0.0479 |
| progressive 210 | 0.1038 | 0.0465 |

Every progressive ordering gives a **worse** W and four of six give a **better**
oracle. Hindsight exploits recombination opportunities that a suboptimal alignment
happens to open up and that no implementable voter can use. So: the exact alignment
is the right one for the arm, and the oracle number is a range, not a target.

What it does say is that three transcripts of this audio carry far more correct text
than any of them emits, and that 73% of that headroom survives W.

## What the vote actually does

80,659 columns:

| | columns |
|---|---|
| all three agree | 62,919 |
| two of three agree | 11,214 |
| epsilon wins (≥2 systems heard nothing) | 4,460 |
| tie, broken by the pivot | 1,812 |
| tie, broken by frozen priority | 254 |

The design turned on one correction from Codex before any number existed. The
issue's flat vote treats epsilon as a candidate, which deletes the column
`(ε, x, y)` — two systems heard a word, one heard silence, and `x != y` hands it to
silence. Voting **occupancy first, identity second** is why W's deletion rate falls
instead of rising. `test_msa.py` locks that behaviour, along with an exhaustive
check of the alignment against brute force on every input triple of length ≤ 3.

## The three arms that failed

**W+L (the LLM as arbiter).** 2,066 arbitration points, 2.6% of columns, each shown
±8 tokens of decided context, that column's own candidates and the meeting's closed
term list, and mechanically restricted to returning an *index*. 1,845 valid answers,
221 no-ops, 850 changed W's choice. Result: −0.00035 [−0.00093, +0.00023], 77 windows
better / 123 tied / 47 worse. This is **not detected as a benefit**; it is not
evidence that an LLM arbiter cannot help. One model, one prompt, one stochastic run,
221 no-ops, and no equivalence margin was preregistered.

Worth recording beside #18: given the *narrow* job — one position, a fixed candidate
list, no ability to write text — the LLM at least stopped doing damage. In #18, given
the wide job, it cost a full WER point.

**W+len (the length guard).** Fires on 116 of 247 windows and **fails the insertion
gate** (0.0467 > 0.0443), landing at 0.1108. Reverting to V whenever W is shorter
throws away most of W's win, because W is *supposed* to be shorter than V — that is
what removing Soniox's insertions looks like. The guard was designed against a
failure mode (#18's LLM preferring short text) that the per-column vote does not
have.

**W+D (speaker-grounded restoration).** 2,430 runs were dropped by W where exactly
one system heard something. 548 were rejected for a gap under 0.30 s, 1,540 by the
speaker rule, and 342 restored — 315 because pyannote showed ≥2 simultaneous
speakers, 27 because it showed a disjoint speaker set — putting 899 tokens back.
Deletions do fall (0.0203 → 0.0187) but insertions rise to 0.0478, **failing the
gate**, and WER worsens by +0.00822 [+0.00601, +0.01075]. The extra text is not
mostly recoverable speech.

### Did D fail at the timestamps or at the idea?

This was the declared technical risk, so it gets a direct answer: **partly at the
timestamps, and W never touched them.**

W and W+L are pure text. They use no timestamp, no audio and no speaker information,
so the timing risk cannot reach the arm that won.

D is the only arm that needs time, and the time is worse than the headline suggests.
Hypothesis tokens are placed by anchoring on pyannote precision-2 word timestamps.
Median anchor fraction is 0.878 / 0.864 / 0.910 per system (p10: 0.715 / 0.680 /
0.748). Over all 199,971 token pairs the MSA matched between two hypotheses, the two
independently derived times agree to within 0.5 s in 97.9% of cases and the median
discrepancy is exactly 0.0.

That 0.0 is an artefact and Codex caught it: a token anchored on **both** sides gets
the identical pyannote timestamp by construction. Restricted to the 17,236 pairs
(8.6%) where at least one side was **interpolated**:

| | p50 | p90 | p95 | p99 | < 0.5 s |
|---|---|---|---|---|---|
| all matched pairs | 0.00 s | 0.00 s | 0.00 s | 1.53 s | 97.9% |
| at least one interpolated | 0.00 s | 1.56 s | 2.98 s | 10.42 s | 79.0% |

Dropped runs are, by definition, text the other systems do not have, i.e. exactly
the unanchored regions where interpolation is at its worst. D is reading its
speaker evidence off the least reliable part of the timeline. That is not the whole
story — restoring text that only one system heard raises insertions whatever the
timing — but no one should read W+D's failure as "speaker information does not
help." It has not been given a fair timeline.

## Caveats

- **Third pass over the same 247 windows.** Design freezing does not remove
  adaptive-overfitting pressure across passes, and Codex named this the first thing
  to rule out. The 34 windows disjoint from **both** the glossary mining fold and the
  fine-tune training manifest (23 meetings, 10,150 reference tokens, no clustered
  CIs, underpowered) point the same way: V 0.1236 → W 0.1100 → W+L 0.1083, with the
  whole-window trio oracle at 0.1131 and the column oracle at 0.0520. Directional
  support, not validation.
- **No claim of generalisation.** Leave-one-out shows no existing cluster drives the
  result. It does not show the result survives on untouched audio. Nothing here has
  ever been confirmed outside this benchmark, and the trio itself was selected
  against these references upstream (`exp-2026-08-02-asr-fusion`).
- **Nothing about speaker-attribution accuracy.** There is no ground truth; it waits
  for [#21](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/21). Only
  WER / del / ins / sub of the composed text are measured. The D arm's speaker rule
  is judged solely by what it does to those numbers.
- **Agreement-with-OpenCouncil, not fidelity-to-audio.** "W recovers deletions"
  means it recovers words present in our own reference text.
- **No multiplicity correction.** Five arms and several contrasts share the 95%
  level. W's margin is wide enough that this is unlikely to matter; W+L's is not.
- The column oracle is hindsight potential under one alignment. Nothing here shows a
  usable composer can reach it.
- Scored with `eval/controlled_eval/scoring.py`, not the benchmark app's scorer, so
  these numbers are comparable to `exp-2026-08-16-fusion-deletions` and **not** to
  the published leaderboard.
- The 6 sealed temporal-holdout windows of `eval-freeze-2026-08` were filtered out
  before anything was computed, by the same explicit filter
  `exp_fusion_deletions.py` carries. They were never scored.
- Two Codex reviews shaped this. Job `8112dc72` replaced progressive alignment with
  exact DP, replaced the flat vote with hierarchical occupancy voting, replaced the
  0.95-of-median length threshold, fixed the oracle DP's boundary row and supplied
  the test list — all before any number. Job `7634f308` cut back the interpretation
  afterwards: it forbade "the LLM adds nothing", forbade "composition is not bounded
  by selection", forbade quoting 0.0475 as a ceiling, and found the
  zero-discrepancy artefact that the stratified table above now answers.

## What this changes

Fusion stops being "pick the best of three" and becomes "build a fourth". The
deletion problem that the serving-stack ladder declared unreachable — the words are
absent from all 8 of *our* hypotheses — is reached again here, harder than by the
window vote: 0.0459 for the adapter alone, 0.0247 for the window vote, 0.0203 for
the per-column vote.

The next question is not another arm on this substrate. It is whether W survives on
audio this project has never scored.
