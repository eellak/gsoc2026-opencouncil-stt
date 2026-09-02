# A per-category chooser over the three-way alignment

Written 2026-08-26. Substrate `2026-08-22-post-june-held-out-test-clean-pack-cont-`,
391 windows split by the pinned city partition of
[`autoresearch.py`](../../eval/controlled_eval/autoresearch.py)
(PROTOCOL `autoresearch-2026-08-16b`): 222 search windows in six cities, 169
confirmation windows in five. `argithea` did not exist in the pinned ten-city split
and was assigned to CONFIRM before any number was computed.

## The question

The hierarchical per-column vote W scores 0.1202 on all 391 windows while the
alignment-conditional oracle scores 0.0611. Where the three systems disagree, can a
per-category policy — a deterministic rule where one works, an LLM where none does —
recover any of that, out of sample?

## What was built

Disagreement islands (maximal runs of non-agree MSA columns, 13,414 of them) are
assigned **algorithmically** to 21 categories, using no reference: four ordered
special routes (number formatting, filler-only, adjacent-duplicate-only, mixed
low-content) then 13 island-level signatures — which PAIR agrees, or which system is
silent — each split atomic / composite. Every island lands in exactly one bucket.

Each category got its own instruction. A search loop then ran on the six search
cities only: score the prompt, hand Codex the failures, take its revision, keep
whichever round scored higher. Three rounds per category.

## What the search found, and what it cost to learn

**Against the vote, the chooser looked worth +21.0 points weighted. Against the best
free rule it was +3.0.** The gap between those two numbers is the result. Comparing
to W flatters, because W is not the incumbent — the incumbent is whichever trivial
rule already wins in that category.

Per category, best LLM round versus best null on the same islands:

| category | islands | vote | Scribe | R1+R2 | delete | LLM | Δ vs best null |
|---|---:|---:|---:|---:|---:|---:|---:|
| PAIR_soniox+ours ATOMIC | 1085 | 48.3% | 46.7% | 48.3% | — | 61.7% | **+13.3** |
| PAIRE_scribe+soniox | 611 | 35.0% | 35.0% | 35.0% | 60.0% | 66.7% | **+6.7** |
| THREE ATOMIC | 1238 | 46.7% | 68.3% | 70.0% | — | 73.3% | +3.3 |
| THREE COMPOSITE | 2196 | 28.3% | 38.3% | 41.7% | — | 41.7% | 0.0 |
| DUP | 526 | 56.7% | 28.3% | 56.7% | 86.7% | 83.3% | −3.3 |
| TWO_no_ours | 459 | 15.0% | 16.7% | 31.7% | 60.0% | 55.0% | −5.0 |

The pattern held everywhere it was tested: **where a trivial rule has an edge, the
trivial rule wins; the LLM only pays where every rule is a coin flip.**

Three of six categories had round 0 as their best, so every Codex revision made them
worse. The revisions consistently traded abstention for aggression, and the wrong
interventions came with the right ones.

## Model comparison, and a correction

Sonnet and Opus were run on the same islands with the same frozen prompts. Opus won
55 of the 118 disagreements to Sonnet's 34, and lost only one category. It also
repeated itself exactly (63.3% twice on `PAIR_soniox+ours`) where Sonnet ranged
56.7–60.0% over three runs.

An earlier reading of this comparison was wrong and is recorded here because the
failure mode is reusable: a credit limit returns `is_error: true` with a **normal
exit code**, the JSON parser found no array, every island fell through to the
fallback, and the run reported both models tied at exactly the null. A failure that
looks like a measurement. `run_chooser.call` now refuses any result without a `[`.

Batch size was measured, not assumed: 60 islands per call lost no numbering and
answered every item in every model, at a quarter of the calls. Haiku 4.5 swung 13
points on batch size alone (43.3% at 15, 56.7% at 60) and was dropped for
instability rather than for accuracy.

## The confirmation

One run, five untouched cities, freeze `6537f8e2451e7dbd`, recorded so a second run
against the same freeze is refused by the code.

- `PAIRE_scribe+soniox`: search +10.0 → **confirmation +9.6** (239 islands)
- `PAIR_soniox+ours`: search +15.0 → **confirmation +5.3** (436 islands)
- Weighted **+6.8 points** over the best null, on 675 of 6,027 confirmation islands.

The +15 shrinking to +5.3 is what the city partition exists to catch.

## The WER, which is the number that counts

Island accuracy is not WER: an island is not a word, and replacing one column
realigns everything after it. Every confirmation window was rebuilt under the frozen
policy and rescored by the frozen scorer.

| arm | WER | sub | del | ins |
|---|---:|---:|---:|---:|
| **policy** | **0.12431** | 0.05276 | **0.02524** | 0.04631 |
| R1+R2 | 0.12825 | 0.05357 | 0.01962 | 0.05506 |
| W | 0.13798 | 0.05598 | 0.01434 | 0.06767 |
| Scribe v2 | 0.15210 | 0.05845 | 0.01046 | 0.08320 |
| Soniox | 0.16791 | 0.06857 | 0.01468 | 0.08467 |
| ours (LoRA) | 0.20883 | 0.12584 | 0.03321 | 0.04978 |

policy vs W: −0.01367 [−0.01953, −0.00869]. policy vs R1+R2: −0.00394
[−0.00762, −0.00066]. Both intervals exclude zero; the largest of 47 clusters
carries 12.1% and 15.5% of the effect.

**These WER values are NOT comparable to the 391-window numbers.** The confirmation
cities are harder: the same W scores 0.10710 on the search cities and 0.13798 here.
Only the rows of this table compare with each other.

## The same policy over the whole 391-window benchmark

Added after the confirmation, at Harold's request, because this is the number that
sits beside the published leaderboard figures. **It is not a held-out number**: the
prompts were chosen on 222 of these 391 windows, so 57% of it is contaminated.

| arm | WER | sub | del | ins |
|---|---:|---:|---:|---:|
| **policy** | **0.10581** | 0.04501 | 0.02517 | 0.03563 |
| R1+R2 | 0.11154 | 0.04544 | 0.01685 | 0.04925 |
| vote W | 0.12020 | 0.04875 | 0.01203 | 0.05942 |
| Scribe v2 | 0.13771 | 0.05242 | 0.00956 | 0.07573 |
| Soniox | 0.14752 | 0.06006 | 0.01237 | 0.07510 |
| ours (LoRA) | 0.17951 | 0.10577 | 0.03168 | 0.04206 |

policy vs W −0.01439 [−0.01720, −0.01164]; policy vs R1+R2 −0.00574
[−0.00786, −0.00375]. Over 117 clusters the largest carries 5.3% and 6.5%.

**The contamination check came out clean, and against expectation.** Splitting the
same run:

| | n | policy | R1+R2 | W | policy gain over R1+R2 |
|---|---:|---:|---:|---:|---:|
| SEARCH | 222 | 0.09428 | 0.09924 | 0.10710 | −0.00496 |
| CONFIRM | 169 | 0.12145 | 0.12825 | 0.13798 | −0.00680 |

**The absolute gain is larger on the unseen cities, and that on its own proves
nothing.** Harder cities carry more disagreement islands, so a policy that helps at
islands has more to work with there; a larger absolute gain is the boring
expectation, not evidence. The statistic that carries information is the RELATIVE
gain: 0.00496/0.09924 = 5.0% on search, 0.00680/0.12825 = 5.3% on confirmation. That
held, and that is what clears prompt-level overfitting.

What no city partition can detect is reference bias. If the published text is
under-verbatim at disagreement points, both halves of the split share the defect
equally, and every WER number here inherits it.

**Run-to-run drift, measured:** the same frozen policy on the same 169 confirmation
windows scored 0.12431 in the confirmation run and 0.12145 here. 0.00286 of WER from
model nondeterminism alone. The confirmation figure of record stays 0.12431, the
first one; 0.12145 is a second sample, not an improvement.

## Deployment shape

391 windows are **14.57 hours** of audio and 110,412 reference words; the median
window is 143 s. A median council meeting is 98 minutes (n=120, range 7–403), so
about **41 windows**.

Per meeting that is roughly 1,407 disagreement islands, of which about **178 reach
the LLM** — only the two categories where no rule works. At batch 60 that is
**3 calls per meeting**, about 15k input and 2k output tokens. Batch 60 was measured
safe in every model tested. Production is intended to run Sonnet rather than Opus;
the two measured per-category deltas were +5.0 and +1.7 points in Opus's favour on
60-island samples, which is not a settled difference.

## What the remaining gap is made of

Decomposing the 6,418 word errors between R1+R2 and the alignment-conditional oracle
over all 13,414 islands: **78% (5,021) is recoverable by choosing better inside the
candidate set we already build**, concentrated on islands of length 1–3. Only 22%
(1,397) needs candidates that do not currently exist, mostly in `THREE|COMPOSITE`.

That settles a question this work kept circling. The candidate set misses the oracle
path for 35.6% of length-4+ islands, which sounds alarming, but those islands carry
little of the word mass. **Effort belongs in choosing, not in composing.**

It also explains the shape of the advance: insertions fall 0.0594 → 0.0356 while
deletions rise 0.0120 → 0.0252 and substitutions barely move. The policy is not
recognising words better. It is deciding, more often correctly, that a word should
not be there — which is exactly the axis where the reference is least trustworthy.

## The open caveat, deliberately not closed

Deletions rose from 0.01434 to 0.02524, a 76% increase over the vote and 2.4x
Scribe's rate. 449 islands are emptied outright; 56% of them are ordinary words, and
by the reference 74.6% of the deletions are correct — leaving **114 that destroy
material that should have stayed**.

Those 114 lose overwhelmingly short function words (το 18, να 18, και 14, για 11),
which is the harmless case. Two entries are not harmless in this domain: **ναι 15
and δεν 11** — a vote and a negation. `δεν` is deleted 24 times and wrong in 11 of
them, `ναι` 24 times and wrong in 10. At disagreement points the policy has no
information about whether a negation was spoken; it is guessing.

A guard that never deletes a vote or a negation was priced: it rescues 26 and breaks
31, **net −5 islands, i.e. slightly worse WER**. It is still probably right, because
WER counts a spurious «ναι» the same as a lost vote and a minutes reader does not.
That asymmetry is not in the metric.

A blind listening test over 203 clips was built and **abandoned after three
judgements**, so this caveat is UNTESTED. The three judgements that were made all
found spoken-but-droppable material (repetition, slips), which if it generalises
means the reference is more verbatim than the product needs and WER penalises the
policy for correct decisions. Three judgements are not evidence. A 36-clip version
restricted to the vote and negation cases exists and was not run.

## THREE|COMPOSITE: closed for the text-only approach

This category is 2,196 islands and worth **0.0211 of absolute WER** if perfected —
about 40% of all remaining headroom, and the largest single prize left. It was
attacked hard and it did not move. Written out in full so nobody starts again from
scratch.

The frozen rule R1+R2 scores **42.5%** there. The whole-path selection ceiling is
**63.5%**: in 36.5% of these islands the correct answer is not any system's whole
span. But the correct answer uses **only words some system emitted, 100% of the
time**, so composition from those words has a ceiling of 100%.

Characterising the 805 islands where no whole span is correct: **68% are exactly one
token edit** from the nearest span (252 deletions, 270 substitutions, 25 insertions),
19% are two edits, and only 13% are genuine merges. Deletions sit at the span edge
72% of the time and are short function words (να, ναι, και, εε). Every replacement
word already exists in another span, 295 of 295, and many are inflection or format
twins (το/τον, στη/στην, 26/2026). Ceilings: one sourced edit **88.3%**, two **95.2%**.

So the mechanism looked right and the instruction was easy to state. It still failed.

| what was tried | result |
|---|---|
| instruction only, narrow context (8 tokens) | 41.7%, no gain over the rule |
| Codex prompt-revision loop, 3 rounds | round 0 was best; every revision was worse |
| few-shot 0/8/20, narrow context | 43.8% → 41.2% → **31.2%**, monotonically worse |
| wide context (30 tokens), few-shot 0/8/16 | 41.2% → 42.5% → 50.0% on 80 islands |
| free-text composition allowed | model composed 0 of 80; 2 invented words |
| structured edits, position-addressed (B) | **35.5%**, edited 3 of 200, 0 correct |
| structured edits, position+word cross-checked (D) | **39.0%**, edited 3 of 200, 0 correct |

The 50.0% at 16 shots was on 80 islands where the single-run 95% band is ±11 points.
On 200 islands with the contradiction removed it did not reproduce.

**The diagnosis, and it is not what it looked like.** The first explanation was a bug
in my own prompt: the base instruction said «μην ανακατεύεις λέξεις από διαφορετικά
συστήματα» and the appended tail invited mixing, so the model obeyed the first rule.
That contradiction was removed entirely for B and D. The model still edited 3 times
in 200, with two different output formats, 16 worked examples including compositions,
30 tokens of context either side, and both Sonnet and Opus.

It is not refusal and it is not format. **The model cannot see which word is the
spurious one.** All three candidate spans are equally plausible Greek; "which token
at the edge does not belong" is an acoustic question, not a linguistic one. Asking
for it also costs real accuracy — selection fell from ~41% to 35.5% — because
attention goes to a task that has no signal in the input.

**Closed for text-only methods.** The next thing with new information is acoustic
rescoring, designed in `exp-2026-08-26-chooser-deletion-safety`'s successor work.

## What this does not establish

- Nothing about unseen cities, other ASR systems, or other domains.
- Nothing about whether readers prefer the output; no reader test was run.
- The +6.8 and the WER deltas come from ONE run of a nondeterministic policy. Opus
  varied 2.8 points and Sonnet 25 points on identical inputs elsewhere in this work.
  Three repeats were recommended before freezing and were not run.
