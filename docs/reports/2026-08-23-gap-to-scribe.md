# The 4.18 points to Scribe, token by token (2026-08-23)

Where our adapter's gap to ElevenLabs Scribe v2 actually is, and what each route to
closing it is worth. 391 held-out post-June windows, 110,694 reference tokens, no
meeting from any training pack (`exp-2026-08-23-post-june-held-out`). Every reference
token was labelled with what each system did to it — kept, substituted, or deleted — so
the two can be compared token by token instead of rate by rate.

Agreement-with-OpenCouncil references throughout. No GPU, no audio, no API calls.

## 1. The gap is substitutions, and we already win on insertions

| | ours | Scribe v2 | ours − Scribe |
|---|---|---|---|
| substitutions | 0.1058 | 0.0524 | **+0.0533** |
| deletions | 0.0317 | 0.0096 | **+0.0221** |
| insertions | 0.0421 | 0.0757 | **−0.0337** |
| **WER** | **0.1795** | **0.1377** | **+0.0418** |

We write fewer words that were never said than Scribe does, by a wide margin. The
entire gap is that we get the words we do write wrong more often, and drop more.

## 2. Per reference token, who got it

| | tokens | share |
|---|---|---|
| both correct | 93,085 | 84.1% |
| **only Scribe correct** | **10,748** | **9.7%** |
| only ours correct | 2,394 | 2.2% |
| neither correct | 4,467 | 4.0% |

Two numbers matter here.

**9.7% is the whole prize.** Copy Scribe's answer into every one of those 10,748
positions and our error count falls from 19,881 to 9,133, or **0.0824** — better than
Scribe itself, because we also keep the 2,394 tokens Scribe misses and we do not.

That figure is a **bound, not a method**, in two separate senses, and both matter:

- **It needs the answer key.** Knowing *where* Scribe is right and we are wrong requires
  the reference. In production there is none; if there were, there would be no need for
  ASR. No decision rule can reach this by being clever.
- **It is not a clean oracle either.** It decomposes as 0.0404 (the 4,467 tokens
  *neither* system gets, which this pair can never recover) plus 0.0421 (our own
  insertions, carried over untouched). A real oracle could also discard insertions,
  which is why the trio's per-column oracle in
  [section 4.8 of the final report](2026-08-23-gsoc-final-report-DRAFT.md) sits lower
  still, at 0.0611.

What it is good for is scale. Composition currently reaches 0.1202; this pair's floor
is 0.0824 and the trio's is 0.0611. Roughly half of what the systems jointly know is
still being thrown away by the vote.

**2.2% is why fusion works.** There is a real, non-empty set of tokens our adapter gets
and Scribe does not. It is smaller than the reverse, but it is not noise, and it is the
mechanism behind [section 4.8](2026-08-23-gsoc-final-report-DRAFT.md) — three-system
composition reaching 0.1202, well below Scribe alone.

**4.0% is the floor.** Neither system gets these. No fusion of the two can recover them.

## 3. Our errors by kind, ranked by what is actually recoverable

"Recoverable" means Scribe got that exact reference token right, so the information
exists in a system we can already call.

| kind of error | ours wrong | Scribe right | recoverable | worth |
|---|---|---|---|---|
| **far substitution** (4+ chars off) | 6,346 | 4,268 | 67.3% | **0.0386** |
| near-miss substitution (1–2 chars) | 4,520 | 3,345 | 74.0% | 0.0302 |
| deletion | 3,507 | 2,404 | 68.5% | 0.0217 |
| homophone misspelling | 842 | 731 | 86.8% | 0.0066 |

This reorders the intuitions the project has been working from.

- **Far substitutions are the largest single bucket**, worth 3.86 WER points. These are
  not misspellings — they are different words. A decoder cannot invent the right word
  without acoustic evidence it did not get.
- **The homophone bucket is the most recoverable but the smallest.** Scribe fixes 86.8%
  of them, the highest rate in the table, and it is worth 0.66 points. This bounds the
  decoder-only proposal ([`2026-08-23-decoder-only-screen.md`](2026-08-23-decoder-only-screen.md)).
- Near-misses and far substitutions are recoverable at roughly the same rate (74% vs
  67%), which argues the near/far distinction is **not** the useful axis for deciding
  what to fix.

## 4. We do not drop words. We drop passages.

| | runs | tokens deleted | runs of 1 | runs of 2–4 | runs of 5+ | tokens in 5+ runs |
|---|---|---|---|---|---|---|
| ours | 1,759 | 3,507 | 1,239 | 392 | 128 | 1,274 (**36%**) |
| Scribe | 731 | 1,058 | 604 | 107 | 20 | 197 (19%) |

**36% of everything we delete disappears in stretches of five or more consecutive
words.** Scribe's figure is half that. This is the deletion property that matters for a
correction product: a missing word is noticed, a missing sentence is not. Long runs
cluster in Athens (19), Vrilissia (19), Argos (17) and Samothraki (17).

## 5. Two hypotheses this kills

**It is not the window edges.** The share of tokens that are ours-wrong/Scribe-right,
by decile of position inside the window: 9.03, 10.02, 9.80, 9.25, 9.01, 9.72, 9.95,
9.92, 10.03, 10.38. Flat. Whatever chunking costs us, it is not visible in this gap, so
`exp-2026-08-18-chunking-aware-decoding` is not the route to Scribe.

**It is not evenly spread across councils.** One city carries a quarter of it:

| city | share of the total gap | ref tokens | ours | Scribe |
|---|---|---|---|---|
| **samothraki** | **24.5%** | 8,893 | 0.270 | 0.103 |
| athens | 18.5% | 17,268 | 0.149 | 0.067 |
| sparta | 9.7% | 11,572 | — | — |
| vrilissia | 9.4% | 12,078 | 0.146 | 0.071 |
| xylokastro | 8.8% | 7,234 | 0.151 | 0.077 |

Samothraki holds 8% of the tokens and 24.5% of the gap, at a WER of 0.270 against
Scribe's 0.103. **Drop it and the gap falls from 0.0418 to 0.0343** — an eighth of the
whole problem is one island council. Nothing here says why; that needs listening, which
this screen deliberately does not do.

## 6. Where Soniox wins, and what our adapter actually adds

Comparing only against Scribe overstated our unique contribution. With all three
systems labelled on the same reference tokens:

| | WER | sub | del | ins | runs of 5+ deletions | tokens lost in them |
|---|---|---|---|---|---|---|
| Scribe v2 | 0.1377 | 0.0524 | 0.0096 | 0.0757 | 20 | 197 |
| Soniox | 0.1475 | 0.0601 | 0.0124 | 0.0751 | 34 | 282 |
| ours | 0.1795 | 0.1058 | 0.0317 | 0.0421 | 128 | 1,274 |

Soniox sits between the two on every axis. It is not a third opinion of a different
kind; on this window set it is a slightly worse Scribe. Its one distinction is that it
writes numerals as digits where Scribe writes them as words — see below.

Which systems got each reference token right:

| combination | tokens | share |
|---|---|---|
| all three | 91,167 | 82.4% |
| Scribe + Soniox | 8,112 | 7.3% |
| **none** | 2,972 | 2.7% |
| Scribe only | 2,636 | 2.4% |
| ours + Scribe | 1,918 | 1.7% |
| ours + Soniox | 1,903 | 1.7% |
| Soniox only | 1,495 | 1.4% |
| **ours only** | **491** | **0.4%** |

**Our adapter's unique contribution is 0.4%, not the 2.2% of section 2.** That figure
was measured against Scribe alone and counted tokens Soniox also recovers. Inside the
trio, the set only our adapter saves is 491 tokens, and they are function words:
`και`, `το`, `η`, `την`, `θα`, `είναι`, `να`.

**And the number signal was a formatting convention, not a listening advantage.** The
tokens Scribe uniquely recovers are number *words* — `δέκα`, `πέντε`, `δύο`,
`χιλιάδες`, `είκοσι`. The tokens Soniox uniquely recovers are *digits* — `2`, `3`, `5`,
`000`. The reference uses both conventions, and each system matches it where its own
convention happens to agree. Section 2's 2.2% is inflated by exactly this, and the
claim that we uniquely recover numerals is **withdrawn**.

**The uncomfortable consequence.** If the third voter is there to add information, ours
is the weakest of the three candidates tested: the trio carrying the **previous**
adapter scores 0.1183, the trio carrying gpt-4o-transcribe 0.1197, and the trio
carrying the new adapter 0.1202. The spread is about 0.002 and no separation is
claimed — but nothing in this data says our adapter is the *right* third voter. What it
says is that **having** a third voter is worth 1.75 WER points, and ours is the only
candidate that is self-hosted and free per minute. That is a cost argument, not an
accuracy one, and it should be written as one.

## 7. Ranked routes to Scribe, with measured ceilings

| route | ceiling | cost | status |
|---|---|---|---|
| **Fusion with Scribe + Soniox** | measured at **0.1202**, already below Scribe | 2–3 ASR accounts | done, `exp-2026-08-23-fusion-postjune` |
| Recover every token Scribe gets and we miss | 0.0824 | not a method, a bound | — |
| Fix long deletion runs (5+) | 0.0115 | unknown; needs the cause | not screened |
| Fix every far substitution Scribe gets right | 0.0386 | needs acoustic improvement | not screened |
| Post-hoc homophone/vocabulary repair | 0.0066 | free | next |
| Decoder-only retraining | ≤ 0.0066 | ~$0.40 GPU | gated behind the repair |
| Chunking-aware decoding | ~0 for this gap | — | **excluded by section 5** |

**The honest conclusion is the one the project already reached by another road.** Our
adapter does not catch Scribe by being fixed; the largest recoverable bucket needs
better listening, not better spelling. What does beat Scribe, today, on these windows,
is composing our adapter with it — because the 2.2% column exists.

## Caveats

- Agreement-with-OpenCouncil, not fidelity to audio. A "substitution" can be the
  published transcript's own choice rather than a true error.
- Error attribution comes from one Levenshtein backtrace per window. Where several
  alignments cost the same, the split between substitution and deletion/insertion is
  arbitrary; the totals are stable but bucket boundaries are not exact.
- The homophone fold is a hand-written Greek approximation, not a validated
  grapheme-to-phoneme model.
- Samothraki's 24.5% is descriptive. No cause is established and none is claimed.
