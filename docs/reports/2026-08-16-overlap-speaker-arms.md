# Overlap-restricted speaker arms, and an omission rule that can count speakers

Date: 2026-08-16
Preregistration: [`docs/specs/2026-08-16-overlap-speaker-arms-prereg.md`](../specs/2026-08-16-overlap-speaker-arms-prereg.md)
Code: [`eval/controlled_eval/overlap_arms.py`](../../eval/controlled_eval/overlap_arms.py),
[`eval/controlled_eval/exp_overlap_arms.py`](../../eval/controlled_eval/exp_overlap_arms.py),
[`eval/controlled_eval/density_omission.py`](../../eval/controlled_eval/density_omission.py),
[`eval/controlled_eval/exp_density_omission.py`](../../eval/controlled_eval/exp_density_omission.py),
[`eval/controlled_eval/exp_density_gold.py`](../../eval/controlled_eval/exp_density_gold.py)
Results: `eval/controlled_eval/results_overlap_arms.json`,
`results_density_omission.json`, `results_density_gold.json`
Parent: `exp-2026-08-16-pyannote-transcription` (CLOSED),
[report](2026-08-16-pyannote-transcription.md)

Two things that experiment left on the table. One is a measured positive it never
shipped; the other is a rule it deliberately did not try, because trying it would have
meant choosing a threshold after seeing the numbers.

## Answers, short

1. **The overlap speaker-cut advantage was not demonstrated on top of W.** The
   preregistered primary — speaker cuts minus a dose-matched placebo, inside the
   overlap mask — came out **+0.00094, CI [−0.00600, +0.00833]**, which includes zero
   *and includes the parent experiment's −0.00558*. This is a **failed demonstration,
   not a demonstrated failure**: the design had roughly a one-in-three chance of
   detecting an effect that size. It does rule out a large benefit (see §2.6).
2. **Patching W inside overlap made things worse, not better.** The three
   selection-shaped arms cost between +0.0022 and +0.0036 WER against W with CIs
   excluding zero; the two composition-shaped arms are statistically unresolved
   (+0.0002, CI includes zero). All five failed the search screen. **Zero of five
   confirmations spent; all five remain.**
3. **The per-speaker omission rule works, and its speaker term is doing real work.**
   Recall goes **0.1075 → 0.2020** (+0.0945 [+0.0581, +0.1323], CI excludes zero) and,
   against a duration-only detector held to the same alert budget, precision is
   **+0.0543 [+0.0042, +0.1089]**, CI excludes zero. That is the preregistered primary
   and it is the first evidence in this project that *how many people are talking*
   carries information a duration rule does not have.
4. **The price is alert volume, and the precision loss is unresolved.** 243 merged
   flags instead of 107, at an observed precision 5.3 points lower whose CI contains
   zero. Higher-sensitivity, higher-workload operating point — not a free upgrade.
5. **The gold set claims nothing.** 4 flags in 27 cells, below the threshold the
   preregistration set in advance for saying anything at all. The withdrawn "lower
   bound" label on the detector's precision stays withdrawn.

## Denominator, stated up front

| | |
|---|---|
| arms designed and preregistered | 5 |
| arms evaluated | 5 |
| arms refused as cosmetic variants | 0 |
| arms through the search screen | **0** |
| confirmation batches frozen | **0** |
| confirmations spent | **0** of 5 |
| detector variants scored | 6 (old rule × 2 timelines, new rule × 3 thresholds × 2 timelines, duration-only comparator) |
| new pyannoteAI API calls | **0** — both timelines for all 247 windows were already on disk |
| GPU | **0** |

The autoresearch journal now stands at 16 registered / 16 searched / 1 duplicate
refused / 0 through the screen, cumulative over this run and the harness's own first
run.

## Sample and discipline

**247 two-minute windows**, 144 meetings, 10 cities, run
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`, with the 6 sealed
temporal-holdout windows of `eval-freeze-2026-08` removed by `load_substrate`'s
explicit filter — which is why the substrate is 247 and not 253. The 16 locked
evaluation windows were never in reach.

Experiment 1 runs on the **six-city search partition** (153 windows, 103 meetings,
47,252 reference tokens, W = 0.09280). Experiment 2 is a detector, produces no fusion
output, and runs on all 247.

Every number is **agreement-with-OpenCouncil**. The gold-set section is
fidelity-to-audio and is kept apart from everything else.

### No confirmation was spent, and that was decided before anything ran

The hypothesis under test here — "speaker information helps inside detected overlap" —
was *generated* by the parent experiment's measurement on all 247 windows, the four
sealed confirmation cities included. Hiding those cities during implementation does not
make them unseen. Spending a one-way confirmation door on a hypothesis read off the
confirmation data would buy nothing, so §1.1 of the preregistration ruled it out in
advance, and the outcome does not change that: nothing cleared the screen anyway.

### What the preregistration had to fix before it could be run

Codex review `5851725675b5` (high effort) read the first draft and found twelve
defects. Four mattered enough to change the design:

- **The confirmation partition is not confirmatory for this hypothesis** (above).
- **The treated region was cut-dependent.** "Cells that contain overlap" moves with the
  cut set, and a speaker cut sits next to a handover by construction while a placebo cut
  does not — so the placebo would have replaced a systematically different amount of
  non-overlap text and could have lost without speaker information doing any work. The
  region is now fixed from the timeline alone, before any cut exists.
- **The first density threshold could not fire on the case it existed for.** "Flag when
  words-per-speaker-second drops below half the corpus rate" scores exactly 0.5ρ when
  two people speak and one is transcribed normally, and a strict `<` does not fire; with
  three speakers and one lost it is 2ρ/3, further away. The quantity had to become a
  **count of missing speakers**. This is now pinned by a test.
- **The detector's null tested a claim nobody disputes.** Beating random intervals only
  shows that low-output regions predict transcript deletions, which is nearly circular.
  The primary comparator became a duration-only detector at a matched alert budget.

## 1. The two constructions

### 1.1 The mask

For each maximal detected-overlap interval `O` (plain `diarization` with ≥ 2
simultaneous speakers), the mask spans from the start of the active interval before `O`
to the end of the active interval after it; overlapping members merge. It is the
overlap *plus its two neighbouring turns*, because that is where a handover cut lives —
restricting to the bare overlap would delete the mechanism under test, since what a
speaker cut carries is exactly *where A's words end and B's begin*.

It is computed from the timeline alone. **Every arm's output is token-for-token
identical to W outside it**, the mask endpoints are edges of every partition, and cut
sets change only how the interior is segmented.

Measured exposure on the search partition:

| | |
|---|---|
| reference tokens inside the mask | 6,411 of 47,252 (**13.6%**) |
| windows the mask touches | 96 of 153 |
| W tokens replaced, every arm | 6,918 (identical across arms by construction) |
| interior speaker cuts | 337 |
| windows with **zero** interior speaker cuts | **80 of 153** |
| windows where the placebo could not be dose-matched | **0** |

That fifth row is the most important one in this report. In 80 of 153 windows the
speaker partition places no cut inside the mask at all, so the treatment is inert there
and both arms emit the same tokens. The contrast lives in 73 windows and 43 meetings.

### 1.2 The five arms

| arm | interior of the mask | fill |
|---|---|---|
| `ov_mask_select` | no cuts | cell-local whole-system selection |
| `ov_turn_select` | speaker handover cuts | cell-local whole-system selection |
| `ov_turn_select_placebo` | equally many cuts from non-handover boundaries inside the same mask | as above |
| `ov_turn_compose` | speaker handover cuts | re-align the trio per cell, re-vote per column, pivot on the cell-local winner |
| `ov_turn_compose_placebo` | matched placebo cuts | as above |

Placebo draws are seeded with SHA-256 of the window id, so they survive a new process;
the placebo takes **exactly** as many interior cuts as the speaker partition does, and
refuses rather than under-dosing when the pool is short (it never had to refuse). The
placebo is evaluated over 20 draws and a bootstrap replicate resamples meetings *and*
which draw each window contributes, so its Monte-Carlo uncertainty is inside the
interval rather than conditioned away.

## 2. Experiment 1 — what happened

### 2.1 Mechanistic estimand: error inside the mask

Edit distance inside the mask, scored on **the mask's own partition for every arm** —
deliberately not each arm's own cells, because a finer partition constrains the
alignment and would penalise whichever arm carried more cuts. Denominator 6,411
reference tokens, identical for all rows.

| arm | error rate inside the mask |
|---|---|
| **W** | **0.23553** |
| `ov_turn_compose` | 0.23662 |
| `ov_turn_compose_placebo` | 0.23675 |
| `ov_turn_select_placebo` | 0.25097 |
| `ov_turn_select` | 0.25191 |
| `ov_mask_select` | 0.26018 |

W is the best thing in its own overlap neighbourhood. Nothing built here improved on it
there.

### 2.2 The preregistered primary

| contrast | δ | CI 95% | excludes 0 |
|---|---|---|---|
| **`ov_turn_select` − `ov_turn_select_placebo`** | **+0.00094** | **[−0.00600, +0.00833]** | **no** |
| `ov_turn_compose` − `ov_turn_compose_placebo` | −0.00012 | [−0.00444, +0.00437] | no |

103 meeting clusters, 43 with a non-zero contribution, largest single-meeting share of
the absolute effect 0.114 — no single-item domination.

The sign of the point estimate is *positive*, i.e. speaker cuts marginally worse than
placebo cuts, which is the direction the preregistration predicted against. But the
interval is wide enough to contain the parent experiment's −0.00558, so **the honest
statement is that transfer was not demonstrated, not that it was refuted.**

### 2.3 Operational estimand: whole-window WER against W

| arm | WER | ΔWER vs W | CI 95% | del | ins |
|---|---|---|---|---|---|
| **W** | **0.09280** | — | — | 0.01737 | 0.03486 |
| `ov_turn_compose` | 0.09301 | +0.00021 | [−0.00056, +0.00109] | 0.01786 | 0.03479 |
| `ov_turn_compose_placebo` | 0.09308 | +0.00028 | [−0.00048, +0.00110] | 0.01782 | 0.03475 |
| `ov_turn_select_placebo` | 0.09504 | +0.00224 | [+0.00101, +0.00363] | 0.01778 | 0.03604 |
| `ov_turn_select` | 0.09519 | +0.00239 | [+0.00125, +0.00372] | 0.01767 | 0.03640 |
| `ov_mask_select` | 0.09636 | +0.00356 | [+0.00207, +0.00527] | 0.01765 | 0.03725 |

All five point estimates are non-improving and all five fail the search screen.
**Detectable degradation is confined to the selection arms**; the two composition arms
are statistically unresolved against W and their intervals still contain a small
improvement. The deletion rate rises slightly in every arm, so none of this is a case
of a WER that improved by dropping hard passages.

Whole-window turn minus placebo, restricted to matched windows: **+0.00015
[−0.00071, +0.00099]** — same story, tighter interval, same conclusion.

### 2.4 What the three preregistered predictions did

| prediction | outcome |
|---|---|
| 1. the difference of differences is negative | **failed** — +0.00094, interval spans zero |
| 2. the arms are worse than or indistinguishable from W | **held** |
| 3. `ov_turn_compose` is closer to W than `ov_turn_select` | **held**, descriptively — no direct compose-vs-select contrast was run, so this is not an inferential claim |

Prediction 1 was the one this experiment existed to test, and it is the one that failed.

### 2.5 The mechanism, as far as this can see it

The parent experiment's positive was measured **on top of whole-window selection**
(0.1201), scored cell-locally on overlap cells. W is a per-column composition at
0.09280 — a full point better — and it is already better than any of these arms inside
the mask. The reading this run supports is that composition has *already collected*
whatever the speaker cut was buying at selection level, and that cutting W's stream at
speaker boundaries and handing the piece to one system throws away more than the cut is
worth. That reading is consistent with everything measured here and is **not
established** by it.

### 2.6 How much this design could have seen

Two-sided normal approximation from the primary interval: SE ≈ 0.00366, so the
**minimum detectable effect at 80% power is about 0.0103** on this estimand and about
0.0119 at 90%. If the parent's −0.00558 transferred one-for-one, two-sided power here
was about **33%** (about 45% one-sided). These are post-hoc design-sensitivity
approximations, not an exact clustered MDE — 43 informative meetings out of 103 nominal
clusters is the constraint that actually binds.

So: a *large* benefit inside the mask is ruled out. An effect of the size previously
reported is not, in either direction.

Nor are the two estimands the same quantity. The parent measured exact-overlap cells,
all 247 windows, on top of selection; this measures a mask that also holds the two
neighbouring turns, on six cities, on top of composition. Their magnitudes must not be
subtracted from each other.

## 3. Experiment 2 — the omission rule that can count speakers

### 3.1 The rule

Hybrid, because the two regimes are different problems:

- **outside detected overlap** — the shipped rule unchanged: an active interval of
  ≥ 1.5 s with none of our words in it.
- **inside detected overlap** — `missing(I) = |S| − obs(I) / (ρ_single × dur(I))`, flag
  iff `missing(I) ≥ 1.0`. Inclusive, so the idealised one-lost-speaker case fires
  exactly at the boundary. Eligibility inside overlap is `ρ_single × dur(I) ≥ 3`: a lost
  speaker too short to have produced a three-word deletion run cannot produce a scorable
  truth event either, and 3 is `MIN_DEL_RUN`, not a new free constant.

`ρ_single` is estimated on **single-speaker intervals only**, leave-one-city-out — 2.429
to 2.454 tokens per second across the ten folds, pooled 2.438. Overlap intervals are
excluded from it on purpose: the omissions being hunted live there and would depress the
rate that is supposed to detect them.

Event scoring is frozen and stricter than the parent's: adjacent flags are **merged**
before scoring and matching is **one-to-one** against maximal deletion runs. Without
merging, one omission split across three adjacent active intervals would have counted as
three true positives.

### 3.2 The numbers, on the regular timeline, 247 windows, 307 truth events

| rule | merged flags | flagged sec | precision | recall |
|---|---|---|---|---|
| old rule (shipped) | 107 | 263.9 | 0.3084 | 0.1075 |
| **speaker-aware, threshold 1.0 (primary)** | **243** | 550.2 | **0.2551** | **0.2020** |
| speaker-aware, threshold 0.75 | 274 | 621.3 | 0.2336 | 0.2085 |
| speaker-aware, threshold 1.25 | 208 | 473.4 | 0.2596 | 0.1759 |
| duration-only comparator, budget-matched | 244 | 618.2 | 0.2008 | 0.1596 |

146 of the 243 primary flags touch a detected-overlap interval. There are only **214
eligible overlap intervals** in the whole substrate against 6,425 eligible intervals
overall, so the new behaviour is concentrated in a small part of the timeline.

**The old rule's 0.3084 here is not the 0.361 the parent reported.** Same rule, same
time axis, same windows — the difference is entirely the stricter event scoring above.
It is not degradation and not a change over time, and the two must not be quoted as a
before-and-after.

### 3.3 The preregistered primary: does speaker count add anything?

The comparator is the same rule with `|S|` forced to 1 — a pure duration-and-density
under-transcription detector — on **exactly the same 6,425 eligible intervals**
(eligibility does not depend on `|S|`, only the score does), with its threshold
calibrated to emit the same number of merged flags (244 against 243).

| contrast | δ | CI 95% | excludes 0 |
|---|---|---|---|
| **precision, speaker-aware − duration-only** | **+0.0543** | **[+0.0042, +0.1089]** | **yes** |
| recall, speaker-aware − duration-only | +0.0423 | [−0.0032, +0.0847] | no |
| precision, speaker-aware − old rule | −0.0533 | [−0.1286, +0.0159] | no |
| recall, speaker-aware − old rule | +0.0945 | [+0.0581, +0.1323] | **yes** |

144 meeting clusters; 81 of them contribute a differing result; the largest
leave-one-meeting-out shift on the primary is +0.0093 on a +0.0543 effect (17%), so no
single meeting carries it. The comparator uses **more** flagged seconds than the
speaker-aware rule (618 vs 550), so it is not disadvantaged on workload.

What this licenses, and no more: *at a matched merged-alert count, adding detected
speaker multiplicity to this density rule improves event precision over this
duration-only comparator by 5.4 points.* `|S|` is **detected** multiplicity, not the
true speaker count; the rule differs from its comparator only inside detected overlap,
so `|S|` may be acting largely as an overlap-conditioned threshold shift; and the
comparator is one scalar rule, not the best duration-only detector obtainable.

### 3.4 The operational statement

The hybrid rule raises recall from **10.8% to 20.2%** — 62 matched events instead of 33,
a gain of 9.45 points whose CI excludes zero — while producing **2.27× as many merged
flags** (243 against 107) and 2.1× the flagged seconds. Its observed precision is 5.3
points lower and **that difference is statistically unresolved**. It is a
higher-sensitivity, higher-workload operating point, not a free upgrade, and this
comparison does **not** show that speaker awareness caused the recall gain at equal
workload — the budget-matched comparator of §3.3 is the evidence for incremental
discrimination, and it is about precision.

The threshold sensitivities behave monotonically and none of them changes the reading:
0.75 buys 0.5 points of recall for 2.2 points of precision and 31 more alerts; 1.25
gives back 2.6 points of recall for 0.5 points of precision.

### 3.5 Secondary nulls

Chance-localisation benchmarks, not proofs of specificity: a caliper-matched random draw
(same window, duration within ±25%, flagged intervals excluded from the pool) hits
precision **0.0441**, with 18.8 of 243 flags per draw finding no match and being dropped;
the parent's unstratified matched null hits **0.0811**. Both are far below every rule in
the table, which is reassuring and is the least interesting thing here.

### 3.6 On the exclusive timeline the new rule is the old rule

`exclusiveDiarization` has resolved every overlap to one speaker, so `|S| = 1`
everywhere and the hybrid degenerates by construction: 122 flags, precision 0.3361,
recall 0.1336 — byte-for-byte the old rule's numbers on that timeline. That is a
construction check, not a performance comparison, and the regular and exclusive
timelines' precisions are not comparable to each other.

### 3.7 The gold set — 4 flags, and nothing claimed

Run over the 27 frozen gold cells (6 meetings, 6 cities, 6.75 scored minutes), with
`ρ_single` frozen at the 247-window pooled value and the adapter's own word timestamps
as the axis:

| | |
|---|---|
| flags, new rule | **4** (in 4 distinct cells) |
| flags, old rule | 2 |
| new flags intersecting human-certain speech | 4 |
| new flags intersecting an overlap-marked block | 1 |
| eligible overlap intervals in the whole gold core | 3 |

The preregistration said, before this ran, that fewer than 5 flags claims nothing. Four
is fewer than five. **No floor is put under the detector's precision, no bound is
computed, and the "lower bound" label that `exp-2026-08-16-gold-set` withdrew stays
withdrawn.** "Intersects human-certain speech" is also not the same event as "correctly
detects a ≥3-word omission", so even the 4-of-4 is a sanity check and not a precision.
The axis differs from the 247-window run, so none of these numbers may be compared with
§3.2.

## Caveats

- **No independent confirmation exists for either result.** Experiment 1 is preregistered
  search evidence on six cities; Experiment 2 is a sixth pass over the same 247 windows
  by an agent that has read the prior reports. Sample splitting controls the multiplicity
  of testing, not the adaptivity of proposing.
- **Experiment 1's null is a failed demonstration, not a demonstrated failure.** Its
  interval contains the effect it was built to reproduce.
- **The three Experiment 1 estimands are different quantities** — the parent's
  overlap-local selection contrast, the mask-local error rate here, and whole-window WER.
  Their magnitudes must not be compared.
- **Everything except §3.7 is agreement-with-OpenCouncil.** A detector "true positive"
  means our text disagrees with the published reference, not that a human heard speech
  there. The parent's precision figure is a lower bound only if the reference is
  complete, which `exp-2026-08-16-gold-set` showed it is not — and which this run does
  not re-establish.
- **The duration-only comparator's threshold was calibrated on the evaluation corpus**
  to match the alert budget, and the bootstrap treats that threshold as fixed rather than
  recalibrating inside each replicate. The +0.0543 is conditional on that calibration.
- **The Experiment 2 contrast rests on 214 overlap intervals.** That is the whole
  support for the speaker term.
- **The time axis is interpolated**, anchored on the whisper-turbo word stream: coverage
  0.880 for our tokens and 0.867 for reference tokens on the 247, 0.872 / 0.878 for W and
  the reference on the search partition. Absolute precision and recall are conditional on
  that interpolation; the paired comparisons share it and are fairer than the absolutes.
- **A truth event with a zero-length span cannot intersect anything** under the strict
  overlap predicate inherited from the parent, so a single-instant deletion run is
  unmatchable by construction. Unchanged from the parent, and unchanged on purpose.
- **A 12-window smoke test was run before the full search**, to check runtime and mask
  exposure. Its numbers were visible and **no definition was changed after seeing them**;
  the mask exposure it suggested (40%) was unrepresentative of the real 13.6%.
- Two Codex reviews (high effort) shaped this. `5851725675b5` ran before any code and
  changed the confirmation status, the mask, the placebo dose matching and the entire
  density-threshold design; `8c1e7bbb2020` ran on the findings and before this text, and
  killed "the positive does not transfer", "all arms are worse than W", "knowing how many
  people are talking adds information" as written, and "the old rule has better
  precision".

## What remains

- The speaker-aware omission rule is the only shippable thing here, and what it feeds is
  candidate mining for `exp-2026-08-13-targeted-deletion-training`: 243 alerts of which
  roughly 62 are real multi-word omissions, with no human and no reference. Whether the
  extra recall is worth 2.3× the alerts is an operating-point decision, not a
  measurement.
- Experiment 1's question is still open at the effect size that matters. Answering it
  would need either far more overlap material than 214 intervals and 43 informative
  meetings, or an estimand that is not diluted by the 80 windows where the treatment is
  inert.
- Nothing here touches fidelity-to-audio. The gold set remains the only place that does,
  and it is too small to speak about this detector.
