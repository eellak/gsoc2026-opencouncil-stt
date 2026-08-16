# Preregistration: overlap-restricted speaker-conditioned arms, and a per-speaker density omission rule

2026-08-16 · written **before any number in either experiment existed** ·
revised once, before any run, on Codex review `5851725675b5` (high effort) ·
parent record `exp-2026-08-16-pyannote-transcription` (CLOSED) ·
evidence [`docs/reports/2026-08-16-pyannote-transcription.md`](../reports/2026-08-16-pyannote-transcription.md)

Two things that experiment measured but never shipped, and one thing it deliberately
did not try. Both are threshold-shaped, which is why this document exists: choosing
either threshold after seeing a WER or a precision would be selection on the outcome,
and this project's expensive mistakes have all been measurement mistakes.

**Twelve blocking defects were found in the first draft of this document, before any
number existed.** The four that mattered most: the confirmation partition is not
confirmatory for this hypothesis; the treated region was cut-dependent, so the placebo
could be made to lose by replacing more text; the density threshold as first written
**mathematically cannot flag the exact case it was designed for**; and the detector's
random null tests a claim nobody disputes. All four are corrected below and the
original wording is not preserved anywhere as if it had been fine.

## 0. What is already known, and is not re-litigated here

- **Speaker cuts are worse than random cuts overall.** Round 2 of the parent
  experiment: per-turn selection beats per-window by −0.0036 (CI excludes zero), but a
  placebo of equally many cuts at non-speaker boundaries beats it by −0.0056. Turn
  minus placebo = **+0.0020 [+0.0004, +0.0037]**. What wins is locality, not speaker
  identity.
- **Inside detected overlap the sign reverses.** On the 823 cells containing detected
  overlap (24,079 reference tokens): turn minus placebo = **−0.00558
  [−0.00888, −0.00228]**, CI excludes zero, 166 wins / 56 ties / 96 losses.
- Independent corroboration from the human gold set, on different audio and against a
  human who listened: speaker recall 0.815 overall but **0.714 inside overlap**; of 38
  overlap blocks, 12 have none of their words in the published transcript and 23 are
  collapsed onto their interlocutor's speaker.
- **The "no words of ours here" omission rule cannot see one lost speaker inside
  simultaneous speech**: in an interval with active set {A, B}, a single recognised
  word of A blocks the flag even if the whole of B's turn was lost. That is a
  structural fact about the rule, established in round 2, not a hypothesis.

**Every speaker-conditioned arm below carries a placebo, and no number from it may be
reported without its placebo beside it.** That control already reversed one headline
in this exact experiment.

## 1. Substrate, and the status of the confirmation partition

247 two-minute windows, 144 meetings, 10 cities, run
`2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`, loaded through
`fusion_lab.load_substrate`. The 16 locked evaluation windows are sealed and out of
reach; the 6 sealed temporal-holdout windows of `eval-freeze-2026-08` that sit inside
the 253 common windows are removed by that loader's explicit filter, which is why the
substrate is **247 and not 253**. No number here is compared against a number computed
on a different window set.

### 1.1 No confirmation is spent, and why

The autoresearch protocol
([`2026-08-16-autoresearch-partition-prereg.md`](2026-08-16-autoresearch-partition-prereg.md))
splits the ten cities into six search and four confirmation cities, and offers five
one-way confirmation slots. **This experiment spends none of them, and would spend none
even if an arm cleared the search screen.**

The hypothesis being tested here — "speaker information helps inside detected overlap"
— was *generated* by the parent experiment's measurement on **all 247 windows,
confirmation cities included**. Hiding four cities during implementation does not make
them unseen. Charging the confirmation budget for a hypothesis that was read off the
confirmation data would spend a one-way door on nothing. Codex `5851725675b5` is the
reason this paragraph exists.

Consequence, declared in advance: **every Experiment 1 number is exploratory and
estimation-only.** It runs on the search partition, through the harness, so that it
lands in the journal with a denominator and under the frozen gates — not because the
gates confer confirmatory status here. If an arm clears the search screen, the report
says so and stops; freezing a batch on it is a decision for a human with fresh data.

Experiment 2 is a detector, not an arm on W. It produces no fusion output, consumes no
confirmation budget, and is scored on all 247 windows exactly as its predecessor was.

Every WER below is **agreement-with-OpenCouncil**. The gold set is the only place in
this project with fidelity-to-audio truth, and where it appears (§3.7) it is labelled
and never merged with anything else.

## 2. Experiment 1 — overlap-restricted speaker-conditioned composition

### 2.1 Definitions taken unchanged from the parent experiment

Every object below comes from `eval/controlled_eval/exp_speaker_fusion.py`, which was
reviewed twice by Codex and locked with 15 tests. Reusing it introduces no new
threshold that could be tuned.

| object | definition | source |
|---|---|---|
| timeline | pyannoteAI precision-2 plain `diarization` (**not** `exclusiveDiarization`) | round 2, primary |
| active intervals | maximal half-open intervals of constant speaker *multiset* | `active_intervals` |
| detected overlap | active intervals with ≥ 2 simultaneous speakers | `overlap_intervals` |
| speaker cuts | one cut per floor handover, including the overlap-mediated `{A}→{A,B}→{B}` form | `handover_cuts` |
| placebo pool | active-set change times that are **not** part of a handover transition | `other_boundary_times` |
| token time axis | linear interpolation between words shared with the whisper-turbo word stream | `token_times` |
| cell-local selection | argmax summed pairwise difflib similarity, frozen tie-break `scribe > soniox > adapter` | `pick_by_similarity` |

**No minimum-overlap-duration threshold is introduced.** An active interval is overlap
iff it carries ≥ 2 simultaneous speakers, exactly as in round 2.

### 2.2 The treated region is fixed before the cuts, and is the same for every arm

This is the correction that matters most. If the treated region were "cells that
contain overlap", it would depend on where the cuts fall: a speaker cut sits near a
handover by construction and a placebo cut does not, so the placebo would replace a
systematically different amount of non-overlap text and could be made to lose without
speaker information doing any work at all.

So the region is defined **from the diarization timeline alone, before any cut set
exists**, and is identical across all arms:

> For each maximal detected-overlap interval `O`, let `M(O)` span from the **start of
> the active interval immediately preceding `O`** to the **end of the active interval
> immediately following `O`** (clipped to the window when a neighbour does not exist).
> `M` is the union of all `M(O)`, with touching or overlapping members merged.

`M` is the overlap plus its two neighbouring turns. It is what a handover cut lives
inside: for the canonical `{A} → {A,B} → {B}`, the handover cut falls at the boundary
between the overlap and `{B}`, which is strictly interior to `M`. Restricting treatment
to the bare overlap interval instead would delete the mechanism under test, because the
information a speaker cut carries is precisely *where A's words end and B's begin*.

**Every arm's output is token-for-token identical to W outside `M`.** The endpoints of
`M` are added to every arm's cut set, so no cell ever straddles the boundary of `M`.
Cut sets change only how `M`'s interior is segmented.

### 2.3 The arms

Baseline **W**, the per-column composition of `exp-2026-08-16-composition-over-selection`
(0.10046 on the 247; 0.09280 on the search partition). Write `cuts∩M` for the cut times
falling strictly inside `M`.

| arm | cells inside `M` | what is emitted inside `M` |
|---|---|---|
| `ov_mask_select` | the maximal spans of `M`, no interior cuts | the cell-local `pick_by_similarity` winner's tokens |
| `ov_turn_select` | split further at `speaker cuts ∩ M` | the cell-local `pick_by_similarity` winner's tokens |
| `ov_turn_select_placebo` | split further at matched placebo cuts (§2.4) | the cell-local `pick_by_similarity` winner's tokens |
| `ov_turn_compose` | split further at `speaker cuts ∩ M` | the trio's three cell slices re-aligned with `msa.align3` and re-voted with `msa.compose`, pivoting on the cell-local `pick_by_similarity` winner |
| `ov_turn_compose_placebo` | split further at matched placebo cuts | as `ov_turn_compose` |

`ov_mask_select` is the locality-free ablation: it separates "restrict attention to the
overlap neighbourhood" from "cut at the speaker handover", and it has no placebo
because it permutes nothing.

**Five arms. That is the denominator for Experiment 1, and it is restated in the report
whatever the outcome.**

### 2.4 The placebo is matched on dose inside `M`, not on cut count per window

Matching total cuts per window does not match the treatment: only cuts inside `M`
change anything. So:

- The placebo draws **exactly `|speaker cuts ∩ M|` cuts, without replacement, from
  `placebo pool ∩ M`**, per window.
- If `|placebo pool ∩ M| < |speaker cuts ∩ M|`, the window **cannot be matched**. It is
  **excluded from the primary paired difference of differences**, counted as
  `unmatched_windows`, and reported. It is *not* silently given a smaller placebo: an
  under-dosed placebo in exactly the hardest windows is how a fake effect gets made.
- Windows with `|speaker cuts ∩ M| = 0` are matched trivially (both arms take zero
  interior cuts and produce identical output). They are counted as
  `zero_dose_windows`, contribute exactly zero to the difference of differences, and
  are reported.
- Draws are seeded by `SHA-256(window_id ‖ arm_name ‖ draw_index)`, never by the
  language runtime's `hash`, so the arms are reproducible across processes.

### 2.5 Two estimands, stated separately, because whole-window WER dilutes

`M` is a small share of the material. A null on whole-window WER could mean the
mechanism is absent or merely that 95%+ of the tokens are untouched noise; and a global
Levenshtein realignment can move errors across a splice boundary, so a whole-window
*gain* is not by itself proof that overlap transcription improved. Both are therefore
reported, and they answer different questions.

**MECHANISTIC (the speaker-information test).** Error rate on `M` only: summed edit
distance between each arm's output and the reference tokens, **within each maximal span
of `M`**, divided by the reference tokens inside `M`. The scoring partition is the mask
itself and is the *same for every arm* — deliberately not each arm's own cells, because
a finer partition constrains the alignment and would penalise whichever arm happens to
carry more cuts. Every arm is scored on the identical token set; only the selection
inside differs. This is the round-2 overlap contrast on a region that no longer depends
on the cuts.

**OPERATIONAL (shipping impact).** Whole-window out-of-fold WER through
`fusion_lab.evaluate` and the autoresearch harness: WER, del, ins, sub, both rate
gates, percentile clustered CI, wild-cluster p-values, meetings touched, single-item
domination, leave-one-out sign flips, per-city deltas.

Both estimands use **ratio of sums** — `Σ(S+D+I) / ΣN` — inside every bootstrap
replicate, never a mean of per-window rates.

### 2.6 The primary comparison, and the rejection rule

**ONE primary comparison**, designated before any run:

> `ov_turn_select` − `ov_turn_select_placebo`, on the **mechanistic** estimand,
> restricted to matched windows, on the search partition.

Everything else is secondary and labelled as such: the same contrast on the operational
estimand, the whole `ov_turn_compose` family, and `ov_mask_select`. Designating one
primary rather than Holm-correcting two families is the simpler honest choice and is
made here, in advance.

**Rejection rule**: two-sided 95% paired meeting-clustered bootstrap CI excluding zero,
10,000 replicates, seed 7, meetings resampled. The placebo is evaluated over
**N_PLACEBO = 20 draws**; a bootstrap replicate resamples meetings *and*, independently
per window, which of the 20 draws that window contributes. The placebo arm is therefore
reported as **expected placebo performance**, with its Monte-Carlo uncertainty inside
the interval rather than conditioned away. The single registered draw (index 1) is what
the harness journals, and the two are reported side by side.

**Directional predictions, written before any run.**

1. The primary difference of differences is predicted **negative** — speaker cuts beat
   placebo cuts inside `M`. That is the transfer being tested.
2. Both `ov_turn_*` arms are predicted **worse than or indistinguishable from W** on
   the operational estimand, because W (0.1005) is a full point better than
   whole-window selection (0.1201) and `M` is a small share of the material. An arm
   that beats W here would be a surprise.
3. `ov_turn_compose` is predicted closer to W than `ov_turn_select`, because it keeps
   composing instead of handing the cell to one system.

Prediction 2 is why this is not a shipping proposal. If the difference of differences
is negative while both arms lose to W, the conclusion is *speaker information helps
inside overlap and is still not worth shipping on this substrate*, written in those
words.

### 2.7 Reported whatever happens

Replaced-token mass per arm and per placebo; `unmatched_windows` and
`zero_dose_windows`; deletion rate beside every WER; head-to-head window counts;
single-item domination on every quoted delta; anchor coverage, longest interpolation
gap and extrapolated-token mass for the time axis, since W is a synthetic text and a
timing failure is a real measurement channel.

The run **aborts** if a cached diarization response is missing. It does not fall back
to the API.

## 3. Experiment 2 — a per-speaker density omission rule

### 3.1 The rule the parent experiment refused to try, and why the obvious version fails

Current rule — precision 0.361 / recall 0.107 on the regular timeline under round 2's
whisper-turbo time axis: *flag an interval of asserted speech ≥ 1.5 s inside which our
transcript has no words at all.* Its blind spot is one lost speaker inside simultaneous
speech.

The obvious generalisation is "flag when words-per-speaker-second falls below half the
corpus rate". **That rule provably cannot flag the case it exists for.** With two
speakers active for `d` seconds and exactly one of them transcribed at the normal rate
`ρ`, the density is `ρd / 2d = 0.5ρ` — the strict inequality `< 0.5ρ` does not fire.
With three speakers and one lost it is `2ρ/3`, further from firing. Codex
`5851725675b5` found this before it was run. The threshold is therefore not a density
fraction but a **count of missing speakers**.

### 3.2 The rule, in the form that targets the stated failure

A **hybrid**, because the two regimes are different problems and merging them would let
a general "low output" detector borrow credit for overlap work:

**Outside detected overlap** (active intervals with exactly one speaker) — the old rule,
unchanged, wall-clock eligibility:

> flag iff `dur(I) ≥ 1.5 s` and no token of ours has a time inside `I`.

**Inside detected overlap** (active intervals with `|S| ≥ 2`) — missing-speaker
equivalents:

> `missing(I) = |S| − obs(I) / (ρ_single × dur(I))`, flag iff **`missing(I) ≥ 1.0`**.

where `obs(I)` is the number of our tokens whose interpolated time falls in `I`,
`|S|` is the number of simultaneously active speakers, and `ρ_single` is the
single-speaker token rate of §3.3. The inequality is **inclusive**, so the idealised
one-lost-speaker case flags exactly at the boundary. That is the whole design.

**Eligibility inside overlap**: `ρ_single × dur(I) ≥ 3`, i.e. one lost speaker would
have to account for at least three words. Three is `MIN_DEL_RUN`, the length of the
shortest deletion run that counts as ground truth — the eligibility rule is tied to the
truth definition rather than to a new free constant.

**Primary threshold `missing ≥ 1.0`.** Prespecified sensitivity: `0.75` and `1.25`,
reported beside the primary and never in place of it. If some other threshold looks
better after the numbers, the primary stays 1.0 and the alternative is labelled
exploration in its own paragraph.

Tokens are `scoring.wtoks` output throughout — the same normaliser as every WER in this
project. "Words" and "tokens" mean that and nothing else.

### 3.3 `ρ_single` is cross-fitted and estimated off single-speaker speech only

`ρ_single` = (our tokens landing in single-speaker eligible intervals) / (seconds of
those intervals), pooled — **leave-one-city-out**, so no meeting influences its own
threshold. Estimating it from overlap intervals too would be self-depressing: the very
omissions being hunted would lower the rate and make further omissions harder to flag.
`ρ_single` never touches the reference, the deletion runs, precision or recall, so it
cannot be selected on the outcome. Its ten fold values are reported.

### 3.4 Ground truth, event scoring, and the time axis — all frozen here

- **Truth events**: maximal runs of ≥ 3 consecutive `D` in the Levenshtein backtrace of
  the benchmark reference against OURS (`oc-runpod-fixed-2026-08-10`), timed by the
  interpolated reference-token times of their first and last token. Maximal by
  construction, so no merging is needed.
- **Predicted events**: flagged intervals, with **touching or overlapping flags merged
  into maximal spans before scoring**. Without merging, one omission split across three
  adjacent active intervals would count as three true positives and inflate precision.
- **Matching**: strict temporal overlap (`a.start < b.end and b.start < a.end`), matched
  **one-to-one**, greedily by earliest predicted-event start, each truth event
  consumable once. `TP` = matched pairs; `precision = TP / #merged flags`;
  `recall = TP / #truth events`; both as ratios of corpus sums, not means of
  per-window ratios.
- **Time axis**: interpolation against the **whisper-turbo** word stream, so these
  numbers are comparable to round 2's A6 (regular 0.361 / 0.107, exclusive
  0.376 / 0.134) and **not** to round 1's 0.320, whose axis came from Parakeet.
  Deletion tokens have no hypothesis timestamp by definition and get their time from
  that same interpolation, extrapolated at the anchor-implied rate at the edges — the
  existing frozen `token_times`, unchanged.
- **Timeline**: the plain `diarization` is primary — it is the only one with an overlap
  to be dense in. `exclusiveDiarization` is secondary.

### 3.5 The primary comparator is a duration-only detector, not a coin

Beating random intervals would prove only that low-output regions predict transcript
deletions, which nobody disputes and which is nearly circular: both quantities are
functions of missing tokens of ours. The question is narrower — **does knowing how many
people are talking add anything?**

> **PRIMARY COMPARATOR: the same rule with `|S|` forced to 1 everywhere** — a pure
> duration-based under-transcription detector — with its threshold set so that it emits
> **the same number of merged flags** as the speaker-aware rule over the corpus (ties
> broken towards fewer flags). Same eligible intervals, same axis, same truth events,
> same budget. Paired meeting-clustered bootstrap on the precision difference.

Secondary: a matched-random null over eligible intervals, **excluding flagged intervals
from the pool**, matched on exact speaker cardinality and on duration within a ±25%
caliper, within the same window; flags for which no match exists are dropped from that
contrast and counted. 200 draws, seed 7. Round 2's unstratified matched null is
reported beside it because it is the published comparator.

**Directional prediction, written before any run**: recall rises above 0.107 and
precision falls below 0.361. The rule earns its place only if it beats the duration-only
detector at equal budget; if it does not, the honest reading is that speaker count adds
nothing and the old rule stands.

**Watched and reported regardless**: number of merged flags (a rule that flags a third
of all speech has no operational value at any precision), flagged seconds, and the
share of flags originating in overlap intervals — that share is the entire point of the
change.

### 3.6 The old rule, unchanged, in the same run

Computed on the same axis, same windows, same event scoring, so that every comparison
is internal. Its round-2 published numbers are quoted only as provenance.

### 3.7 The gold set is descriptive, and no floor is claimed from it

The parent experiment labelled precision 0.320 a **lower bound** because its ground
truth is agreement-with-OpenCouncil: a span our own reference also omits counts as a
false positive. `exp-2026-08-16-gold-set` **withdrew that label as unproven** — exactly
one flag appeared in its 27 cells and it was adjudicated non-speech.

The new rule is run over the 27 frozen gold cells (6 meetings, 6 cities, 6.75 minutes
of scored core) and the flag count reported. **Declared in advance: this is descriptive
whatever the count.** 27 cells of 15 seconds is not a probability sample of anything,
so no population lower bound is computed, no confidence-bound method is applied to it,
and the withdrawn "lower bound" label is **not** restored. What is reported is the flag
count, how the human adjudication falls, and the meeting distribution. No gold-set
number is merged with any agreement-with-OpenCouncil number.

## 4. Cost, and what may not be spent

- pyannoteAI is on a **free trial** as of 2026-08-16, and both timelines for all 247
  windows are already on disk and byte-identical between rounds 1 and 2 (247/247,
  round-2 measurement A5). **Both experiments are expected to make zero new API calls**;
  actual usage is reported either way. A missing cache aborts the run.
- **Zero GPU.** No pod is created for any part of this.
- No other paid API is called.

## 5. The tests that stand in for a proof of implementation

Written before the implementation, following Codex `5851725675b5`. All four must pass
before any number is read.

1. **Fixed-mask splice and token conservation.** A synthetic window with one short
   overlap and speaker/placebo cut sets that induce very different cells, with sentinel
   W tokens immediately outside `M`. Every arm must reproduce every outside token
   exactly once and in order, must replace only the atomic spans inside `M`, and must
   place each boundary token in exactly one cell.
2. **Placebo matching and determinism.** The draw is identical across processes and
   `PYTHONHASHSEED` values; no duplicate cut times; the number of placebo cuts inside
   `M` equals the number of speaker cuts inside `M`; and a short pool raises the
   documented unmatched signal rather than silently producing a smaller placebo.
3. **Detector semantics.** With `ρ_single` fixed: a single-speaker interval behaves
   exactly like the old zero-word rule (including the 1.5 s wall-clock eligibility); a
   two-speaker interval with exactly one speaker's worth of tokens flags at the
   boundary (`missing == 1.0`, inclusive); a three-speaker interval with two speakers'
   worth does not; and the eligibility rule `ρ_single × dur ≥ 3` is enforced on the
   half-open boundary.
4. **Event scoring.** Two synthetic meetings in which two adjacent flags cover one
   deletion run and a third flag matches nothing: merging happens before matching, the
   matching is one-to-one, TP/FP/FN are exact, precision and recall are ratios of sums,
   flagged intervals are excluded from the null pool, and meetings are resampled
   jointly for flags and nulls.

## 6. Honesty rules carried from the project protocol

- Denominators stated: arms registered, arms evaluated, arms refused as duplicates,
  arms through the screen, confirmations spent (zero, by §1.1), confirmations remaining.
- Single-item domination checked before any delta is quoted.
- Deletion rate reported beside every WER.
- Fidelity-to-audio and agreement-with-OpenCouncil are never merged.
- The 16 locked evaluation windows stay sealed; the 6 sealed holdout windows stay out
  of the substrate.
- Transcript text and audio never enter git. Every intermediate lives under `$SC`.
