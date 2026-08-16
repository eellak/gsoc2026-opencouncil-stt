# An idea loop that cannot quietly manufacture a result

2026-08-16 · wayfinder [#24](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/24) follow-on ·
prereg [`2026-08-16-autoresearch-partition-prereg.md`](../specs/2026-08-16-autoresearch-partition-prereg.md) ·
Codex review jobs `362e2a7b` (design) and `59c9564` (implementation) ·
protocol `autoresearch-2026-08-16b` · sixth pass over the same 247 windows

**Denominator first, because a leaderboard without one is dishonest: 11 ideas
registered, 11 evaluated on the search partition, 1 refused as a cosmetic variant of
an earlier one, 0 passed the search screen, 0 confirmations spent of a budget of 5.
The confirmation partition has not been read.**

This is a **smoke test of the harness, not a result about the model**. Nothing here
says anything new about whether W can be beaten. What it does establish is that the
machinery for asking that question eighty times cannot answer it by accident.

## The problem the harness is built against

`fusion_lab.evaluate` costs about ten seconds of CPU per idea and hands back a paired
clustered bootstrap CI. Those two facts are a machine for false results. Simulated
under the exact null — pure-noise ideas, unequal meeting sizes, the same inference
code the harness uses, 400 trials per row:

| ideas tried | P(some idea has raw p < 0.05) | P(some idea ships under Holm) |
|---:|---:|---:|
| 5 | 0.207 | 0.060 |
| 10 | 0.415 | 0.055 |
| 40 | **0.873** | 0.055 |

The marginal p-values are calibrated (P(p<0.05) = 0.046 / 0.054 / 0.051 across the
three rows; P(p<0.25) = 0.234 / 0.257 / 0.252), which is the precondition for any
multiplicity correction to mean anything. A loop of forty ideas without a correction
finds "a winner" seven times in eight. The same loop with the correction ships noise
about one cycle in twenty, which is what α = 0.05 was supposed to buy.

There is a second, nastier trap specific to this project's gate. The earlier power
analysis (`2026-08-16-harness-coverage-mde.md`) measured that a **monotone** arm — one
that never worsens anything — touching k of B meeting clusters produces a percentile CI
excluding zero as soon as k ≥ 4, *at any effect size*, because the percentile CI is
measuring how many blocks carry the sign rather than how big the effect is. An
automated loop will find such arms, because they are the easiest kind to write. So the
harness never treats significance as sufficient.

## The defence, in four parts

Full statement in the prereg; the short version and why each part is there.

**1. A search / confirmation city split, cut by an outcome-blind rule.** Cities are
sorted by reference-token count and assigned greedily toward a 65/35 token target.
Nothing about WER, and no idea, enters the rule; the realized split is pinned in code
and re-derived on every run, so it cannot drift.

| partition | cities | windows | meetings | ref tokens | W's WER | column oracle |
|---|---|---:|---:|---:|---:|---:|
| SEARCH | athens, chalandri, chania, orestiada, vrilissia, zografou | 153 | 103 | 47,252 | 0.09280 | 0.03847 |
| CONFIRM | argos, samothraki, sparta, xylokastro | 94 | 61 | 27,665 | 0.11354 | 0.06286 |

Each side holds a city that contributed nothing to fine-tuning (orestiada, argos). The
16 locked evaluation windows are sealed and unavailable; 6 of the 7 sealed temporal
holdout windows sit inside this run's 253 common windows and are filtered out before
anything is computed, which is why the substrate is 247.

The split is **enforced, not documented**: `run_search` refuses a substrate containing a
confirmation city, and `run_confirmation` refuses anything that is not exactly the
confirmation partition — not the same substrate passed twice, not a hand-picked subset.

**2. Exactly one confirmation batch per cycle, frozen before any confirmation number
exists.** Codex's correction to my first draft was sharp here: "at most five" is not a
protection if the first confirmation result decides who gets the second slot — the
second hypothesis would then depend on confirmation data and the split would be for
nothing. The implementation review sharpened it again: five *sequential* singleton
batches, each Holm-corrected inside itself, give a familywise error of 1 − 0.95⁵ =
22.6%, so a second batch is refused outright and needs a new protocol version.
Parameters are fitted once on the whole search partition and frozen; confirmation is a
locked-box application, not a second cross-fitted estimate. The one-way door survives a
process restart, and every read-decide-append runs inside one file lock so two
concurrent processes cannot both spend it.

**3. A null-imposed studentized wild cluster bootstrap-t, not the percentile tail.**
The first draft fed BH a two-sided percentile tail mass. Codex rejected it: resampling
the observed data centres the distribution on the observed effect, so its tail is not a
p-value under H₀; it double-counts the atom at zero; and it can exceed 1. The harness
now imposes the null on meeting contributions, studentizes with a cluster-robust SE,
and reports p = (1 + #{|T\*| ≥ |T|}) / (R+1) with R = 9,999 and Rademacher weights
**shared across every idea in a batch**, so the tests keep their joint dependence. The
percentile CI of the frozen scorer is untouched and still reported, so every number
stays comparable with the five earlier passes.

**4. The ship test is a minimum-effect test, Holm-corrected.** Testing against zero is
the wrong test when a monotone arm can excite the CI at any magnitude, so what Holm
corrects is a **one-sided** test of H₀: ΔWER ≥ −0.0010 — an idea ships only when the
data reject "smaller than useful". Holm rather than BH, because Holm bounds the
probability that *any* shipped idea is false, which is the decision; BH bounds the
expected false *fraction*, and its guarantee needs independence or PRDS, which sharing
bootstrap weights does not establish for ideas scored on the same meetings. BH is
reported beside it, over the confirmation batch and separately over the whole search
family, as description. Beside the test sit a support floor (≥ 8 meetings with a
non-zero error delta) and a single-item domination check
(max_b |d_b| / Σ|d_b| < 0.50) — the two gates no significance test supplies.

**And a behavioural dedup guard.** An idea is fingerprinted by the canonical edit events
between its output and W's, anchored to W's token positions so one insertion does not
renumber everything after it, each keyed-hashed. Enforcement uses exact Jaccard;
MinHash only shortlists. ≥ 0.90 against any already-evaluated idea is a refusal, written inside the
search-result record itself, and a refused idea can never reach confirmation. It
**fails closed**: if an earlier idea's hash set is missing from the cache or was keyed
differently, the run aborts rather than waving the new idea through.

## The first run

Eleven ideas, drawn from the #24 diagnostic of where the 5.30-point W-to-oracle gap
actually sits: 2-of-3 majorities that are jointly wrong hold 25.0% of it, occupancy
columns 14.2%, all three-way disagreements 12.7%, and across 62,919 columns where all
three systems agree the oracle never disagrees at all.

Search partition, leave-one-search-city-out, W = 0.09280.

Every idea's one-sided minimum-effect p is 1.000 — every effect points the wrong way —
so the ship column is omitted; the `wild p` column below is the descriptive two-sided
test against zero.

| idea | WER | ΔWER | percentile CI95 | wild p | del | ins | meetings touched | screen |
|---|---:|---:|---|---:|---:|---:|---:|---|
| W (baseline) | 0.09280 | — | — | — | 0.01737 | 0.03486 | — | — |
| null_identity (A/A) | 0.09280 | +0.00000 | [0, 0] | 1.000 | 0.01737 | 0.03486 | 0 | fail |
| occupancy_restore | 0.09280 | +0.00000 | [0, 0] | 1.000 | 0.01737 | 0.03486 | **0** | fail |
| singleton_oov_drop | 0.09280 | +0.00000 | [0, 0] | 1.000 | 0.01737 | 0.03486 | **0** | fail |
| unresolved_lexicon_pick | 0.09301 | +0.00021 | [−0.00013, +0.00054] | 0.254 | 0.01737 | 0.03486 | 35 | fail |
| majority_oov_override | 0.09439 | +0.00159 | [+0.00114, +0.00210] | 0.0001 | 0.01737 | 0.03486 | 50 | fail |
| two_present_oov_drop | 0.09496 | +0.00216 | [+0.00125, +0.00311] | 0.0001 | 0.02125 | **0.03304** | 78 | fail |
| majority_freq_ratio | 0.09528 | +0.00248 | [+0.00186, +0.00320] | 0.0001 | 0.01737 | 0.03486 | 70 | fail |
| majority_function_word | 0.09551 | +0.00271 | [+0.00175, +0.00380] | 0.0001 | 0.01742 | 0.03490 | 86 | fail |
| occupancy_restore_singleton_strict | 0.11166 | +0.01886 | [+0.01526, +0.02276] | 0.0001 | 0.01122 | 0.06027 | 97 | fail |
| occupancy_restore_singleton | 0.12133 | +0.02853 | [+0.02303, +0.03497] | 0.0001 | **0.00916** | 0.07240 | 98 | fail |
| majority_oov_override_restyled | — | — | — | — | — | — | — | **REFUSED, cosmetic variant** |

Every non-null idea made things **worse**, most of them decisively. No idea reached the
screen, so no confirmation batch was frozen and the confirmation partition was never
read. Confirmations remaining: 5 of 5.

### What the harness caught that a human would have quoted

Three of these are worth naming, because they are the failure modes the machinery
exists for.

- **Two ideas fired on zero columns.** `occupancy_restore` was written to restore text
  at `[x, x, ε]` columns the vote dropped — but the hierarchical vote never drops those,
  so there was nothing to restore. `singleton_oov_drop` was written to drop lone tokens
  W kept — but W already drops singletons, so there were none. Both produce a *perfect*
  ΔWER of zero, both rate gates pass to the digit, and both look serene in any summary
  table. The firing-set size, which is 0 for each, is what says they never ran. Without
  it the leaderboard would carry two ideas that were never tested.
- **The dedup guard fired.** `majority_oov_override_restyled` is the same rule as
  `majority_oov_override`, written with occupancy counts instead of the class function
  and with sorting instead of `Counter.most_common`. Its firing set was byte-identical
  and it was refused before it could be scored a second time. This was registered
  deliberately as a control; a PASS there would have been a harness bug.
- **The occupancy hindsight reproduced, and much worse than in hindsight.** The #24
  replay put occupancy at 14.2% of the gap while failing the insertion gate (ins 0.0374
  → 0.0391). The learnable version halves deletions — 0.01737 → 0.00916, which is a real
  effect — and pays for it with insertions at 0.03486 → 0.07240, more than doubling
  them, for a net +2.85 WER points. Restoring singletons by frequency is not a weak
  version of the oracle; it is the wrong operation.

### The one directional signal, stated as a signal and nothing else

`two_present_oov_drop` is the only idea that moved the insertion rate in the right
direction: 0.03486 → 0.03304. It paid 0.01737 → 0.02125 in deletions and lost overall.
That is a hint that a precision-side rule over `[x, x, ε]` columns has *something*, and
it is not evidence of anything — it failed the screen, on the search partition, in a
run whose purpose was to test the harness. It is recorded in the journal with its
hypothesis and is available to be beaten, not to be quoted.

### The majority class, three ways, all negative

The #24 diagnostic's largest single line is 2-of-3 token majorities: hindsight replay
recovers 25.0% of the gap there. Three text-only rules for *when* to override a majority
— out-of-vocabulary majority, frequency ratio, short function word — cost +0.16, +0.25
and +0.27 points respectively, each with a CI excluding zero on the wrong side and each
worse in all six search cities. That is consistent with what #24 already said and could
not prove: three systems agreeing 2-to-1 and being wrong together is precisely the case
where their text carries no signal, and a training-frequency prior is not the missing
signal. Search-partition evidence, so it forecloses nothing, but it is the third
independent time this project has found text-only arbitration empty.

## What was verified, and how

`.venv-eval/bin/python -m pytest eval/tests eval/controlled_eval -q` → **351 passed**
(311 before this work, 40 added). The tests that matter are the ones that test the
defence rather than the ideas:

- the partition rule is deterministic, disjoint, exhaustive, and the pinned split is
  re-derived from the real substrate;
- an all-zero arm returns exactly p = 1; the null rejection rate of the wild bootstrap
  is measured, not assumed;
- the sparse monotone arm that fools the percentile CI is caught by the effect and
  support floors;
- Holm and BH match hand-computed values and Holm is never looser;
- edit events survive a leading insertion without renumbering;
- a firing set contains no character above U+007F, and changes with the key;
- editing, reordering **or truncating the tail of** the journal makes it fail to load,
  and a journal whose checkpoint is missing is refused;
- a search substrate holding a confirmation city is refused, and a confirmation run on
  anything but the whole confirmation partition is refused;
- the dedup guard fails closed when its cache is gone;
- a uniform non-zero effect is not mistaken for an A/A arm, and the one-sided test
  ignores effects pointing the wrong way;
- a consistent but too-small effect is significant against zero and **not** against the
  effect floor — which is the whole reason the ship test is the minimum-effect one;
- confirmation cannot be run without a frozen batch, cannot be run twice, survives a
  process restart, cannot exceed its budget, and a *second* batch is refused outright;
- and the end-to-end noise simulation above, as a test at reduced replication.

## Caveats

- **This is a smoke test.** Eleven ideas is not a search. Nothing here is evidence about
  the model, and the one directional hint above must not be reported as a finding.
- **Sixth pass over the same 247 windows.** The split controls the multiplicity of
  *testing*. It cannot undo the adaptivity of *proposing*: these ideas were written by
  an agent that has read five prior reports about this exact substrate, confirmation
  cities included. Round 2 of the seed set was written after seeing round 1's search
  numbers — legitimate under the protocol, and exactly the adaptivity the split has to
  absorb.
- **The partitions are not exchangeable.** W scores 0.09280 on search and 0.11354 on
  confirm; the column oracle is 0.03847 and 0.06286. The confirmation cities are simply
  harder. A confirmation failure will therefore always be ambiguous between overfitting
  and city heterogeneity, and a confirmation success licenses a claim about meetings in
  these four cities, not about the next municipality. Four city clusters cannot support
  a population-of-cities claim.
- **Agreement-with-OpenCouncil, not fidelity-to-audio.** Same scorer as
  `exp-2026-08-16-composition-over-selection` and `exp-2026-08-16-char-vote-homophones`,
  not the benchmark app's.
- **The dedup guard is behavioural, not semantic.** Two genuinely different ideas that
  fire identically are indistinguishable to it, and an idea can dilute its Jaccard by
  adding irrelevant firings. It stops accidental and lazy duplication, not a determined
  one.
- **The implementation fingerprint has a stated hole.** It pins the factory, its whole
  MRO, `fusion_lab` and `autoresearch`, but not free helper functions in a caller's
  module — hashing the caller's module would re-key every idea in a library whenever one
  idea is added to it, which would make the journal useless across sessions. Ideas must
  keep their logic inside the class.
- **The wild bootstrap's calibration is verified under a well-behaved simulated null**
  (unequal cluster sizes, mean-zero contributions). It has not been verified under
  adversarial nulls such as a single dominating meeting or near-degenerate cluster
  variance; the domination and support floors are what stand in front of those.
- **`R_WILD` and the search-stage `n_boot` are Monte-Carlo knobs.** With R = 9,999 the
  smallest attainable p is 0.0001, which is what the table shows for the decisive
  failures; it is a floor, not a measurement.
- **The journal was reset during bring-up**, before any confirmation existed and before
  any number was reported, when the implementation review moved the protocol from
  `autoresearch-2026-08-16` to `-16b`. A reset is visible, because the sequence numbers
  restart at 1 and the protocol version is on every record.
- **The journal is tamper *evidence*, not a security boundary.** The chain is unkeyed
  and both it and its checkpoint are writable, so anyone willing to recompute them can
  rewrite the history. It stops accident and forgetfulness.
- **CodeRabbit could not review this change**: the free CLI allowance for this
  repository was exhausted ("You've used all 3 included reviews currently available
  under your plan"). The implementation review above is Codex `59c9564` standing in for
  it, and its eight findings were all addressed.

## Files

- [`eval/controlled_eval/autoresearch.py`](../../eval/controlled_eval/autoresearch.py) —
  registry, journal, partition, inference, multiplicity, dedup.
- [`eval/controlled_eval/autoresearch_seed.py`](../../eval/controlled_eval/autoresearch_seed.py) —
  the eleven ideas of this run and the runner.
- [`eval/tests/test_autoresearch.py`](../../eval/tests/test_autoresearch.py) — 40 tests.
- `research/autoresearch/journal.jsonl` — the append-only record, counts only.
- `eval/controlled_eval/results_autoresearch.json` — leaderboard and per-idea summaries.
