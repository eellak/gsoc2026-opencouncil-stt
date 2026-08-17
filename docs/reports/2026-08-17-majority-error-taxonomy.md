# What the wrong 2-of-3 majorities actually are

2026-08-17 · descriptive follow-up to
[`2026-08-16-char-vote-homophones.md`](2026-08-16-char-vote-homophones.md) ·
sixth pass over the same 247 windows · no arm, no gate, no prereg

**Every number in this report is computed with the reference text in hand.** It is
hindsight. None of it is an achievable gain, none of it may be quoted as one, and
nothing here went through a promotion gate — the autoresearch confirmation partition
was not read.

**Erratum, 2026-08-17.** That last clause is true only of the harness API, which this
report never called. It is **false** as a claim about the labels: `load_substrate()` has
no city filter, so every count here spans all 247 windows and all 10 cities, including
the four sealed confirmation cities and their 27,665 reference tokens. Anything selected
by reading the numbers below cannot afterwards claim a confirmatory CI on that
partition. See [`2026-08-17-confirmation-audit.md`](2026-08-17-confirmation-audit.md)
(`exp-2026-08-17-confirmation-audit`).

`exp-2026-08-16-char-vote-homophones` found that a hindsight replay over
`exact_2_of_3` columns closes **25.0%** of the 5.30-point gap between W (0.1005) and
the alignment-conditional column oracle (0.0475) — twice what every unresolved column
holds together. Three arms then tried to override such majorities and all three made
WER worse. Nobody had looked at what the errors *are*. This is that look.

The headline: **the class is not one thing, and 27.7–39.6% of it is unreachable by
any voting rule at all.** The largest coherent linguistic bucket is function-word pair
confusion. And the entity cross-check the ticket asked for comes back the way
`exp-2026-08-16-error-mined-terms` predicted: of the wrong majorities whose correct
word is a frozen term, **93 of 99 are in that window's own city file and 83 of 99
survive the roster gate** — but only 28 of those 83 are recoverable at all, so
"application, not coverage" is right and the prize behind it is small.

## Method

Substrate: the frozen `fusion_lab` 247 windows / 144 meetings / 10 cities / 74,917
reference tokens, W = the per-column vote of
`exp-2026-08-16-composition-over-selection`, 7,526 edits against the oracle's 3,557,
so the gap is **3,969 edits**. `msa.py` was not modified: `fusion_lab` hashes it into
the 9 MB alignment cache key, so the attribution DP is a local copy pinned to
`msa.oracle_select` by `test_majority_taxonomy.py`.

Three axes, deliberately not merged. The shape is Codex job `95b03e7c`'s; the first
draft folded "what kind of token is this" and "what relation holds between the two
tokens" into one first-match partition, which throws one of the two dimensions away.

**Outcome** is read off the DP's *optimal-support set*, not off one backtrace. On a
column whose candidates are all wrong the backtrace is a tie-break, and substituting
`x` for `z` costs exactly what deleting `z` and inserting `x` costs, so the
coverage/spurious label would otherwise be decided by transition order rather than by
the data. Cases where both explanations are optimal get their own `ambiguous` label
instead of being forced.

**Recoverability is priced three ways and never merged:**

| quantity | question it answers |
|---|---|
| `g_oracle` | can the majority token lie on *some* globally optimal oracle path? |
| `g_W` | with every other column frozen to what W emitted, does changing **this one** column reduce the edit distance? |
| replay | joint hindsight effect of changing a whole named set of columns, frozen scorer |

`g_W` is the decision-relevant one. Everything an arm could do at one column is
bounded by it.

## The population, and a correction to a tempting reading

Of 6,645 `exact_2_of_3` columns, **1,719 (25.9%) have a wrong majority.**

That is not the same set as the census's 1,245 "W differs from the oracle's choice",
and the reconciliation matters:

- 1,215 of the 1,245 are in the wrong set.
- The other **30** are columns where the majority token still lies on an optimal
  path — the oracle's disagreement there is a pure tie-break.
- A further **504** wrong majorities are *not* in the 1,245 at all: the oracle picks
  the same wrong token W did, because it is the least-bad candidate available.

It is true, and independently verified here, that on all 1,245 the oracle takes the
lone dissenter, that 804 of them are within character distance 2, that 104 are strict
homophone pairs and 108 loose, and that 70 of the majority tokens are a single
character. **The first of those is a candidate-set identity, not a finding.** An
`exact_2_of_3` column has all three systems emitting a token, so epsilon is not a
candidate and a disagreeing oracle has exactly one alternative to take. It does not
follow that the dissenter is the reference word, and this report exists partly to say
so: it is not, in 681 of 1,719 cases.

## Outcomes

| outcome | n | share | `g_W ≥ 1` |
|---|---:|---:|---:|
| `selection` — the minority token **is** the reference word | 1,038 | 60.4% | 1,000 |
| `coverage` — the majority substitutes for a word neither system proposed | 318 | 18.5% | 0 |
| `ambiguous` — substitution and insertion are both optimal explanations | 205 | 11.9% | 0 |
| `spurious` — the majority is an insertion on every optimal path | 158 | 9.2% | 0 |

**Definitely not a selection failure: coverage + spurious = 476, 27.7%. Not cleanly
attributable to selection: 681, 39.6%.** The honest statement is the range, because
`ambiguous` is undecidable by construction under a unit-cost scorer.

Operationally the split is sharper still: **719 of 1,719 columns (41.8%) have zero
marginal benefit** in the frozen-W single-column replay, and a replay over exactly
those 719 saves **2 edits** out of 3,969. The complementary 1,000 columns save 999 of
the 1,001 edits the whole class is worth. That 41.8% includes 38 rows structurally
classified as `selection`, so it answers a slightly different question — and the
two-edit residue shows that `g_W = 0` means "no gain given W's other choices", not
"impossible under any surrounding decisions".

The whole-class replay reproduces the earlier headline: 1,719 columns, 1,001 edits,
WER 0.10046 → 0.08710, **25.2% of the oracle gap**. That is a share of the *gap*, not
of WER; in WER terms it is 1.336 absolute points, 13.3% relative to W.

## The taxonomy

Partition. `closure` is the standalone hindsight replay of that class alone; the
leave-one-class-out marginals agree with it to within 0.03 percentage points, so on
this substrate the classes happen to be near-additive — that is measured, not
guaranteed, and it does not license adding them up in general.

| class | n | share | per 1k ref tokens | `g_W ≥ 1` | edits | closure of gap | share of W's errors | yield/col |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `different_content_word` | 550 | 32.0% | 7.34 | 265 | 266 | 6.70% | 3.53% | 0.48 |
| `function_word_pair` | 342 | 19.9% | 4.57 | 277 | 276 | 6.95% | 3.67% | 0.81 |
| `numeric_identifier` | 236 | 13.7% | 3.15 | 164 | 164 | 4.13% | 2.18% | 0.69 |
| `morphology_suffix_neighbor` | 188 | 10.9% | 2.51 | 142 | 142 | 3.58% | 1.89% | 0.76 |
| `no_target` | 147 | 8.6% | 1.96 | 0 | 0 | 0.00% | 0.00% | 0.00 |
| `named_entity` | 110 | 6.4% | 1.47 | 41 | 41 | 1.03% | 0.54% | 0.37 |
| `orthography_homophone` | 109 | 6.3% | 1.45 | 85 | 85 | 2.14% | 1.13% | 0.78 |
| `alignment_artifact` | 30 | 1.7% | 0.40 | 24 | 24 | 0.60% | 0.32% | 0.80 |
| `protocol_legal` | 7 | 0.4% | 0.09 | 2 | 2 | 0.05% | 0.03% | 0.29 |

Reading it:

- **Function-word pair confusions are the largest coherent linguistic bucket**: one
  fifth of the wrong majorities and about 28% of the measurable hindsight gain (276 of
  1,001 edits). They also have the highest per-column yield of any large class. The
  bucket was not audited into subtypes, so it is "function-word pairs", not
  "articles and particles", and no claim is made about whether a reader would notice.
- **The residual is the biggest single bucket and it is heterogeneous.** 550 columns,
  but 222 of them are function↔content pairs rather than two content words, and its
  yield per column (0.48) is the worst of any large class — half its members are
  coverage or ambiguous.
- **`morphology_suffix_neighbor` is not "most of it".** The ticket's prior — that
  Greek inflection would dominate — is not supported: 10.9% by count, 3.58% closure.
  The label is deliberately string-shape, not linguistic: a long shared prefix and two
  short tails. It is not a lemma test and no morphological analyser was used.
- **Orthography is small here too**, 6.3%, consistent with the 34 strict-homophone
  columns the census found in the unresolved classes.
- **`no_target` is the pure-loss bucket**: 147 columns where the majority is an
  insertion with nothing to compare it to. Zero recoverable, zero closure. (The
  `spurious` *outcome* has 158 members; 11 of them land in other classes because the
  partition puts type flags — numeric, entity, protocol — ahead of the relation.)
- **Character distance** across the whole population: 1 in 701 columns, 2 in 263, >2
  in 597, and 158 with no target at all.

**Single-item domination: no class is dominated.** Every class with n ≥ 100 has a
top-meeting share ≤ 11.6% and a top-city share ≤ 35.4%, spread over 55–128 distinct
meetings. The only concentrated row is `protocol_legal`, where one meeting holds 28.6%
of 7 columns — that row establishes nothing and is reported for completeness.

## The entity cross-check

This is the part `exp-2026-08-16-error-mined-terms` left open: SCHOINA was already in
Chania's frozen term list and was still corrected 11 times by hand, and the record
concluded the binding constraint is the repair/selection point, not lexicon coverage.
Nobody followed it up.

**99 wrong majorities have a reference target that is a term in a frozen city term
file** (98 of them sit in the `named_entity` class; one is routed to
`numeric_identifier` by the partition's precedence). 67 distinct terms, 52 meetings,
all 10 cities, top term 5.1% of the rows — not a single-name artefact.

| funnel stage | n | of 99 |
|---|---:|---:|
| target is a term in some frozen city file | 99 | 100% |
| … in that window's **own** city file | 93 | 93.9% |
| … admitted to that **meeting's** roster-gated list | 83 | 83.8% |
| frozen `name_repair.select()` fires on the wrong token | 36 | 36.4% |

Where the other 47 admitted rows go, running the frozen rule with the meeting's own
`RosterContext`:

| decision | n |
|---|---:|
| `fire` | 36 |
| `no_candidate` | 33 |
| `abstain_already_valid` — the wrong token is itself a valid alias | 9 |
| `abstain_margin` | 4 |
| `abstain_multi_person` | 1 |

Of the 33 `no_candidate`: **21 have a wrong token shorter than 6 characters**, which
the candidate-eligibility policy rejects outright; 10 are length 6–9 at character
distance > 1; 2 are length ≥ 10 at distance > 2.

Two things follow, and they pull in opposite directions.

1. **Coverage is not the constraint for these errors, application is.** 93.9% of the
   targets are already in the right city's frozen file and 83.8% survive the roster
   gate. Attrition happens downstream, at candidate eligibility and abstention. The
   largest single attrition reason is the minimum-length gate — 21 of 33
   `no_candidate`, 63.6%. **That is a diagnostic attribution, not a recommendation.**
   The gate exists to stop false positives, and loosening it would need its own
   benefit-and-harm replay; nothing here says WER would improve.
2. **The prize inside this funnel is small.** Only **28 of the 83** admitted rows are
   recoverable at all (`g_W ≥ 1`), which caps a one-token replacement at 28 edits =
   **0.037 WER points** in this class. 16 of the 36 fires overlap with a recoverable
   row, and that is an overlap statistic only — it was not verified that the fired
   candidate is the beneficial target.

This is **not** a ceiling on `exp-2026-08-11-name-repair`. That arm's −0.083 points
was measured on top of the whole-window vote, over all positions, not only over wrong
2-of-3 majorities inside the per-column composition. What the funnel does establish is
narrower and still useful: among errors whose correct word is already known to a
frozen list, own-city routing and roster admission retain most cases, and the observed
losses are downstream of the lexicon.

## What this refuses to claim

- That oracle disagreement proves the dissenter is the reference word, or that the
  whole `exact_2_of_3` class is a selection failure. It is not: 27.7% definitely is
  not, and up to 39.6% is not cleanly attributable to selection.
- That exactly 39.6% is *proven* non-selection. `ambiguous` is undecidable under a
  unit-cost scorer, so the supported statement is a range.
- That `g_W = 0` proves no per-column method could ever help — it means no gain given
  W's other choices, and the 719-column replay's two edits show the difference.
- That any closure here is an achievable online gain, a causal effect, or additive in
  general. The near-additivity is measured on this substrate only.
- That 25.2% is a share of WER. It is a share of the **oracle gap**.
- That the entity funnel measures overall dictionary coverage — its 99 rows were
  selected *because* the target is in a frozen file, so 93/99 is correct-city routing
  conditional on membership, not coverage. Nor does it price the whole name-repair arm.
- That a rule firing is necessarily correct or beneficial.
- That relaxing the length or edit-distance gate would improve WER.
- That `function_word_pair` is all articles and particles, or perceptually
  insignificant; or that `different_content_word` is a pure content-content class —
  222 of its 550 rows are function↔content.
- That the term lists estimate production coverage. They were mined from material
  overlapping this benchmark, so they are optimistic and leaky.
- Anything general from `protocol_legal` (n = 7).

## Caveats

- **Sixth pass over the same 247 windows.** Adaptive-overfitting pressure across
  passes is not removed by a design being descriptive.
- The 6 sealed temporal-holdout windows of `eval-freeze-2026-08` were removed by the
  same explicit filter before anything was computed, and the autoresearch confirmation
  **API** was not called. **Erratum 2026-08-17:** that is not the same as the partition
  being unread. Its four cities' labels are in every count here — see the erratum at the
  top and [`2026-08-17-confirmation-audit.md`](2026-08-17-confirmation-audit.md).
- Agreement-with-OpenCouncil, not fidelity-to-audio. The "reference word" throughout
  is the OpenCouncil published text, and nobody listened to the audio for this report.
- The class labels are text heuristics: closed word lists for function words, numerals
  and protocol markers, a phonemic key for orthography, a string-shape prefix test for
  morphology, and frozen term-file membership for entities. No morphological analyser,
  no NER, no gazetteer beyond the frozen files, no audio. An acoustic-confusion class
  is *not* claimed from text alone.
- The name-repair funnel runs `select()` on W's normalized lowercase token stream, so
  the capital-mid signal of `has_signal()` is structurally unavailable and `seen_freq`
  is empty, which makes `abstain_common` unreachable. Both make the funnel
  **optimistic** about firing.
- Alignment-conditional throughout: outcomes, targets and gains all depend on the one
  exact 3-way MSA. The column oracle ranged 0.0461–0.0479 over seven alignments.
- Scored with `eval/controlled_eval/scoring.py`, not the benchmark app's scorer.
- Zero GPU, zero paid API, CPU only. `msa.py` is byte-identical to `main`; the
  alignment cache key is `align_65b1c4d64618a429`, unchanged.
- Two Codex passes shaped this. Job `95b03e7c`, before any number: replaced the
  single-backtrace outcome labelling with optimal-support sets, added the `ambiguous`
  class, separated `g_oracle` from `g_W`, forced the type flags to be cross-cutting
  rather than a first-match class, killed the "count / reference tokens = WER share"
  denominator, and required the entity funnel to key on the target rather than on
  either side. Job `6340c156`, on the findings: supplied the 27.7–39.6% range, cut the
  entity conclusion back to what a membership-conditioned sample supports, and wrote
  most of the refusal list above.

## Artefacts

- `eval/controlled_eval/exp_majority_taxonomy.py` — the analysis, self-contained
  against an untouched `msa.py`.
- `eval/controlled_eval/test_majority_taxonomy.py` — pins the local DP copy to
  `msa.oracle_select` and the forced-choice costs to brute-force edit distance.
- `eval/controlled_eval/results_majority_taxonomy.json` — counts and class labels only.
- `~/.cache/oc-public/majority_taxonomy/examples.json` — the 1,719 verbatim column
  pairs. **Council speech; never in git.**
