# Arm E on top of W: the positive survives the substrate change, and the coverage story was wrong

2026-08-17. Record: `exp-2026-08-11-name-repair`.
Prereg: [`docs/specs/2026-08-17-name-repair-on-w-prereg.md`](../specs/2026-08-17-name-repair-on-w-prereg.md),
frozen and Codex-reviewed (job `73670922`) before any WER number on W was computed.
Script: `eval/controlled_eval/exp_name_repair_w.py`.
Aggregates: `eval/controlled_eval/results_name_repair_w.json`.

**Metric: agreement-with-OpenCouncil, not fidelity-to-audio.** Nothing here says a
repaired name is what the speaker said; it says it is what our published text says.

## What was asked and what came back

Arm E — the frozen, LLM-free phonetic roster repair of
`scripts/serving_stack/name_repair.py` — had exactly one measured positive, and it sat
on **V**, the whole-window vote that `exp-2026-08-16-composition-over-selection`
displaced with **W** the same day. This re-measures it on W.

| arm | protection | WER | ΔWER vs W | 95% CI | del | ins | sub | gates |
|---|---|---|---|---|---|---|---|---|
| W | — | 0.10046 | — | — | 0.020316 | 0.037428 | 0.04271 | — |
| **W+E** | `agree` columns | **0.09971** | **−0.00075** | **[−0.00109, −0.00044]** | 0.020316 | 0.037428 | 0.04197 | **PASS** |
| W+E-strict | + `exact_2_of_3` | 0.10003 | −0.00043 | [−0.00066, −0.00022] | 0.020316 | 0.037428 | 0.04229 | PASS |
| W+E-unrestricted | none | 0.10015 | −0.00031 | [−0.00070, **+0.00007**] | 0.020316 | 0.037428 | 0.04241 | FAIL |

Raw integer counts behind the primary arm: S 3200 → **3144**, D 1522 → **1522**,
I 2804 → **2804**, over 74,917 reference tokens. The repair moves substitutions and
nothing else.

The preregistered primary endpoint — the 95% CI upper bound on ΔWER below zero — **is
met**, both rate gates pass, and leave-one-out over 247 windows, 144 meetings and 10
cities produces zero sign flips (city range −0.00082 to −0.00064; 9 of 10 cities
negative, xylokastro +0.00018). Head to head: 36 windows better, 205 tied, 6 worse.

97 firings in 52 windows and 44 meetings: 86 at edit distance 1, 11 at distance 2. A
further 53 firings were blocked by the protection rule.

## The design question, and why it was the whole job

E's firing restriction was "act only where the three systems disagree", implemented on
V by masking every token whose normalised string appears in **all three hypotheses
anywhere** — a *set* rule over token strings. That rule is well defined only because V
is one system's whole window. W's output is a text none of the three systems produced.

The re-specification, frozen before any number: **a W token inherits the MSA column it
was voted from and is protected iff that column's class is `agree`** ([x, x, x]).
Codex's review of the prereg confirmed this is the correct occurrence-level port and
named the one behavioural difference: it is *less* protective than the set rule, since
a string that appeared in all three transcripts somewhere was globally immune on V but
is eligible on W wherever its own column is disputed. There is no reverse case — every
`agree` occurrence also satisfies the old set condition.

The column census on this substrate: `agree` 62,919, `exact_2_of_3` 6,645,
`two_present_same` 4,569, `singleton` 4,460, `unresolved_three` 1,104,
`unresolved_two` 962, of 80,659 columns. W never emits a `singleton` token, so the
76,199 W tokens sit in the other five classes.

The choice **not** to protect `exact_2_of_3` in the primary arm was the judgement call.
`column_classes.py` declares that class settled — but for the *arbitration* arms (H, C),
which swap the voted token for the dissenting system's candidate. E never picks a
candidate from the column; it rewrites toward a closed roster list that may appear in no
hypothesis. Two systems mishearing a surname the same way is E's target case. Both
readings were preregistered and both are reported above.

**What the arms say about the protection rule.** The direct paired contrasts (declared
exploratory, sharing the 95% level with the primary endpoint, no multiplicity
correction):

- `W+E-unrestricted` − `W+E` = **+0.00044 [+0.00020, +0.00069]**
- `W+E-strict` − `W+E` = **+0.00032 [+0.00013, +0.00053]**

So the 53 firings on unanimous columns are net harmful: unrestricted still improves on
W by −0.00031 but its CI includes zero and 24 windows get worse against 6. And
protecting `exact_2_of_3` costs real gain — the 49 firings on token-majority columns
are net positive. The primary arm is the better of the two preregistered readings, by a
contrast whose CI excludes zero.

## Is 0.075 WER points robust, or an artefact of a handful of windows?

56 net reference-edit operations across the whole substrate. That is small, and the
report says so plainly rather than dressing it up.

- Largest single **window**: 5 of 56 = 8.9%.
- Largest single **meeting** — the bootstrap unit, so the check that matters: 5 of 56 =
  8.9%; maximum absolute meeting contribution 5; 33 meetings improve, 5 worsen.
- Meeting-level leave-one-out: no sign flip, range −0.00078 to −0.00068.

No single item dominates. The result is statistically resolved under the specified
resampling and **operationally small**: roughly 56 reference edits would erase it.

Deletion and insertion invariance was checked **below the pooled level**, because a
substitution can move the optimal Levenshtein alignment even at constant token count:
**zero of 247 windows** show any change in D or I, for any of the three arms. The rate
gates are not passing by cancellation.

## The comparison to V, which is descriptive only

| | V-era | W-era |
|---|---|---|
| baseline WER | 0.1201 | 0.10046 |
| ΔWER | −0.00083 [−0.00119, −0.00049] | −0.00075 [−0.00109, −0.00044] |
| firings | 103 (18 at distance 2) | 97 (11 at distance 2) |
| windows better / tied / worse | 36 / 205 / 6 | 36 / 205 / 6 |

Similar point estimates. This is **not** an independent replication: it is the same 247
windows, and both the base text *and* the evidence available to E changed. W's tokens
are lowercased and unpunctuated (`scoring.wtoks`), so the distance-2 `capital_mid`
signal of `has_signal()` structurally cannot fire — which is the visible cause of 18
distance-2 firings becoming 11. Read the two numbers as a reproduced point estimate
after a substrate re-specification, not as a replication.

E recovers **1.4%** of the gap from W to the alignment-conditional column oracle
(0.0475).

## Roster coverage: the 21.5% figure is wrong

`docs/reports/2026-08-17-untried-inventory.md` §A2 states that "only 56 of 260
benchmark windows (21.5%) and 43 of 203 meetings have a per-meeting roster" and
concludes E is a structural no-op on 78.5% of production. **That does not replicate.**

Counting non-empty `data/pii/rosters_full.json` entries for `{city}/{meeting}`:

| population | with roster |
|---|---|
| full benchmark report, 260 windows | **238 (91.5%)** |
| full benchmark report, 203 meetings | **183 (90.1%)** |
| analysis substrate, 247 windows | **232 (93.9%)** |
| analysis substrate, 144 meetings | **133 (92.4%)** |

All 232 roster-covered substrate windows also yield at least one roster-derived person
term, so "has a roster" and "has a usable roster" coincide here. Per city, coverage
ranges from 33/40 (athens) to 19/19 (orestiada, samothraki) and 38/38 (sparta).

The audit's figure appears to be a counting error rather than a different definition:
the last line of `data/pii/fetch_rosters.log` reads `216 ok, 56 failed ->
rosters_full.json (311 meetings total)`, and its first line reads `367 dataset
meetings, 95 already fetched, 272 to fetch` (95 + 216 = 311). **56 is the fetch-failure
count.** This is an inference from the arithmetic, not a confession from the author of
the audit, and it is possible some third definition produces 56. What is established is
the contradiction: under both definitions this report could construct — "a roster entry
exists" and "the built context contains ≥1 roster person term" — coverage is above 90%,
not 21.5%.

What *is* true, and is the narrower real finding: **all 7 sealed temporal-holdout
meetings have a zero-length roster.** Every one of them was among the 56 fetch
failures. That, not a 78.5% structural hole, is why E fired zero times on the holdout
in the serving-stack ladder.

Denominators, reconciled because they look inconsistent otherwise: the benchmark run
has 260 items; 253 are common to all 9 providers; of the 7 sealed holdout windows, 6
are inside those 253; the analysis substrate is the remaining 247.

## The two strata, never merged

| stratum | n | W | arm | ΔWER | 95% CI |
|---|---|---|---|---|---|
| **pooled, all windows** — *the shipping number* | 247 / 144 mtg | 0.10046 | 0.09971 | −0.00075 | [−0.00109, −0.00044] |
| **roster-conditional** — *the mechanism number* | 232 / 133 mtg | 0.09922 | 0.09842 | −0.00080 | [−0.00117, −0.00048] |
| no roster | 15 / 11 mtg | 0.11948 | 0.11948 | **0.00000** | [0, 0] |
| fired windows only | 52 / 44 mtg | 0.09322 | 0.08962 | −0.00360 | [−0.00497, −0.00235] |

The 15 no-roster windows are a byte-exact no-op, 15/15 tied — the structural no-op is
confirmed, it is just far smaller than the audit claimed. Because coverage is 94% and
not 21.5%, the pooled and conditional numbers are close, and the conditional stratum is
not rescuing anything.

## The Βήμα-3 deployment gates, judged honestly

The gates of [`docs/specs/2026-08-11-name-repair-plan.md`](../specs/2026-08-11-name-repair-plan.md)
are to be judged **on untouched meetings**. The only untouched meetings this project
has are the 7 sealed temporal-holdout windows, and they have no rosters at all.

| gate | status |
|---|---|
| (1) lower 95% bound on beneficial-change precision ≥ 95% | **unassessable** — needs per-change adjudication against a name ground truth. Exists only for the Step-0 audit sample, never for W |
| (2) upper 95% bound on harmful changes < 5% | **unassessable**, same reason |
| (3) **zero correct→wrong across ≥ 300 activatable points** | **unassessable as written on the current holdout.** Zero rosters there means zero activatable points, and the supply is far below 300 in any case (below) |
| (4) no valid name replaced by a different person | **unassessable**, same reason as (1) |
| (5) 95% CI of total ΔWER entirely below zero | **numerically met on development data** — [−0.00109, −0.00044]. The gate as written is on untouched meetings, so the *deployment* gate remains unevaluated |
| (6) beneficial − 5 × harmful positive in every stratum | **unassessable** without (1)–(2)'s adjudication |

**The supply of activatable points, measured.** Running the frozen rule's `select()`
over every one of the 76,199 W tokens on all 247 windows: 150 `fire`, 29
abstain-with-a-candidate, 648 `abstain_already_valid`, 75,372 `no_candidate`. Taking
firings only, that is 0.61 per two-minute window; taking any point where the rule
reached the candidate stage (179), 0.72 per window.

Reaching 300 activatable points therefore needs roughly **390–465 roster-covered
two-minute windows — 13 to 15 hours of council audio** on meetings this project has
never scored. The sealed holdout is 7 windows with zero rosters. Gate (3) is not
failed; it is **unassessable on the current holdout by roughly two orders of
magnitude**, and the plan already anticipated this: *"Αν δεν υπάρχουν αρκετές
ενεργοποιήσεις για να ικανοποιηθούν οι πύλες, δεν φεύγει αυτόματα — μπαίνει σε
shadow/suggestion mode."* That clause, not a pass or a fail, is what governs.

## Conclusion

The preregistered primary endpoint is met on development data: E produces a small,
statistically resolved improvement in agreement-with-OpenCouncil on top of W, with no
aggregate *and no per-window* deletion or insertion increase, stable in direction under
leave-one-window, -meeting and -city checks, and not carried by any single window or
meeting.

**The project's one measured positive survives the substrate change.** It does not
become a shipping decision. 0.075 WER points is 1.4% of the remaining headroom to the
column oracle; four of the six deployment gates cannot be evaluated at all without a
name-level adjudication this project has never done for W, and the one that could be
evaluated needs untouched, roster-covered audio that does not exist yet. Shadow
evaluation is supported. Shipping is not.

The lexicon's pending v2 alias fix for the two registry-vs-reference orthography
conflicts (Αδραχτάς / Καδόλου) is **still pending**: neither the v1 nor the v2 term
files carry `αδραχτας` or `καδολου` as an alias, and this measurement used the frozen
v1 files, unchanged from the V-era run.

## Caveats

- **Agreement-with-OpenCouncil, not fidelity-to-audio.** A "repaired" name is one that
  matches our published text. Nothing here shows a human who listened would agree.
- **Not independent confirmation.** This is the fifth analysis pass over the same 247
  windows, and the arm, its firing rule and its lexicon were chosen by people who had
  seen earlier numbers on them. The nominal CI is a preregistered re-measurement on
  reused development data. The sealed holdout stays sealed.
- **The V→W comparison is descriptive**, not causal: base text and available evidence
  both changed.
- **Small.** 56 net reference edits. Statistically resolved, operationally marginal,
  and fragile to any change in the reference text.
- **No multiplicity correction.** Three arms, two arm-to-arm contrasts and four strata
  share the 95% level. Only the pooled `W+E` vs `W` contrast is the preregistered
  primary endpoint; everything else is exploratory.
- **The rate gates say nothing about semantic damage.** Unchanged D and I do not
  establish zero correct→wrong or zero wrong-person substitutions; those are gates
  (1)–(4), and they are unassessed.
- **The coverage correction is arithmetic, not testimony.** The contradiction with
  §A2 of the untried-inventory report is established; the diagnosis (a misread
  fetch-failure count) is an inference. Benchmark coverage also does not establish
  *production* coverage — this counts what is cached in `rosters_full.json`, not what
  the live endpoint can fetch.
- **`msa.py` was not modified.** The cached MSA alignment key
  `align_65b1c4d64618a429.json` is identical before and after the run, verified in the
  results file.
- Two Codex passes shaped this. Job `73670922`, before any number: approved the column
  re-specification and the `exact_2_of_3` ordering, and required the directional CI
  test, the round-trip assert on the token↔regex mapping, the "conditional may not
  rescue pooled" clause and the fifth-pass caveat. Job `de7a5729`, before this
  conclusion was written: cut "replicates" to "reproduces a similar point estimate",
  cut "safe" to "no aggregate rate-gate regression", cut "unreachable" to
  "unassessable", required the direct arm-to-arm contrasts, the meeting-level
  domination check, the per-window D/I invariance check, the raw integer counts, the
  denominator reconciliation, and softened the coverage diagnosis from an accusation to
  an inference.
