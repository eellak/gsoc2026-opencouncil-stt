# Prereg: arm E (frozen phonetic roster repair) on top of W

Frozen 2026-08-17, **before any WER number on W was computed**. Record:
`exp-2026-08-11-name-repair`. Substrate: `exp-2026-08-16-composition-over-selection`.

## 1. What is being re-measured, and why it is not the same measurement

Arm E is the frozen, LLM-free phonetic repair of `scripts/serving_stack/name_repair.py`:
a hypothesis token is rewritten to a closed-list roster alias only under the audited
length/distance budget, unique-best-candidate, margin, single-person, common-word and
distance-2-signal conditions. Nothing in that rule is changed here.

E has one measured positive, on **V** (the whole-window vote):
WER 0.1201 → 0.1193, −0.00083 [−0.00119, −0.00049], both rate gates unchanged to the
digit because the repair only ever moves substitutions
(`exp-2026-08-16-roster-grounded-selection`).

`exp-2026-08-16-composition-over-selection` displaced V with **W** the same day
(0.1201 → 0.1005). W's output is a text none of the three systems produced, assembled
column by column from an exact three-way MSA. The E number does not transfer to it.

## 2. The crux: re-specifying "act only where the three systems disagree"

E's firing restriction on V was implemented in `restricted_repair()`
(`eval/controlled_eval/exp_roster_selection.py:104`) by masking every token whose
normalised string appears in **all three hypotheses anywhere** — a *set* rule over
token strings. That is a selection-era concept: V is one system's whole window, so
"the systems agree on this token" can be answered by set membership.

W has no such token set. Each W token is the output of a per-column vote over one MSA
column. The re-specification is therefore **positional, not set-based**:

> **A W token inherits the MSA column it was voted from. It is protected iff that
> column's class is `agree`** — i.e. all three systems supplied a token in that
> column and all three supplied the same one.

Justification, stated before any number:

- It is the *faithful* port of the V-era intent. The V-era rule protected a token
  because "the string appears identically in all three hypotheses"; the column
  analogue of "all three assert this token here" is exactly `column_class == "agree"`.
- It is what Codex asked for and the V-era measurement could not deliver. Job
  `ada1cc4a` asked for agreement on **aligned occurrences**; the set rule was kept
  only because it errs safe. On W the alignment exists, so the occurrence-based rule
  is available and is used.
- It is well founded on this substrate. `results_column_census.json`: `agree` is
  62,919 of 80,659 columns, and the column oracle never disagrees with a unanimous
  column in any of the 62,919. Protecting them cannot cost anything the oracle would
  have taken.
- W never emits a `singleton` token (occupancy vote needs two systems), so the only
  unprotected classes W can emit into are `two_present_same`, `exact_2_of_3`,
  `unresolved_two` and `unresolved_three`.

### The `exact_2_of_3` question, decided here and not later

`column_classes.py` declares `exact_2_of_3` ([x, x, y]) "SETTLED, no arm may touch it".
That freeze was written for the **arbitration** arms (H, C), which replace the voted
token with *another system's candidate from the same column* and would therefore
overturn a real 2-of-3 token majority. E is a different mechanism: it never picks a
candidate from the column, it rewrites toward a closed roster list that may appear in
no hypothesis at all. Two systems mishearing the same surname the same way is exactly
E's target case.

Decision, frozen: **`exact_2_of_3` is NOT protected in the primary arm.** The primary
arm protects `agree` only. Because this is a judgement call and not a derivation, the
stricter reading is preregistered as a **declared secondary arm**, reported beside the
primary in the same table, never substituted for it.

## 3. Arms

| id | protection | status |
|---|---|---|
| `W` | — | baseline |
| `W+E` | `agree` columns protected | **primary** |
| `W+E-strict` | `agree` ∪ `exact_2_of_3` protected | declared secondary |
| `W+E-unrestricted` | nothing protected | labelled sensitivity, not an arm |

The V-era report carried its unrestricted variant as a sensitivity too (103 restricted
vs 161 unrestricted firings); the same shape is kept.

## 4. Declared substrate differences that are not the arm's fault

These are consequences of measuring on a token stream instead of raw hypothesis text.
Both are declared now, not discovered later:

1. **No casing, no punctuation.** W's tokens come from `scoring.wtoks` (NFD, combining
   marks stripped, lowercased, `\w+`). The distance-2 `capital_mid` signal of
   `has_signal()` can therefore never fire on W. Distance-2 firings on W require
   `kyrios` or `first_name` adjacency. This makes E on W **strictly more conservative
   at distance 2** than E on V. 18 of the 103 V-era firings were at distance 2; how
   many survive is reported, not predicted.
2. **`_match_case` is inert**, so replacements are written in the term file's stored
   lowercase accent-free surface form. Every scorer here strips combining marks and
   lowercases, so this cannot change a number.

A replacement must be a single `\w+` token, asserted at runtime; if any admitted alias
is multi-token the run aborts rather than silently changing the token count.

## 5. Lexicon, frozen

`research/ds_wer/terms/{city}.json` — the **v1** hash-frozen files of 2026-08-12, the
same ones the V-era measurement used, via `roster_lexicon.load_city_terms()`. The
`.v2.json` files (argos, orestiada) are **not** used: switching term files would change
the arm as well as the substrate and the two effects would not be separable.

The `next_action` of `exp-2026-08-11-name-repair` mentions a pending v2 alias fix for
two registry-vs-reference orthography conflicts (Αδραχτάς/Καδόλου). **It is not in the
frozen files** — neither v1 nor v2 carries `αδραχτας` or `καδολου` as an alias — so it
is still pending and is not part of this measurement.

Rosters: `data/pii/rosters_full.json`, plus the frozen mined slice of
`roster_lexicon.admitted_mined()` and the leave-one-city-out common-word table, exactly
as `exp_roster_selection.py` builds them.

## 6. Gates, frozen

Identical to the project's standing fusion gates (`fusion_lab.evaluate`):

- paired clustered bootstrap by **meeting**, 10,000 replicates, seed 7;
- **primary, directional** (Codex job `73670922`): with ΔWER defined as arm minus W,
  the 95% CI **upper bound must be below zero**. A CI entirely *above* zero
  establishes harm and must not be read as "significant, therefore ship";
- **deletion rate gate**: `del_rate(arm) <= del_rate(W)`;
- **insertion rate gate**: `ins_rate(arm) <= ins_rate(W)`;
- leave-one-out over windows, meetings and cities: zero sign flips;
- **single-item domination**: the largest single window's share of the total gain is
  reported; a delta whose sign depends on one window is reported as not established.

Deployment value requires the directional CI **and** every standing gate, not
statistical significance alone.

The deletion and insertion gates are kept even though E preserves the token count:
token-count preservation does not make D and I invariant, because the scorer's optimal
realignment can redistribute S/D/I around a changed token.

E is fitted-parameter-free, so leave-one-city-out is vacuous by construction and
`fold_note` says so.

## 7. Two reporting strata, frozen, never merged

1. **Pooled (all 247 windows)** — the *shipping number*. What production would see if
   E were switched on for every window, including those where it is a structural
   no-op.
2. **Roster-conditional (windows whose meeting has a roster)** — the *mechanism
   number*. What the repair does where it can act at all. Its own clustered CI over
   its own meetings.

Neither is a correction of the other. The pooled number decides deployment value; the
conditional number describes the mechanism. **The conditional number may not rescue a
pooled null**: if the pooled CI fails the directional test, the arm is negative and the
conditional stratum is reported as mechanism description only.

Roster coverage on this substrate is itself a preregistered reported quantity, with
its definition stated: a window counts as roster-covered iff
`rosters_full.json` has a non-empty entry for `{city}/{meeting}`. The count of windows
where the built context additionally contains at least one roster-derived person term
is reported beside it.

## 8. What would make this a negative

If the pooled CI on ΔWER vs W includes zero, the arm is reported as **not
demonstrated on W**, the record closes on that, and the report states plainly that
this project then has no measured positive left. No variant hunting: the three arms
above are the whole list, and `W+E-strict` and `W+E-unrestricted` are sensitivities,
not fallbacks.

## 8b. What this cannot be

This is the **fifth analysis pass over the same 247 windows**, and the arm, its
firing rule and its lexicon were all chosen by people who have seen earlier numbers on
them. The nominal CI is therefore a preregistered **re-measurement on reused
development data**, not independent confirmation (Codex job `73670922`). The sealed
temporal holdout of `eval-freeze-2026-08` remains the only confirmation substrate, and
nothing here releases it.

The V-vs-W effect-size comparison is **descriptive, not causal**: the base transcript
changed *and* the evidence available to E changed (§4). Only the W-vs-W+E contrast is
an estimand this design supports.

## 9. Hard constraints

- `eval/controlled_eval/msa.py` is **not modified**. `fusion_lab._cache_path()` keys on
  its sha256; the 18 MB cached alignment must stay valid. The key is verified unchanged
  at the end of the run.
- No transcript text or audio in git. Aggregates only in `results_name_repair_w.json`.
- Zero GPU, zero paid API.
- Agreement-with-OpenCouncil throughout. Nothing here is fidelity-to-audio.
