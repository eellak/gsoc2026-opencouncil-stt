# Is the autoresearch confirmation partition still confirmatory for F1?

2026-08-17 · forensic audit, zero GPU, zero paid API, zero LLM calls on the substrate ·
step 0 of [`docs/specs/2026-08-17-llm-composer-draft.md`](../specs/2026-08-17-llm-composer-draft.md) ·
reviewed by Codex job `6eefa29e` (sol, high effort)

**Verdict: (b). The confirmation partition is not confirmatory for F1.** F1 may be
built and run, but it reports **exploratory** results with **no confirmatory CI**, and
**no confirmation is spent**. Option (c) is refused: there is no material left to carve
a holdout from that is both untouched and large enough to test the ship floor.

The one clean answer in the whole audit is the split geometry: it is by city, which is
strictly coarser than meeting, so nothing straddles. That is also the only question the
plan asked that was not already lost.

## What F1 is, and where it came from

F1 is "override a wrong 2-of-3 majority" — the `exact_2_of_3` column class. It is the
single family that survived the two independent reviews of revision 2. Its scope, its
success condition and the reason its two siblings were cut are each read directly off
three reference-conditioned analyses. Every one of those ran on all 247 windows.

## Evidence table

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Did the oracle counts and the majority-error taxonomy include confirmation cities? | **YES — they did include them** | below |
| 2 | Did earlier experiments expose confirmation outcomes? | **YES** | below |
| 3 | Did roster and term-list construction use confirmation references? | **PARTLY YES** — a separate, narrower channel | below |
| 4 | Is the split by meeting, not by window? | **YES, and coarser still: by city** | below |
| 5 | Has the confirmation partition ever been read *through the harness*? | **NO** | below |

### 1. The oracle counts and the taxonomy ran on all ten cities

The partition, verified against the preregistration
([`docs/specs/2026-08-16-autoresearch-partition-prereg.md:41-44`](../specs/2026-08-16-autoresearch-partition-prereg.md)):

| partition | cities | windows | meetings | ref tokens |
|---|---|---:|---:|---:|
| SEARCH | athens, chalandri, chania, orestiada, vrilissia, zografou | 153 | 103 | 47,252 |
| CONFIRM | argos, samothraki, sparta, xylokastro | 94 | 61 | 27,665 |

Pinned in code at `eval/controlled_eval/autoresearch.py:102-103`, recomputed by
`plan_partition` (`:131`) and cross-checked by `assert_partition` (`:153-159`).

The substrate loader every analysis script uses is
`eval/controlled_eval/fusion_lab.py:128-138`. Its **only** filter is the 6 sealed
`eval-freeze-2026-08` holdout windows. There is no city filter anywhere in it. The
partition restriction lives in a different module and is applied only by
`autoresearch._restrict` (`autoresearch.py:203`), which the analysis scripts never call.

Consequences, one script at a time:

- **`eval/controlled_eval/column_census.py:38`** calls `load_substrate()` bare. Its
  output header is `"n_windows": 247, "n_meetings": 144, "n_cities": 10`
  (`results_column_census.json`, first line). It reports `exact_2_of_3` = **6,645
  columns** with **1,245** where W differs from the column oracle — and
  `oracle_select` reads the reference text (`column_census.py:11` and `:113-114`:
  *"w_differs_from_column_oracle READS THE REFERENCE and is descriptive only"*).
  Those 1,245 include argos, samothraki, sparta
  and xylokastro.
- **`eval/controlled_eval/exp_majority_taxonomy.py:365`** calls `load_substrate()` bare.
  `results_majority_taxonomy.json` header: `"n_windows": 247, ... "n_cities": 10`;
  `"n_wrong": 1719`. The linguistic buckets the plan uses to decide what can ship
  (function words 19.9%, named entities 6.4%), the 41.8% zero-marginal-value figure and
  the 39.6% not-cleanly-a-selection-failure figure are all computed over those 1,719.
  The ledger record for `exp-2026-08-17-majority-error-taxonomy` states the entity
  funnel spans **"52 meetings, all 10 cities"**.
- **`eval/controlled_eval/exp_char_homophone.py:169`** calls `load_substrate()` bare and
  evaluates `fold="city"` over **10** folds (`:197`, `results_char_homophone.json`:
  `"n_folds": 10`). This is where the "`exact_2_of_3` holds 25.0% of the gap" figure —
  the reason F1 exists at all — was measured.
- **`eval/controlled_eval/exp_composition.py:295`** asserts `len(items) == 247` and does
  leave-one-out by `city_id` (`:674`).

`exp_majority_taxonomy.py:13` carries the sentence *"and the autoresearch confirmation
partition is untouched."* **That sentence is wrong**, in the sense that matters here.
It is true that the script never called the harness API. It is false that the
confirmation cities' reference labels did not enter the numbers the script produced —
they are 27,665 of its 74,917 tokens. The same claim is repeated in the ledger caveat
for that record and in [`docs/reports/2026-08-17-majority-error-taxonomy.md:9-10`](2026-08-17-majority-error-taxonomy.md)
(*"nothing here went through a promotion gate — the autoresearch confirmation partition
was not read"*). Those three places need the correction this report supplies.

### 2. Earlier experiments quoted per-city outcomes over all ten cities

Not a single leak — a habit. From the ledger conclusions:

- `exp-2026-08-16-composition-over-selection`: *"leave-one-out over windows, meetings
  AND cities produces zero sign flips"*, `208 of 247 windows improve`. A per-city
  result over all 10.
- `exp-2026-08-17-name-repair-on-w` (record `exp-2026-08-11-name-repair`): *"LOO over
  windows, meetings and cities gives no sign flip; **9 of 10 cities negative**"*, and
  D and I byte-identical in *"ZERO of 247 windows"*.
- `exp-2026-08-16-char-vote-homophones`: *"**5 cities better / 4 worse**, zero LOO sign
  flips"*.
- `exp-2026-08-16-roster-grounded-selection`: *"No single window (6.7%), meeting (6.8%)
  or city carries it"* over 247 windows.
- `exp-2026-08-16-fusion-deletions`: paired swap over *"the overwhelming majority of 247
  windows"*.

By the plan's own count this is the **seventh pass** over these 247 windows. Confirm
outcomes have been visible in aggregate, per city, and per leave-one-city-out fold in
every one of them.

Codex's qualification is worth keeping: the mere existence of these records does not
contaminate *every* future hypothesis. It contaminates F1 because F1's design decisions
are traceable to specific numbers in them.

### 3. Term lists — a second, narrower channel

The four confirmation cities all have frozen term files
(`research/ds_wer/terms/{argos,samothraki,sparta,xylokastro}.json`). Their provenance is
mixed, and the honest answer is "partly".

**Clean:** the v1 base lists are built from the OpenCouncil public registry.
`research/ds_wer/terms/argos.json` `source.note`: *"OpenCouncil public city registry.
**No transcript text, no provider hypothesis.**"* The rule adds *"building the list from
transcripts would fit the metric to the data it scores"*. `argos.v2.json` `rule.never`
repeats it: *"no term is taken from, or filtered by, any model output, decode, provider
hypothesis or error analysis"*.

**Not clean, two ways:**

- `eval/controlled_eval/roster_lexicon.py:12-14` documents a third merged source: *"an
  explicitly chosen slice of the 147 **error-mined** candidates of
  `exp-2026-08-16-error-mined-terms`"*. Those were mined from `data/asr/export.jsonl`,
  the human-correction corpus, which overlaps this benchmark. Counting
  `eligible_cities` in `data/glossary/candidates_error_mined.json`: **sparta 15,
  xylokastro 10, samothraki 7, argos 6 — 38 candidates routed to confirmation cities.**
- `argos.v2.json` `rule.common_word_stoplist.rule_a` rejects an alias when it *"occurs
  >= 5 times in the 220 benchmark windows (66912 tokens) of the eight cities that are
  NOT argos/orestiada"* — a set that includes samothraki, sparta and xylokastro. It is
  a rejection-only filter, not target fitting, and it rejected exactly 1 alias at the
  frozen threshold. Real but small.

The wording the plan asked for, in its context —
`docs/reports/2026-08-17-majority-error-taxonomy.md:229`, in the "what this does not
show" list:

> That the term lists estimate production coverage. They were mined from material
> overlapping this benchmark, so they are **optimistic and leaky**.

Repeated verbatim as a ledger caveat on `exp-2026-08-17-majority-error-taxonomy`.

**Scope of this channel.** Per Codex: this is *resource* contamination, distinct from
the *hypothesis-selection* contamination of finding 1. If F1 neither uses the term lists
nor was designed from their results, it does not independently contaminate F1's
function-word arm. It does contaminate **any name or entity measurement on confirm**,
and does so in the optimistic direction. Since the plan already restricts names to
descriptive reporting (§4 — the funnel caps name-targeted gain at 28-41 edits against a
75-edit floor), this changes no decision, but it must ride with any entity number.

### 4. The split is by city, and that part is sound

`autoresearch.py:102-103` fixes the two city tuples; `_restrict` (`:203`) filters
`w.city in cities`. A city partition is strictly coarser than a meeting partition, so no
meeting and no window can straddle the boundary. `plan_partition` (`:131-151`) reads
only `len(w.ref)` per city — reference token counts, no WER, no outcome. `assert_partition`
(`:153-159`) fails if the recomputed rule and the pinned tuples ever disagree.

This is the one question in §0 with a clean answer. It is also the only one that could
not have been lost by reading, which is why it survived.

### 5. The harness itself never read confirm

`research/autoresearch/journal.jsonl` holds **33 records: 16 `REGISTERED`, 16
`SEARCH_RESULT`, 1 `DUPLICATE_REFUSED`**. **Zero `CONFIRM_BATCH_FROZEN`. Zero
`CONFIRM_RESULT`.** Every one of the 16 search results carries `"partition": "search"`,
`"n_windows": 153`, `"n_meetings": 103`.

The five overlap-speaker arms appended later (`ov_mask_select`, `ov_turn_select`,
`ov_turn_select_placebo`, `ov_turn_compose`, `ov_turn_compose_placebo`, journal seq
25-33, `git_head 437993d9`) are the source of the 11-vs-16 discrepancy against the
`exp-2026-08-16-autoresearch-harness` record, which was written when the journal held
11. **All five ran on 153 windows / 103 meetings.** None touched confirm.

So the harness report's claim that the partition "was never read" is **true as stated
about the harness** and true about `run_confirmation`. It is not true as a claim about
analyst knowledge, and the harness report knows this — see §6.

### 6. The project already made this exact call once

This is not a new standard being invented for F1.
`docs/reports/2026-08-16-overlap-speaker-arms.md:80-85`:

> The hypothesis under test here — "speaker information helps inside detected overlap" —
> was *generated* by the parent experiment's measurement on all 247 windows, the four
> sealed confirmation cities included. **Hiding those cities during implementation does
> not make them unseen.** Spending a one-way confirmation door on a hypothesis read off
> the confirmation data would buy nothing, so §1.1 of the preregistration ruled it out
> in advance.

And the preregistration itself,
[`2026-08-16-autoresearch-partition-prereg.md:193-197`](../specs/2026-08-16-autoresearch-partition-prereg.md),
under "what this protocol does NOT buy":

> **It does not undo adaptive proposing.** The ideas are proposed by agents that have
> already seen five passes over these same 247 windows, confirmation cities included.
> Sample splitting controls the multiplicity of *testing*, not the adaptivity of
> *hypothesis generation*.

F1 is a stronger case than the overlap arms, not a weaker one. The overlap hypothesis
came from one prior measurement. F1's eligibility class, its success condition
("abstain on ~40%"), and the elimination of its two siblings each cite a specific
reference-conditioned statistic computed with confirmation labels in the denominator.

## Why not (c)

A holdout must be unread, and it must be large enough.

- **Unread.** All 247 windows have been through at least six reference-conditioned
  passes. Carving four windows out of material already analysed does not make them
  unread; meeting-level carving prevents row overlap but cannot restore information
  independence (Codex, A).
- **Large enough.** The ship floor is H₀: ΔWER ≥ −0.0010, ≈ **75 net edits on 74,917
  tokens**. On the confirm partition's 27,665 tokens that floor is ≈ 27.7 edits, and
  even there `exp-2026-08-16-harness-mde`
  ([report](2026-08-16-harness-coverage-mde.md)) had to add effect and support floors
  because a monotone arm touching 4 of 61 meeting clusters excites a percentile CI at
  any effect size.
- **The only genuinely unread material** is the 7 sealed temporal-holdout windows of
  `eval-freeze-2026-08` — **2,101 reference tokens**, 2.8% of the substrate. At the ship
  floor that is **2.1 edits**. It cannot test anything, and it is sealed by a hard rule
  no arm has ever passed a gate to release.

There is no third option here. Manufacturing one would be the eighth pass wearing a
different hat.

## Verdict

**(b).** Precisely, and the precision matters (Codex, D): *CONFIRM is invalid **as
confirmation for F1***, not universally poisoned. A genuinely independent hypothesis
whose design demonstrably did not depend on the prior analyses could in principle still
use it — but demonstrating that now would itself be hard, and F1 is not such a
hypothesis.

Therefore:

- F1 may be built and run. **No confirmation batch is frozen. No confirmation is spent.**
  The budget stays at 5 of 5 and the one available batch stays available.
- Any F1 number is **exploratory / descriptive**. Intervals may be reported and must be
  labelled descriptive — never confirmatory, never gate-valid, never multiplicity-controlled.
- F1 may not be described as having passed the ship gate, whatever it measures.
- Any entity or name number carries the finding-3 caveat in the same sentence.

The strongest honest claim available: *"on this fixed, previously analysed benchmark, a
frozen F1 produced an exploratory ΔWER of X, with meeting and city heterogeneity Y."*

If F1 runs, the exploratory framing is worth something only if it is treated as a
prospective-design exercise: freeze the composer, prompt, thresholds, abstention policy
and analysis before evaluating; ablate the leaky term resources out; report abstentions,
failures, net edits and per-city heterogeneity; and use the result for effect-size and
power planning against a future sealed set, not as a ship decision.

## Secondary question: how many confirmation batches?

**The reviewer who reads it as one batch ever is right, and the report does not actually
disagree with them.**

`autoresearch.py:802-830`, `freeze_confirmation_batch`:

```
batches = [r for r in recs if r["type"] == CONFIRM_BATCH_FROZEN]
if batches:
    raise ValueError(... "a second family would break the familywise guarantee. "
                         "Bump PROTOCOL_VERSION to start a new cycle.")
...
if len(keys) > CONFIRM_BUDGET:      # CONFIRM_BUDGET = 5, autoresearch.py:112
    raise ValueError(...)
```

Two separate guards. The first permits **exactly one frozen batch per
`PROTOCOL_VERSION`** (currently `autoresearch-2026-08-16b`, `:97`). The second caps that
one batch at **5 ideas**, tested jointly under Holm.

So "a budget of 5" (`docs/reports/2026-08-16-autoresearch-harness.md:10`) means five
*ideas inside the single batch*, not five batches. The same report says so explicitly
nine lines later (`:67-73`): *"Exactly one confirmation batch per cycle... five
sequential singleton batches, each Holm-corrected inside itself, give a familywise error
of 1 − 0.95⁵ = 22.6%, so a second batch is refused outright."* The 22.6% scenario is
exactly what the code forbids, not what it permits. There is no conflict between the
code and the report — only an ambiguity in the one-line denominator summary, which this
paragraph resolves.

## What this audit does not show

- That any *result* on the confirmation partition is wrong. No confirmation result
  exists.
- That the harness is broken. It did what it was built to do; the leak is upstream of
  it, in scripts that never called it.
- That the term lists are unusable. Their v1 base is externally sourced and documented
  as such; only the error-mined slice and one stoplist rule touch benchmark material.
- That F1 is a bad idea. It says only that F1 cannot buy a confirmatory CI here.
- Anything about fidelity-to-audio. Every number cited is
  agreement-with-OpenCouncil.
