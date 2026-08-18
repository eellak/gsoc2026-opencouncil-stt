# F1 (LLM majority arbiter) — frozen preregistration

Status: **FROZEN 2026-08-17, before any F1 outcome number existed.**
**Amendment 1, 2026-08-17** (Codex job `c854a11a`, sol @ high), applied below and
listed in §9. At amendment time **zero F1 model answers existed**: no question had been
sent to any model, the inference cache was empty, and no WER had been computed. The
amendment is a tightening of the firewall, the outcome partition, the order control,
the cache key and the power arithmetic — it changes no hypothesis and relaxes nothing.

**`EXPLORATORY_CONTAMINATED_NOT_CONFIRMATORY`** — failed audit
`exp-2026-08-17-confirmation-audit`. This stamp appears in every artifact produced
under this document.
Governing plan: [`2026-08-17-llm-composer-draft.md`](2026-08-17-llm-composer-draft.md)
revision 2. §0 of that plan is binding on this document.

Experiment record: `exp-2026-08-17-llm-f1-arbiter`.
Report: [`../reports/2026-08-17-llm-f1-arbiter.md`](../reports/2026-08-17-llm-f1-arbiter.md).

## 0. Standing of every number produced under this document

`exp-2026-08-17-confirmation-audit` established that F1's eligibility class, its
abstention success condition and the elimination of families F2/F3 are each traceable
to reference-conditioned quantities computed with the autoresearch confirmation
partition in the denominator. Therefore:

- **No confirmation batch is frozen. No confirmation is spent. The budget stays 5 of 5.**
  The autoresearch API (`freeze_confirmation_batch`, `confirm_*`) is **not called** by
  any code written for this experiment, and no `CONFIRM_BATCH_FROZEN` or
  `CONFIRM_RESULT` journal record is written.
- **Every interval reported is descriptive.** Never confirmatory, never gate-valid,
  never multiplicity-controlled. F1 may not be described as having passed the ship
  gate whatever it measures.
- The deliverable is the **prospective-design and power/effect-size planning output**
  for a future sealed evaluation, not the WER number.
- Any entity or name number carries the term-list leak caveat in the same sentence:
  the term lists and rosters were mined from material overlapping this benchmark, and
  the source reports call them "optimistic and leaky"
  (`docs/reports/2026-08-17-majority-error-taxonomy.md:229`).

## 1. Substrate, frozen

- Benchmark run `2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`, trio
  `scribe-v2-clean` / `soniox` / `oc-runpod-fixed-2026-08-10`.
- 247 windows, 144 meetings, 10 cities, 74,917 reference tokens, after the 6 sealed
  eval-freeze holdout windows are removed by `fusion_lab.load_substrate()`.
- Baseline **W = 0.10046**, the hierarchical per-column vote of
  `exp-2026-08-16-composition-over-selection`.
- References are **agreement-with-OpenCouncil**, never fidelity-to-audio.
- **MSA tie-breaking is frozen**: the exact 3-way sum-of-pairs alignment of
  `eval/controlled_eval/msa.py` at its current sha256, cached at
  `~/.cache/oc-public/fusion_lab/align_65b1c4d64618a429.json`. `msa.py` **is not
  modified**; `fusion_lab._cache_path()` keys on its sha256 and an edit would
  invalidate the 18 MB cached alignment. The cache filename is verified unchanged at
  the end of the run.
- The 247 windows are reported both pooled and split by the autoresearch city
  partition (`SEARCH_CITIES` = athens, chalandri, chania, orestiada, vrilissia,
  zografou; `CONFIRM_CITIES` = argos, samothraki, sparta, xylokastro). The split is
  reported **for heterogeneity description only**. It is not a search/confirm
  inference and the confirm side is not a held-out test here.

## 2. Eligibility, and it must be reference-blind

**Eligible = every MSA column whose `column_classes.column_class` is `exact_2_of_3`.**
That is **6,645** columns over the 247 windows (verified census, this document's
freeze). Nothing else.

The eligibility function reads only: the three aligned token entries of the column
(presence and string identity). It does **not** read `Window.ref`, the oracle, the
taxonomy labels, or any per-column outcome. Revision 1 of the plan repeatedly spoke of
"the 1,245 wrong majorities"; that set is knowable only from the reference and using it
would label-leak the experiment.

This is enforced two ways and both are reported:

1. `eligible_columns(window)` takes the column list only — the reference is not in
   its signature.
2. A test (`test_llm_arbiter.py::test_eligibility_is_reference_blind`) shuffles /
   replaces `Window.ref` with garbage and asserts the eligible set is byte-identical.

The 133 eligible columns that are also `split_merge` are **kept** (the plan says all
6,645) and reported as a descriptive stratum. A `split_merge` column maps to exactly
one W token like any other eligible column; the boundary disagreement it belongs to is
a property of the *neighbouring* column pair, not of this column's own W span.

**Executable eligibility assertions (Amendment 1).** Every eligible column must satisfy,
and a violation **hard-fails the run** rather than becoming a model `invalid`:

- three non-epsilon entries, exactly two of them the same string and one distinct;
- W emitted the majority token at that column (`decisions[i]["token"] == majority`);
- the column maps to exactly one index in the W token stream;
- no two overrides in one window map to the same W index (collision check at apply
  time).

### 2.1 Reference firewall — two stages, sealed between them

The run is split into two processes that cannot see each other's inputs:

- **Inference stage.** Builds questions, calls the model, writes answers. It never
  loads `Window.ref`, the oracle, any taxonomy label, any scorer output or any
  per-column correctness. `build_questions()` and `eligible_columns()` do not take a
  reference in their signatures. The MSA itself (`msa.align3`) takes three hypotheses
  and no reference, so column identity and class are reference-free upstream too.
- **Seal.** The question set and the resolved per-column decisions are written to
  `$SC/llm_arbiter/` and hashed (sha256 over the canonical JSON). The hashes go into
  the report.
- **Analysis stage.** Only now is the reference loaded, and only to score.

The blindness test replaces `Window.ref` with garbage and asserts the eligible set, the
question set and the rendered prompts are byte-identical.

## 3. The question, frozen

For each eligible column: two distinct candidate tokens exist — the **majority** token
(held by two systems) and the **minority** token (held by one). The model is asked
which belongs in the slot, or to abstain.

- **Masked slot.** The context is the composed W token stream of that window, with the
  decision position replaced by `_____`, and `CTX = 20` tokens on each side (clipped at
  window edges). The already-composed W token at the slot is **never shown**, so its
  current token cannot look grammatically inevitable.
- **Only existing token candidates. No epsilon, no generated text.** The model returns
  a label, never a word.
- **Neutral labels Α / Β**, inheriting the convention of `exp_fusion.py:128`
  (`order_for`, `LABELS`). Provider names, system counts and majority/minority status
  are never shown.
- **Abstain (`ΑΠΟΧΗ`) is an explicit modal option in the schema**, not a fallback.
- Context also carries the meeting's closed term list (roster-gated per-meeting list
  from `roster_lexicon.build_meeting_context`, `seen_freq = Counter()` — that argument
  feeds only the name-repair stoplist and is not consulted when the term set is built).
  Agenda-item text is **not available** in the benchmark record; the field is omitted
  rather than faked.
- **The term list is prompt leakage, not merely a metric caveat (Amendment 1).** Part
  of it (the 64 admitted error-mined candidates) was mined from material overlapping
  this benchmark. F1 as run is therefore explicitly a **contaminated auxiliary-context
  arm**: the model is given a resource that partly derives from this benchmark's own
  references. It is kept because the governing task requires the roster and per-city
  term list in context, and because removing it would not make the *design* of F1
  clean — the family was selected by reading a reference-conditioned taxonomy either
  way. Provenance and sha256 of every input list are recorded in the results JSON, and
  a future sealed evaluation must source its lists independently and freeze them before
  benchmark access.

### 3.1 Both candidate orders, every question

Every question is asked **twice**:

- Which of (majority, minority) occupies label Α in **pass 1** is fixed by
  `sha256(question_id)` parity — reference-blind, frozen, reproducible, and
  decorrelating position from majority status across the corpus. Pass 2 is the exact
  swap.
- **Amendment 1: the two passes have identical batch membership and identical
  within-batch question order.** Only the Α/Β candidate mapping differs. Revision 0 of
  this document shuffled the batches differently per pass, which confounds candidate
  order with batch context; that is corrected.
- The decision is accepted **only if the two passes name the same token**. Otherwise
  the column is `order_disagree`.

Order randomisation alone is cosmetic — the documented bias is *content* preference
(shorter, smoother), not position. Order-invariance is the countermeasure; the
per-question randomisation is hygiene for the residual "pick the first one" anchor.

**Honest naming (Amendment 1).** The bridge exposes no deterministic decoding, so a
two-pass disagreement mixes true order sensitivity with ordinary sampling noise. The
control is therefore reported as a **two-pass consistency filter**, and an **A/A
replicate** on the pilot sample (same candidate order asked twice) estimates the
stochastic disagreement floor. Order sensitivity is only the excess of the A/B
disagreement rate over the A/A rate, and is reported as such.

### 3.2 Outcome per eligible column, frozen partition

Applied **in this precedence order**, which makes the five outcomes exhaustive and
mutually exclusive over the 6,645:

1. `invalid` — either pass's answer is missing, unparseable, not one of
   `{Α, Β, ΑΠΟΧΗ}` after label normalisation, or belongs to a batch whose transport
   failed and was never re-answered. No change to the transcript.
2. else `abstain_explicit` — either pass returned `ΑΠΟΧΗ`. No change.
3. else `override` — both passes resolve to the **minority** token. **W's token is
   replaced.**
4. else `confirm` — both passes resolve to the **majority** token. No change.
5. else `order_disagree` — the two passes resolve to different tokens. No change.

**Frozen parsing rules.** The answer array is parsed by `parse_json_array`
(inherited, `exp_composition.py:200`). Latin `A`/`B` are normalised to Greek `Α`/`Β` —
that is a transcription of the same choice, not a repair of a wrong one. Everything
else, including a missing id, a non-string pick, a third label, or extra prose, is
`invalid` for that pass. If an id appears **more than once** in one answer array the
question is `invalid` for that pass (no first-wins, no last-wins). Ids not asked for
are discarded and counted. A batch that raises is **not cached**, and its questions are
retried once; questions still unanswered when the run ends are `invalid` with their
count reported.

**Application invariant (Amendment 1).** Every `override` must alter its intended W
index exactly once. The run reports `override_decisions`, `overrides_applied`,
`mapping_failures` and `collisions`, and **asserts the first two are equal and the last
two are zero**.

**Rates, each with its denominator stated (Amendment 1). They are reported separately
and never merged into one flattering number:**

| rate | numerator | denominator |
|---|---|---|
| `explicit_abstention_rate` | `abstain_explicit` | 6,645 |
| `order_instability_rate` | `order_disagree` | 6,645 |
| `invalid_rate` | `invalid` | 6,645 |
| `override_rate` | `override` | 6,645 |
| `confirm_rate` | `confirm` | 6,645 |
| `operational_non_decision_rate` | `abstain_explicit + order_disagree + invalid` | 6,645 |

Only `confirm` and `override` are **valid model decisions**. An `invalid` or an
`order_disagree` leaves W unchanged but is **not** evidence that the model knowingly
abstained, and the report may never present it as such. A pass-level invalid rate is
reported alongside the column-level one.

## 4. Frozen interpretation rule (descriptive, not a gate)

Preregistered before any outcome. This is a **reading rule for an exploratory result**,
not a ship gate — no F1 result can pass a ship gate under §0.

F1 **meets the preregistered descriptive heuristic** iff all three hold:

- `explicit_abstention_rate ≥ 0.30`, and
- `override_rate ≤ 0.25`, and
- `invalid_rate ≤ 0.02` (without which neither of the first two is readable).

Note the first condition uses **explicit** abstention only. Transport failures and
two-pass inconsistency cannot be spent to buy the heuristic.

**Honesty note frozen with the rule (strengthened by Amendment 1):** these two
thresholds are **normative descriptive heuristics chosen by the analyst**, not an
extrapolation from any measured quantity, and the report must say so. The plan's
"abstains on roughly 40%" figure is the taxonomy's 39.6–41.8% share of the **1,719
wrong majorities** — roughly 681–719 columns, which is only **10.2–10.8% of the 6,645
eligible columns** even under an unjustified assumption of full overlap. It supplies no
empirical basis for 30%. What the figure does support is the direction: a model that
overrides frequently is the shortest-picker in new clothes, because three text-only
override rules already came back significantly negative (+0.16 to +0.27 WER points,
worse in all six search cities).

Three text-only override rules already came back significantly negative
(+0.16 to +0.27 WER points, worse in all six search cities), which is the prior this
rule encodes.

## 5. Model and call shape, frozen

- `gpt-5.6-luna` at `high` effort, through the codex bridge only:
  `enqueue exec -c model=gpt-5.6-luna -c model_reasoning_effort=high`. The model is
  passed **explicitly** because `high` otherwise routes to `sol`, and `xhigh` is
  normalised to `high` outside the execute lane.
- **No `--timeout` is passed.** The bridge's per-effort default applies and `wait` is
  called with no second argument.
- Concurrency 3 (the bridge's read-only ceiling).
- **Cache key = sha256 of the complete wire request (Amendment 1):** question id,
  prompt version, preamble text, the full rendered batch (membership **and** order),
  batch size, the question's own candidate mapping, model, effort, seed and pass
  number. Prompt iteration therefore cannot silently reuse stale picks, and answers
  obtained under a **rejected pilot batch size are never mixed into the production
  run** — their request differs, so their key differs.
  Batch assignment is computed once over the **full frozen question list**, not over
  the outstanding set, so a resumed run reproduces byte-identical batches and only the
  unanswered ones are re-asked.
  Cached answers are matched by the question id embedded in the model's reply;
  duplicate or unknown ids are rejected, not silently coerced.
  The inference cache stores **no reference, no correctness label and no score**.
  **Transport failures are never cached as no-ops** — that bug already inflated an
  invalid count once in `exp_composition.py`. A failed batch leaves its questions
  uncached and is re-asked once; questions still unanswered when the run ends are
  counted `invalid`, with their denominator reported.

### 5.1 Batching pilot — decided before the full run, on invalid rate and wall clock only

The claim that batching 24 questions caused the previous arm's 6.8% invalid rate is
**untested**. Before the full run, a pilot is run and the production batch size is
chosen by a frozen rule. Exact specification (Amendment 1):

- **Sample: exactly 120 questions**, drawn reference-blind and deterministically as the
  120 smallest `sha256("f1-pilot|" + question_id)` over the 6,645. No reference, no
  outcome, no WER enters the draw.
- **Conditions: batch sizes {6, 12, 24, 48}.** The same 120 questions and **both
  passes** are run at every batch size. 48 is included because wall clock is the
  binding constraint at 13,290 decisions.
- **Invalid rate is pass-level**: (pass-answers that are missing / unparseable / not a
  legal label) / 240 per condition. The column-level rate is reported beside it.
- **Selection rule, frozen:** take the **largest** batch size whose pass-level invalid
  rate is `≤ 0.02`. If none qualifies, take the batch size with the lowest invalid
  rate; ties broken toward the **larger** batch size. **Wall clock is reported, never
  used by the selection rule.**
- **Transport:** a failed batch is retried once; a batch that fails twice is recorded
  and its questions counted invalid for that condition. Retry limit 1, frozen.
- **A/A replicate:** the same 120 questions are additionally asked a third time in the
  pass-1 candidate order at the selected batch size, to estimate the stochastic
  disagreement floor (§3.1).
- Pilot answers are **not** reused in the production run: their wire request differs,
  so their cache key differs.

**Throughput probe, disclosed (2026-08-17).** Partway through the pilot's batch-6
condition the measured throughput (~0.9 questions/minute/lane) implied ~82 hours for
the production run, so three single-job **timing probes** were run at batch 48 / 96 /
192-slot before deciding whether the frozen pilot was affordable at all. They measured
wall clock and invalid count only — no reference, no WER, no eligibility change — and
their answers were **never written to any cache**, so they enter no reported outcome.
Result: 31.2 / 28.5 / 22.8 questions-per-minute-per-lane, i.e. per-question cost is
dominated by fixed per-job overhead and large batches are ~30× more efficient. The
frozen pilot was then **resumed unchanged**; no condition was added or dropped. The
probes asked 144 of the 120 pilot questions plus 120 others once, outside the cache;
that is recorded here rather than hidden.

Wall clock is the binding constraint, not tokens: bridge p50 ≈ 62 s per job, p95 ≈
500 s, at 3 concurrent. One decision per call would multiply wall clock by the batch
size and is not affordable at 13,290 decisions.

## 6. Analysis, frozen before any evaluation

- **Primary: frozen benchmark WER on the full composed transcript.** Every resulting
  transcript is scored end-to-end through `fusion_lab.evaluate` with the frozen scorer.
  Column deltas are **not summed**: WER is globally aligned and non-additive, so
  `W ≠ oracle` counts are diagnostics only and the "ceiling" figures must not be used
  inferentially.
- The F1 idea has **no fitted parameter**, so leave-one-city-out is vacuous by
  construction and `fusion_lab` says so in `fold_note`.
- **Mandatory secondary:** named-entity error rate (DS-WER of `scripts/ds_wer.py`
  against the frozen per-city v1 term lists in `research/ds_wer/terms/`), plus the
  insertion rate and the deletion rate.
- **Descriptive bucket breakdown** of the overridden columns: function words, names
  (entity), numerics, morphology (surface-suffix neighbours), other content words —
  reusing `exp_majority_taxonomy.py`'s `FUNCTION_WORDS`, `is_numeric`,
  `surface_suffix_neighbor` and the term lexicons unchanged. **Frozen precedence, first
  match wins:** `name_entity` → `numeric` → `morphology` → `function_word` →
  `other_content`, applied to the (majority, minority) pair. Every outcome class is
  bucketed, not only the overrides, so the denominators are comparable.
- **Floor arithmetic, frozen and binding on the reading:** the ship test is
  H₀ ΔWER ≥ −0.0010, i.e. ≈ **75 net edits** on 74,917 tokens. Perfect play on named
  entities yields **41** edits and the roster funnel caps one-token replacement at
  **28**. A name-targeted result therefore **cannot clear the floor even played
  perfectly** and may only be reported descriptively, never gated. Function words are
  **276** edits and are the only bucket that could carry a shippable effect — and a
  gain there is largely **OpenCouncil house orthographic style normalisation**, never
  to be headlined as ASR improvement.
- **Reported without exception:** abstentions, invalids, order-disagreements, every
  eligibility denominator, per-meeting and per-city domination, and the denominator of
  everything attempted including failures.
### 6.1 Power / effect-size planning output — the actual deliverable

Frozen mathematically (Amendment 1). For meeting *m*: `n_m` reference tokens,
`e_Wm` baseline edit count, `e_Fm` F1 edit count, `d_m = e_Fm − e_Wm`. Then

    δ̂  = Σ d_m / Σ n_m
    SE² = ( M / (M−1) ) · Σ (d_m − δ̂·n_m)² / (Σ n_m)²

with M the number of meetings (144). Meeting-level scoring is asserted to aggregate to
the reported pooled WER delta.

Frozen inferential planning conventions: **one-sided α = 0.05, power 0.80, margin
δ₀ = −0.0010**, i.i.d. replication of the observed meeting-size and city mix. Then

- if `δ̂ < δ₀`:  `K_req = ceil( M · ( (z₀.₉₅ + z₀.₈₀)·SE / (δ₀ − δ̂) )² )` meetings,
  converted to token mass through the frozen mean tokens per meeting. **Cluster count
  is the primary planning number**, because token mass alone does not determine power.
- if `δ̂ ≥ δ₀`:  required mass is reported as **undefined / unbounded**, not as a
  large finite number.
- at the mass actually available: `δ_MDE = δ₀ − (z₀.₉₅ + z₀.₈₀)·SE`.

**Machine-readable planning block**, emitted to `results_llm_arbiter.json` so a future
sealed evaluation can use it without re-deriving anything: per-meeting `n_m, e_Wm,
e_Fm, d_m`; the totals; the residual sum of squares; `SE`; the sign convention; the
margin gap `δ₀ − δ̂`; `z` critical values; `K_req` in meetings and in tokens; `δ_MDE`
in WER points and in edits; rounding rules; the city mix; and a **sensitivity grid**
over assumed true effects (not only the unstable observed-effect plug-in), so the
planner is not hostage to one noisy δ̂.

This is planning arithmetic for a hypothetical future sealed evaluation. **It is not a
test.** No p-value, no significance claim and no gate result is emitted for F1.

## 7. Scope of any conclusion, frozen

One fixed trio, one benchmark, one realization. Conditional on that, and unable to
establish anything about ASR systems generally. This is the **seventh** reference-
conditioned pass over the same 247 windows. Per-seed WER spread on training is **2.1
points**, larger than the effect sought. The search/confirm split would absorb test
multiplicity, not the adaptivity of having proposed the family by reading a
reference-conditioned taxonomy — and here it does not even do that, because no
confirmation is spent.

## 8. Files

- `eval/controlled_eval/exp_llm_arbiter.py` — the arm, the questions, the run.
- `eval/controlled_eval/test_llm_arbiter.py` — reference-blindness, outcome partition,
  order-invariance, cache-key and no-fallback tests.
- `eval/controlled_eval/results_llm_arbiter.json` — counts and labels only.
- `~/.cache/oc-public/llm_arbiter/` — prompts, raw answers, verbatim council speech.
  **Never in git.**
- `eval/controlled_eval/run_llm_arbiter.py` — the two-stage runner (inference, seal,
  analysis) and the pilot.
- **Not touched:** `eval/controlled_eval/msa.py`, `eval/gold_set_score.py`, anything
  named `*insertion_fidelity*`, `eval/controlled_eval/autoresearch.py`.

## 9. Amendment 1 (2026-08-17) — what changed, and it changed no hypothesis

Applied after Codex review (`c854a11a`, sol @ high), before any model answer existed.

1. Two-stage reference firewall with a sealed, hashed question set and decision set
   between inference and analysis (§2.1).
2. Executable eligibility assertions; mapping failures hard-fail instead of becoming
   model invalids (§2).
3. The term list is named as **prompt leakage** and F1 as a **contaminated
   auxiliary-context arm**, with provenance and hashes recorded (§3).
4. Outcome partition rewritten as an explicit precedence list, with frozen parsing
   rules for duplicate ids, extra prose and partial batches (§3.2).
5. Application invariant: `override_decisions == overrides_applied`, zero collisions
   (§3.2).
6. Every rate reported separately with its own denominator; failures may not be spent
   as evidence of knowing abstention (§3.2, §4).
7. Both passes now share identical batch membership and order; the control is renamed
   a **two-pass consistency filter** with an A/A replicate to estimate the stochastic
   disagreement floor (§3.1).
8. Cache key expanded to the complete wire request; batch assignment computed over the
   full frozen question list so resumes are byte-identical (§5).
9. Pilot made exact: 120 deterministic reference-blind questions, batch sizes
   {6, 12, 24, 48}, pass-level invalid rate, retry limit 1, wall clock reported but not
   a selection criterion (§5.1).
10. §4 thresholds relabelled as **normative descriptive heuristics** with the
    10.2–10.8% arithmetic spelled out; `invalid_rate ≤ 0.02` added as a readability
    precondition.
11. Power arithmetic frozen mathematically with a machine-readable planning block and a
    sensitivity grid (§6.1).
12. `EXPLORATORY_CONTAMINATED_NOT_CONFIRMATORY` stamped into every artifact; "ship
    test" is replaced by "hypothetical future sealed-evaluation margin"; the word
    "confirm" is reserved for the per-column decision label and used nowhere else.

Not adopted: Codex's suggestion to remove the term list entirely (the governing task
requires roster and per-city term list in context — it is labelled instead), and its
suggestion of a deterministic paired decode (the bridge exposes no such control —
handled by renaming the control and adding the A/A replicate).
