# W-rt: does Soniox per-word confidence beat the plain per-column vote?

Preregistration, **revision 2** (Codex job `b71f2dca0cad451db62cfb8f65e9d08e`, high
effort, reviewed revision 1 before any arm was implemented or scored). 2026-08-16.
`exp-2026-08-16-w-rt-confidence`.

Written **before** any arm existed and before any WER number of any arm was computed.
Zero GPU, zero paid API.

**Status of this experiment, decided up front on Codex's first correction: this is a
DEVELOPMENT / hypothesis-generating run, not a confirmatory one.** The arm classes and
eligibility rules were chosen by a human who has already seen five reference-informed
passes over these same 247 windows; leave-one-city-out fitting and a Bonferroni
adjustment over three arms do not price that. **No arm ships on this run**, whatever it
scores. What a passing arm earns is a named confirmation obligation (§9), not
deployment.

Predecessors: [`exp-2026-08-16-soniox-confidence`](../reports/2026-08-16-soniox-confidence-probe.md)
(the signal exists on the gold set),
[`exp-2026-08-16-composition-over-selection`](../reports/2026-08-16-char-vote-homophones.md)
(W, the per-column vote), [`exp-2026-08-16-char-vote-homophones`](../reports/2026-08-16-char-vote-homophones.md)
(the column census and hindsight replays this arm list is drawn from).

## 1. Why this is a new substrate and not a new arm on the old one

The cached Soniox text for the 247 benchmark windows was produced by the paid
`stt-async-v5` and carries **no confidence values** — the client discarded them. The
only free path is the realtime model `stt-rt-v4`. Different model, therefore different
text. Soniox is one of W's three voters, so re-running it **changes W itself**.

Attaching new confidences to the old cached text is forbidden here: the probe measured
only 14 of 27 gold-set cells reproducing token-identically between two runs of the
*same* model, so a token-by-token graft across two *different* models has no defensible
alignment.

So this experiment builds a **parallel substrate, W-rt**, and every number it reports is
measured inside it:

- The same window set the frozen evaluator scores: **247 windows after removing the 6
  sealed temporal-holdout windows** of `eval-freeze-2026-08` (which stay sealed and are
  never transcribed, aligned or scored here). 247 / 144 meetings / 10 cities / 74,917
  reference tokens are all **post-removal** totals and are asserted in code.
- `scribe-v2-clean` and `oc-runpod-fixed-2026-08-10` byte-identical from the cached
  benchmark report `2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`.
- The `soniox` arm replaced by a fresh `stt-rt-v4` transcription of the same window
  audio (`$SC/bench_windows/<item_id>.wav`), carrying per-token confidence.
- The whole-window consensus pivot recomputed from the new trio (`consensus_pick`).
- The alignment recomputed (`msa.align3`, same band rule, same frozen transition order)
  and **W-rt** composed by the same unmodified hierarchical vote (`msa.compose`).

A **manifest** is frozen with the run: item / meeting / city IDs, the SHA-256 of each
window WAV, of each cached token file, and of the reference text, plus the asserted
totals above.

**Nothing existing is modified.** New cache paths only
(`~/.cache/oc-public/composition-rt-2026-08/`), new result files, no edit to
`fusion_lab.py`, `msa.py`, `column_classes.py`, `scoring.py`, or any existing cache.

### Old-W facts do not transfer to W-rt

The census numbers quoted below as motivation (80,659 columns, 2,066 unresolved,
62,919 `agree`, 14.2% / 25.0% / 12.7% hindsight shares) are facts about the **old**
Soniox text and its alignment. They are **not** facts about the recomputed W-rt MSA.
The report computes and states W-rt's **own** class census and its **own**
alignment-conditional column oracle before interpreting any arm.

### The old-W vs W-rt difference is descriptive, not a result

The report states the baseline difference (WER, deletion, insertion, substitution
rates of W vs W-rt) **explicitly labelled as a model swap, not an experimental
comparison**. It confounds model, decoding path, audio pacing and non-determinism.
It is worth one paragraph because it prices how much of W depended on `stt-async-v5`.
No arm is ever compared to old W.

## 2. Acquisition protocol, frozen

Frozen in `scripts/run_soniox_rt_bench.py` before the batch ran:

| knob | value |
|---|---|
| model | `stt-rt-v4`, `wss://stt-rt.soniox.com/transcribe-websocket` |
| client | `soniox-tools/file_transcribe.py --json --realtime --lang el` (not a git repo; no revision to cite) |
| audio | ffmpeg-decoded to `pcm_s16le`, mono, **24 000 Hz**, from the 16 kHz window WAVs |
| chunking | 4 800 bytes (100 ms) per WS frame, one frame every **0.1 s** (~1x real time) |
| config | `endpoint_detection = False` (whole file), `language_hints = ["el"]` |
| silence trimming | **OFF** (`?trim=1` measured at ~11% of real words lost — this project's exact failure mode) |
| finalization | send `""` sentinel, drain until the `<end>` token; only `is_final` tokens are used |
| key | free Perplexity temp key, re-read from `~/.cache/soniox-dictate/temp_key.json` **per session** (one subprocess per window); Soniox validates only at the handshake |
| timeout | 600 s per session subprocess |
| retry | up to 3 attempts, linear back-off 5 s x attempt |
| retention | **the first protocol-valid success is kept, unconditionally.** No retry is ever triggered by a transcript looking short, poor or empty. A window with a cached token file is never re-transcribed. |
| concurrency | **12** for a 12-window pilot, **18** for the remaining 235 (`soniox-core`'s working ceiling). This ramp is a disclosed deviation: it was decided after the pilot returned 12/12 with zero errors, and the 12 pilot windows are kept under the first-success rule. Sustained concurrency and every failure are recorded in `run_log.json`. |

**Stop rule (Codex correction 6).** The primary analysis requires **247/247** windows
transcribed. If any window fails after the frozen retry protocol, the primary analysis
is **aborted**; a reduced-set analysis may be reported only as explicitly exploratory,
listing the failed item IDs with their city, duration, reference-token count and old-W
per-window WER, so that non-random failure is visible.

Raw tokens (verbatim council speech) live at
`~/.cache/oc-public/composition-rt-2026-08/soniox-tokens/<item_id>.json` and never
enter git.

## 3. Word confidence — inherited, not invented

Taken unchanged from the user's production client (`soniox-core`,
`internal/soniox/soniox.go:36,102`, in production since 2026-06-14) and from
`eval/soniox_confidence_probe.py:group_words`, which already implements it:

1. **finals only** (`is_final`);
2. explode each token to runes, each rune carrying its token's confidence;
3. a word is a whitespace-delimited run of runes;
4. **score = MIN confidence over LEXICAL runes only** (`isalpha or isdigit`),
   punctuation excluded so an uncertain comma cannot condemn its word;
5. the flag is strict `<` against a threshold.

The aggregate used by every arm is **`conf_min_lex`** — the production definition. This
differs from the probe's *preregistered* aggregate (`conf_min`, punctuation included),
and the choice is made here for a stated reason, not because it scored better: it is the
definition that has been in production for two months and the probe calibrated exactly
it at precision 0.706 / recall 0.164 / lift 4.86x at the production threshold 0.5.
`conf_min` and `conf_mean` are reported as sensitivity only.

### Confidence-to-column mapping, specified completely (Codex correction 11)

The Soniox hypothesis handed to the MSA is **derived from the cached finals**, not from
the client's `text` field, so the token stream and the confidence stream are the same
object by construction:

- `group_words(tokens)` gives an ordered list of words; `word_units(words)` maps each
  word onto the scorer's normalized token space (`wtoks`, which is
  `re.findall(r"\w+", norm(s))` — accent-stripping and case-folding happen inside
  `norm`, so Greek combining marks and apostrophes are handled by the frozen scorer, not
  by new code here).
- The Soniox hypothesis for window *w* is exactly `[u["tok"] for u in units]`, in order.
  Confidence is carried **by occurrence index**, never by token string, so repeated
  adjacent words are never confused.
- A word yielding several normalized tokens gives each of them the same word-level
  confidence. A word yielding none (pure punctuation, empty after normalization) is
  dropped from both streams together.
- Column → confidence: walk the MSA columns in order; each column whose Soniox entry is
  non-epsilon consumes the next Soniox occurrence. This is exact, and it is asserted at
  build time that the number consumed equals the length of the Soniox stream.
- **Invalid confidence makes the unit ineligible for every arm** while leaving the W-rt
  baseline text unchanged: missing, `None`, `NaN`, infinite, or outside `[0, 1]`.
  Counts of each are reported. A final token with text but no confidence is already
  dropped by `group_words`; that count is reported too.

## 4. The arms

All arms rewrite W-rt's token stream and are evaluated by
`fusion_lab.evaluate(idea, substrate_rt, fold="city")`, unmodified. Column classes and
the split/merge quarantine come from `column_classes.py`, unmodified.

`agree` columns are never touched by any arm.

### Arm O — occupancy, confidence-gated

Occupancy is confidence's clearest theoretical role: deciding whether there is a word
here at all.

**Eligible columns:** class `singleton` (`[x, ε, ε]`) where the lone present token is
**Soniox's**, the column is not in the split/merge quarantine, and the confidence is
valid. In W these columns always produce epsilon — the hierarchical vote drops any
column with fewer than two present tokens — so Soniox-only speech is discarded
wholesale.

**Rule:** emit the Soniox token iff `conf_min_lex >= tau_O`; otherwise keep epsilon.
No other column is altered.

**Fitting:** `tau_O` is fitted leave-one-**city**-out. On the nine training cities, for
each candidate threshold on the grid `{0.00, 0.05, ..., 1.00} ∪ {0.99, 0.999, 1.01}`
the arm's output is scored against the training references with the frozen scorer and
the threshold minimising **training pooled WER** is chosen; ties broken by the
**larger** threshold (fewer additions). `1.01` is on the grid so that "never fire" is
reachable; `0.00` so that "always fire" is reachable.

### Arm O2 — the same gate on `unresolved_two` (declared variant, not in the family)

**Eligible columns:** `unresolved_two` (`[x, y, ε]`) where Soniox is present with a
token, not split/merge, valid confidence. W-rt already emits a token here (occupancy is
settled 2:1; identity is tied and resolved by `tie_pivot` / `tie_priority`).

**Rule, stated unambiguously:** if `conf_min_lex >= tau_O2`, output **Soniox's token**;
otherwise output **W-rt's existing token, unchanged**. It fits its **own** threshold
`tau_O2` on its own eligible set, on the same grid and tie rule; it does **not** reuse
`tau_O`, and singleton and unresolved-two examples never share a training objective.

O2 is an identity arm wearing an occupancy arm's name, which is why it is a declared
variant and not a family member.

### Arm A — asymmetric confidence-weighted identity vote

**Eligible columns:** `unresolved_two` and `unresolved_three` — the tie set where W
falls back to `tie_pivot` / `tie_priority`. Soniox must be present with a token and a
valid confidence, and the column must not be split/merge quarantined — the same
quarantine every other arm here respects, so that the mutation-scope invariant of §8.4
holds uniformly. `exact_2_of_3` majorities are **protected** and out of scope for this
arm.

**Rule:** each present system casts a weighted vote for its token; the winner is the
argmax of summed weight; an exact tie keeps W-rt's existing choice, unchanged.

**This vote is asymmetric and that is its central weakness: only one of three systems
has a confidence signal.** Scribe and the adapter must be given a constant, and the
constant is preregistered:

> **k = 0.5** for `scribe-v2-clean` and for `oc-runpod-fixed-2026-08-10`; Soniox's
> weight is its `conf_min_lex` in [0, 1].

Reason, stated before running: k = 0.5 makes Soniox win an otherwise 1-against-1
contest exactly when its confidence exceeds **0.5**, the production operating point —
the only externally fixed, pre-existing threshold available to this project, chosen as a
UX judgement in June and calibrated for the first time by the probe at precision 0.706.
It is not fitted and it is not chosen by looking at WER. A fitted k would make the
asymmetry invisible rather than priced.

`k ∈ {0.3, 0.4, 0.6, 0.7}` is a **sensitivity envelope only**. If the sign of the effect
changes across that envelope, that is reported as instability and the arm does not ship
whatever k = 0.5 did.

### Arm M — minority override on 2-of-3 majorities

The old census put the largest single share of the W→oracle gap in `exact_2_of_3`
columns where the majority is jointly wrong. The autoresearch harness already measured
three text-only ways of overriding such majorities and all three came back **worse, CI
excluding zero on the wrong side, in all six search cities**. Confidence is a signal
those three did not have, which is the only reason this arm is written down.

**Eligible columns:** `exact_2_of_3` where **Soniox is the minority voice**, not
split/merge, valid confidence.

**Rule:** replace the majority token with Soniox's iff `conf_min_lex >= tau_M`.
**Fitting:** `tau_M`, leave-one-city-out, same grid and tie rule as Arm O.

### Controls that isolate confidence (Codex correction 2) — preregistered, mandatory

An arm that wins at a fitted `tau = 0` has shown that the *structural rewrite* helps,
not that *confidence* helps. Three controls are run through the identical pipeline:

- **O-all** — emit every eligible Soniox singleton, unconditionally (`tau = 0`).
- **M-all** — always take the eligible Soniox minority (`tau = 0`).
- **Confidence permutation** — 200 replicates in which `conf_min_lex` is permuted
  **within (meeting × eligibility class)**, preserving the marginal confidence
  distribution and the eligible-column structure, with the **complete fitting procedure
  rerun** inside each replicate. This gives a null distribution of each fitted arm's
  out-of-fold ΔWER.

**Attribution criterion, frozen:** an arm may be described as a *confidence* benefit
only if (a) it beats its ungated control (O vs O-all, M vs M-all; A has no ungated
control since its rule is degenerate without confidence — for A the permutation null is
the sole attribution test), **and** (b) its observed ΔWER is more extreme than the 5th
percentile of its own permutation null. If a fitted threshold comes back at the
"always fire" end of the grid, the report says explicitly that confidence supplied no
discrimination.

## 5. Gates

Every arm is judged against the **W-rt** baseline, never against old W. Definitions are
given in full because Codex correction 14 found them underspecified.

Let `S_i, D_i, I_i, N_i` be the frozen scorer's counts for window *i*.
`WER(set) = Σ(S+D+I) / ΣN` over that set. `ΔWER = WER(arm) − WER(W-rt)`, pooled, over
all 247 windows.

1. **Primary.** `ΔWER < 0` **and** the multiplicity-adjusted interval excludes zero.
2. **Deletion rate gate.** `ΣD(arm)/ΣN <= ΣD(W-rt)/ΣN`.
3. **Insertion rate gate.** `ΣI(arm)/ΣN <= ΣI(W-rt)/ΣN`.
   Gates 2 and 3 are **point-estimate operational constraints**, not evidence that the
   population rates are non-inferior. They are described that way in the report.
4. **Leave-one-out sign stability.** Over leave-one-window, leave-one-meeting and
   leave-one-city: recompute `ΔWER` on the remaining windows **reusing the fixed
   cross-fitted outputs** (no refitting). A sign flip is `sign(Δ_loo) != sign(Δ_full)`
   with `Δ == 0` counted as **not** a flip. Zero flips required.
5. **Single-item domination.** For meeting *m*, contribution
   `c_m = [Σ_{i∈m} (S+D+I)(W-rt) − Σ_{i∈m} (S+D+I)(arm)] / [Σ_all (S+D+I)(W-rt) − Σ_all (S+D+I)(arm)]`.
   If the denominator is `<= 0` the arm has no improvement to dominate and the check is
   reported as not applicable. Otherwise the arm is flagged **dominated** if
   `max_m c_m > 0.5`, and does not carry a headline on that number.

An arm "passes the gates" only if 1–4 pass and 5 does not fire. **Passing the gates is
not shipping** (§9).

### Multiplicity and the family

The **confirmatory family is exactly {O, A, M}** — three arms, no ordering, no
"secondary/tertiary" language. Bonferroni: the adjusted interval is the **central
two-sided 98.333% percentile interval, i.e. the 0.8333% and 99.1667% quantiles** of the
same 10,000 bootstrap replicates. Both the unadjusted 95% and the adjusted interval are
reported for each arm. O2, the k envelope, O-all, M-all and the permutation nulls are
**not** in the family and can never be promoted.

### Inference, and what the interval is conditional on (Codex corrections 3, 4)

- The paired percentile bootstrap is **clustered by meeting**, 10,000 replicates,
  seed 7, paired on identical references, **no refitting inside replicates**. It is
  therefore an interval **conditional on the ten fitted thresholds**, for future
  meetings *within these ten cities*. It is not an interval for a deployable trained
  arm, and it is not city-level generalization.
- Ten cities are too few for a percentile bootstrap over cities. City-level evidence is
  reported as (i) the full per-city ΔWER table, (ii) the city-unweighted mean ΔWER,
  (iii) an **exact paired sign test over the 10 cities** (two-sided, ties excluded),
  reported with its power stated as low.
- **The estimand claimed is the finite frozen benchmark plus, weakly, future meetings in
  these ten cities.** Nothing here claims future cities.

### Cross-fitting

Fold = **city**, 10 folds. `fit` sees only training-fold windows and may read their
reference text; `apply` never sees a reference. Arm A fits nothing, gets
`fitted: false`, and its out-of-fold number is identical to its in-fold one by
construction — stated, not hidden.

**What cross-fitting does not buy:** the class definitions, the eligibility rules and
this arm list were chosen by a human who has already seen five passes over these same
247 windows. Leave-one-city-out prices *parameter* overfitting only.

### Exposure reporting (Codex correction 16)

For every arm and every fold: eligible columns, columns with valid confidence, firings,
tokens added / replaced / removed, windows changed, per-fold fitted threshold, and —
**reporting only, never visible to `fit` or `apply`** — how many firings agreed with the
W-rt alignment-conditional column oracle and how many introduced a new disagreement.

## 6. What would be shipped, if anything were (Codex correction 5)

Cross-fitting produces up to ten different `tau` values per fitted arm; none of them is
a deployable rule. Preregistered now:

- The **candidate deployable threshold** is refitted **once on all ten cities** with the
  same grid and the same larger-threshold tie rule, and is **reported and frozen** in the
  report. It is *not* the number the gates are computed on, and it is *not* deployed.
- All ten fold thresholds are reported beside it, with eligible and firing counts per
  fold, so that a threshold chosen off one or two edits in a sparse fold is visible.

## 7. Predictions, recorded before the numbers

- **Arm O fails the insertion gate.** Weaker and more correct than revision 1's claim:
  under global Levenshtein alignment an added wrong token does **not** always increment
  the insertion count — it can convert a deletion or a substitution instead. The
  prediction is empirical, not constructional: the probe measured insertion detection as
  confidence's weakest arm (mean within-meeting AUROC 0.773), and the old census's
  hindsight replay of *every* occupancy column — perfect knowledge — already failed the
  insertion gate (deletions 0.0203 → 0.0124, insertions 0.0374 → 0.0391). A real model
  at 0.773 AUROC should do worse than hindsight.
- **Arm A produces a small effect whose CI includes zero.** The eligible set is a low
  single-digit percentage of columns, only a subset has Soniox present, and the
  char-vote arm on a neighbouring subset of the same tie set moved WER by −0.00008.
- **Arm M is worse than W-rt.** Overriding majorities has already failed three times on
  this substrate with different signals.
- **The W-rt baseline is worse than old W** (higher WER), because `stt-rt-v4` is the
  cheaper realtime model. The size of that gap is unknown and is the one number here
  nobody can guess.

If all three arms fail, that is the result. There is no fallback arm, and no arm is
added after seeing a number.

## 8. Implementation acceptance tests, required before any WER is read

Adopted from Codex's list, scoped to what is testable here. These run under
`eval/tests` / `pytest` and must be green before the arms are scored:

1. **No-op reproduction.** An identity idea through `fusion_lab.evaluate` on the W-rt
   substrate returns byte-identical output, `ΔWER == 0` exactly, equal S/D/I rates, and
   a degenerate bootstrap.
2. **Confidence-mapping fixtures.** Greek combining marks, apostrophes,
   punctuation-only words, digits, repeated adjacent words, a word producing two
   normalized tokens, non-final tokens ignored, absent timestamps, and invalid
   confidences (`None`, `NaN`, `inf`, `-0.1`, `1.2`) all handled as specified in §3.
3. **Golden arm fixtures.** Hand-built columns covering singleton / unresolved_two /
   unresolved_three / exact_2_of_3 / agree / quarantined / Soniox-absent /
   `conf == tau` / just below and above `tau` / exact weighted ties. The exact output
   token of O, O2, A and M is asserted.
4. **Mutation-scope invariant.** On every real window, every column at which an arm's
   output differs from W-rt's must be eligible for that arm; `agree` and quarantined
   columns never change.
5. **Cross-fitting leakage.** Each window is applied exactly once out of fold; no
   held-out reference reaches `fit`; the fitted threshold equals a brute-force grid
   recomputation; the larger-threshold tie rule is exercised.
6. **Determinism.** Re-running the scorer from the frozen token cache reproduces every
   metric byte-identically.

## 9. Confirmation obligation for any arm that passes (Codex corrections 1, 8)

Passing §5 on this run earns exactly two things, both preregistered here:

1. **A second, independently collected W-rt substrate** under the identical frozen
   acquisition protocol (a second nondeterministic draw of `stt-rt-v4`), with the
   already-frozen fold thresholds and rules applied **without redesign or retuning**.
   The direction of the effect and both rate gates must reproduce. If they do not, the
   result is classified **run-sensitive** and dies there. It is free and takes ~35
   minutes, so there is no excuse for skipping it.
2. **A named sealed evaluation that must precede any deployment**: cities and meetings
   that contributed nothing to any lexicon, roster, class definition or arm here. The
   6 sealed temporal-holdout windows of `eval-freeze-2026-08` are **not** spent on this
   — they remain sealed, and nothing in this experiment proposes opening them.

## 10. Honesty constraints that travel with every number

- **Confidence is conditional on emission.** 22.8% of edit operations in the probe's
  scored region were deletions the signal cannot see. A good AUROC on the gold set is
  not a claim about WER on the benchmark.
- The gold-set probe is 6 meetings in 6 cities with meeting and city fully confounded.
  It motivated this experiment; it licenses no claim here.
- `stt-rt-v4` confidence is not `stt-async-v5` confidence, and W-rt is not W. Nothing
  measured here transports to the shipped fusion stack without a paid re-run.
- Agreement-with-OpenCouncil, not fidelity-to-audio. Scored with
  `eval/controlled_eval/scoring.py`, not the benchmark app's scorer, so these numbers
  are comparable to `exp-2026-08-16-composition-over-selection` and **not** to the
  published leaderboard.
- **Sixth pass over the same 247 windows.** Freezing a design does not remove adaptive
  overfitting pressure across passes; that is why §9 exists.
- Realtime Soniox is not deterministic (97.8% word agreement run-to-run on the same
  model, measured by the probe). W-rt is one draw and is not exactly reproducible.

## 11. Cost

Zero. Free Perplexity temp key on the realtime path, local CPU, no GPU, no paid API
call. If any step would cost money the run stops and reports instead.

## 12. What Codex changed between revision 1 and revision 2

Job `b71f2dca0cad451db62cfb8f65e9d08e`, high effort, before any arm was implemented.
Adopted: the development-vs-confirmatory reclassification and the confirmation
obligation (§9); the ungated and permutation **controls** and the frozen attribution
criterion, without which a `tau = 0` win would have been mislabelled a confidence
result; the estimand/clustering statement and the city-level sign test; the explicit
statement that the bootstrap is conditional on the fitted thresholds; the all-or-nothing
transcription stop rule replacing the intersection rule; the fully frozen acquisition
table including the disclosed 12→18 concurrency ramp and the first-success retention
rule; the manifest and the post-removal totals; the ban on transferring old-W census
facts to W-rt; the complete confidence-to-column mapping and the invalid-confidence
rule; an unambiguous O2; the correction of the false "an added wrong token is an
insertion by construction" argument; mathematical definitions of every gate; the exact
0.8333% / 99.1667% adjusted quantiles and the single "family" vocabulary; the exposure
and oracle-agreement reporting; the deployable-threshold definition; and the acceptance
test list.

Not adopted, with reasons: **nested resampling that refits inside every bootstrap
replicate** (correction 4's stronger form) — the honest fix for threshold-fitting
uncertainty here is §9's independent second draw plus the permutation null, and nested
refitting inside 10,000 replicates would cost more compute than the experiment is
worth while still being conditional on the same 247 windows; the interval is instead
**labelled** conditional everywhere it appears. **Aborting on any single window
failure without reporting anything** — the abort rule is adopted for the *primary*
analysis, but the failed-window characterisation is reported rather than discarded,
which is what makes non-random failure visible.
