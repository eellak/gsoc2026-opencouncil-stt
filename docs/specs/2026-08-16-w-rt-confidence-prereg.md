# W-rt: does Soniox per-word confidence beat the plain per-column vote?

Preregistration. 2026-08-16. `exp-2026-08-16-w-rt-confidence`.
Written **before** any Soniox re-run token was scored and before any WER number of any
arm existed. Zero GPU, zero paid API.

Predecessors: [`exp-2026-08-16-soniox-confidence`](../reports/2026-08-16-soniox-confidence-probe.md)
(the signal exists on the gold set),
[`exp-2026-08-16-composition-over-selection`](../reports/2026-08-16-char-vote-homophones.md)
(W, the per-column vote), [`exp-2026-08-16-char-vote-homophones`](../reports/2026-08-16-char-vote-homophones.md)
(the column census and the hindsight replays this spec's arm list is drawn from).

## 1. Why this is a new substrate and not a new arm on the old one

The cached Soniox text for the 247 benchmark windows was produced by the paid
`stt-async-v5` and carries **no confidence values** — the client discarded them. The
only free path is the realtime model `stt-rt-v4`. Different model, therefore different
text. Soniox is one of W's three voters, so re-running it **changes W itself**.

Attaching new confidences to the old cached text is forbidden here: the probe measured
only 14 of 27 gold-set cells reproducing token-identically between the two runs of the
*same* model, so a token-by-token graft across two *different* models has no defensible
alignment.

So this experiment builds a **parallel substrate, W-rt**, and every number it reports is
measured inside it:

- Same 247 windows, same 144 meetings, same 10 cities, same references, same removal of
  the 6 sealed temporal-holdout windows of `eval-freeze-2026-08` (which stay sealed).
- `scribe-v2-clean` and `oc-runpod-fixed-2026-08-10` byte-identical from the cached
  benchmark report `2026-08-10-corrected-adapter-label-prefix-fix-vs-ju`.
- The `soniox` arm replaced by a fresh `stt-rt-v4` transcription of the same window
  audio (`$SC/bench_windows/<item_id>.wav`), carrying per-token confidence.
- The whole-window consensus pivot recomputed from the new trio (`consensus_pick`).
- The alignment recomputed (`msa.align3`, same band rule, same frozen transition order)
  and **W-rt** composed by the same unmodified hierarchical vote (`msa.compose`).

**Nothing existing is modified.** New cache paths only
(`~/.cache/oc-public/composition-rt-2026-08/`), new result files, no edit to
`fusion_lab.py`, `msa.py`, `scoring.py`, or any existing cache. The old W and every
frozen number stay exactly as they are.

### The old-W vs W-rt difference is descriptive, not a result

The report will state the baseline difference (WER, deletion, insertion, substitution
rates of W vs W-rt) **explicitly labelled as a model swap, not an experimental
comparison**. It confounds model, decoding path, audio pacing and non-determinism, and
it is not paired in any meaningful causal sense. It is worth one paragraph because it
prices how much of W depended on `stt-async-v5`. No arm is ever compared to old W.

## 2. Data collection, frozen before it ran

- Model `stt-rt-v4`, realtime WebSocket, free Perplexity temp key, `--lang el`,
  `endpoint_detection=False` (whole file), audio fed at ~1x (`--realtime`).
- **Silence trimming stays OFF.** Measured at ~11% of real words disappearing; that is
  this project's exact failure mode.
- N parallel sessions, one session per 140 s window (no segmentation or merging is
  needed — the windows are already short). The concurrency actually sustained, and any
  back-off, is recorded in the report. The `maxJobs = 18` figure inherited from the
  user's `soniox-core` is a working ceiling with no surviving measurement artefact.
- The key is re-read from `~/.cache/soniox-dictate/temp_key.json` per session (one
  subprocess per window), never cached in memory across the batch. Soniox validates the
  key only at the WS handshake.
- Resumable per window: a window with a cached token file is never re-transcribed.
- Raw tokens cached under
  `~/.cache/oc-public/composition-rt-2026-08/soniox-tokens/<item_id>.json`, never in
  git. Retry policy: up to 3 attempts per window, then the window is recorded as failed.

**Stop rule.** If fewer than 247 of 247 windows transcribe successfully, the substrate
is built on the **intersection** of successful windows and every number — including the
W-rt baseline and the descriptive old-W comparison — is computed on that same reduced
set, with the count stated. Arms are never scored on a different subset from their own
baseline.

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

The aggregate used by every arm here is **`conf_min_lex`** — the production definition.
This differs from the probe's *preregistered* aggregate (`conf_min`, punctuation
included) and the choice is made here for a stated reason, not because it scored
better: it is the definition that has been in production for two months, and the probe
calibrated exactly it at precision 0.706 / recall 0.164 / lift 4.86x at the production
threshold 0.5. `conf_min` and `conf_mean` are reported as sensitivity only.

`start_ms` / `end_ms` are `omitempty` on the wire; words without a timestamp are counted
and excluded from timing use, but their confidence is still used (no arm here needs a
timestamp). Confidence is mapped onto the scorer's normalized token space by
`word_units`: a word yielding several normalized tokens gives each of them the same
word-level confidence; a word yielding none (pure punctuation) is dropped. Both counts
are reported.

## 4. The arms

All arms rewrite W-rt's token stream and are evaluated by
`fusion_lab.evaluate(idea, substrate_rt, fold="city")`, unmodified. Column classes and
the split/merge quarantine come from `column_classes.py`, unmodified. Ordered by
expected value.

The frozen partition from the census is respected: `agree` columns are never touched
(the oracle disagrees with W there zero times in 62,919 columns).

### Arm O — occupancy, confidence-gated (primary)

Occupancy is confidence's clearest theoretical role: deciding whether there is a word
here at all. The census puts 14.2% of the remaining W→oracle gap in occupancy columns.

**Eligible columns:** class `singleton` (`[x, ε, ε]`), where the lone present token is
**Soniox's**, and the column is not in the split/merge quarantine. In W these columns
always produce epsilon — the hierarchical vote drops any column with fewer than two
present tokens — so Soniox-only speech is currently discarded wholesale.

**Rule:** emit the Soniox token iff `conf_min_lex >= tau_O`. Otherwise keep W-rt's
epsilon. No other column is altered.

**Fitting:** `tau_O` is fitted leave-one-**city**-out. On the nine training cities, for
each candidate threshold on the grid `{0.00, 0.05, ..., 1.00}` plus `{0.999}`, the
arm's output is scored against the training references with the frozen scorer, and the
threshold minimising training WER is chosen; ties broken by the **larger** threshold
(the conservative direction, fewer insertions). Only held-out city outputs are scored.

**Declared secondary variant O2:** the same rule extended to `unresolved_two` columns
(`[x, y, ε]`) where Soniox is one of the two present systems — identity is contested but
occupancy is already 2:1 for presence, so this is not an occupancy decision and O2 is
reported as an exploratory variant, not as a second primary.

### Arm A — asymmetric confidence-weighted identity vote (secondary)

**Eligible columns:** `unresolved_two` and `unresolved_three` — the 2,066-column tie set
(2.56% of columns) where W currently falls back to `tie_pivot` / `tie_priority`. Soniox
must be present with a token. `exact_2_of_3` majorities are **protected** and out of
scope for this arm, per the frozen census partition.

**Rule:** each present system casts a weighted vote for its token; the winner is the
argmax of summed weight; an exact tie keeps W-rt's existing choice, unchanged.

**This vote is asymmetric and that is its central weakness: only one of three systems
has a confidence signal.** Scribe and the adapter must be given a constant, and the
constant is preregistered here:

> **k = 0.5** for `scribe-v2-clean` and for `oc-runpod-fixed-2026-08-10`; Soniox's
> weight is its `conf_min_lex` in [0, 1].

Reason, stated before running: k = 0.5 makes Soniox win an otherwise 1-against-1
contest exactly when its confidence exceeds **0.5**, the production operating point —
the only externally fixed, pre-existing threshold available to this project, chosen as a
UX judgement in June and calibrated for the first time by the probe at precision 0.706.
It is not fitted, and it is not chosen by looking at WER. Arms with `k` fitted on the
training folds are **not** run: with a single free constant and one signal, a fitted k
would make the asymmetry invisible rather than priced.

`k ∈ {0.3, 0.4, 0.6, 0.7}` is reported as a **sensitivity envelope only**. If the sign
of the effect changes across that envelope, that is reported as instability and the arm
does not ship whatever k = 0.5 did.

### Arm M — minority override on 2-of-3 majorities (tertiary, expected to fail)

The census puts 25.0% of the W→oracle gap — the largest single class — in
`exact_2_of_3` columns where the majority is jointly wrong. The autoresearch harness
already measured three text-only ways of overriding such majorities and all three came
back **worse, CI excluding zero on the wrong side, in all six search cities**.
Confidence is a signal those three did not have, which is the only reason this arm is
written down at all.

**Eligible columns:** `exact_2_of_3` where **Soniox is the minority voice** and the
column is not split/merge.

**Rule:** replace the majority token with Soniox's iff `conf_min_lex >= tau_M`.

**Fitting:** `tau_M` leave-one-city-out on the same grid and the same tie rule as Arm O.

This arm is tertiary. It is preregistered so that a null result is recorded rather than
quietly dropped, and so that a positive one cannot be claimed as a discovery made after
looking.

## 5. Gates — the project's existing frozen gates, unchanged

Every arm is judged by `fusion_lab.evaluate` against the **W-rt** baseline, never
against old W:

1. **Primary.** ΔWER vs W-rt is negative **and** its paired bootstrap CI excludes zero.
   Bootstrap: percentile, **clustered by meeting**, **10,000 replicates**, seed 7,
   paired on identical references, no refitting inside replicates.
2. **Deletion rate gate.** `del_rate(arm) <= del_rate(W-rt)`.
3. **Insertion rate gate.** `ins_rate(arm) <= ins_rate(W-rt)`.
4. **Leave-one-out sign stability.** Zero sign flips of ΔWER over leave-one-window,
   leave-one-meeting and leave-one-city.
5. **Single-item domination check.** The per-city delta table and the city-unweighted
   mean delta are reported beside the meeting-clustered CI, and any arm whose headline
   effect is more than 50% supplied by one meeting is reported as dominated and does not
   ship on that number.

An arm ships only if **all** of 1–4 pass and 5 does not fire.

### Multiplicity

Three primary arms (O, A, M). Family-wise control is **Bonferroni over the three**: the
adjusted gate is a `100 × (1 − 0.05/3) = 98.33%` percentile CI from the same bootstrap
replicates, and both the unadjusted 95% and the adjusted 98.33% interval are reported
for each arm. Declared variants (O2, and the k envelope of A) are **not** in the family
and are never promoted; they are sensitivity reporting.

### Cross-fitting

Fold = **city**, 10 folds. An idea's `fit` sees only training-fold windows and may read
their reference text; `apply` never sees a reference. Arms with no fitted parameter
(Arm A) get `fitted: false` and their out-of-fold number is identical to their in-fold
one by construction — this is stated, not hidden.

**What cross-fitting does not buy**, restated from `fusion_lab`'s own docstring: the
class definitions, the eligibility rules and this arm list were chosen by a human who
has already seen five passes over these same 247 windows. Leave-one-city-out prices
*parameter* overfitting only.

## 6. Predictions, recorded before the numbers

- **Arm O fails the insertion gate.** This is the strong prediction. Any added token
  that is wrong is a new insertion by construction, so passing gate 3 needs
  near-perfect precision on the added words. The probe measured **insertion detection
  as confidence's weakest arm (mean within-meeting AUROC 0.773)**, and the census's
  hindsight replay of *every* occupancy column — perfect knowledge — already failed the
  insertion gate (deletions 0.0203 → 0.0124, insertions 0.0374 → 0.0391). A real model
  with a 0.773 AUROC should do worse than hindsight, not better. The likely fitted
  `tau_O` is therefore high enough that the arm barely fires.
- **Arm A produces a small effect whose CI includes zero.** The eligible set is ~2,066
  columns of 80,659, only a subset has Soniox present, and the char-vote arm on a
  neighbouring subset of the same tie set moved WER by −0.00008.
- **Arm M is worse than W-rt.** Overriding majorities has already failed three times on
  this substrate with different signals.
- **The W-rt baseline is worse than old W** (higher WER), because `stt-rt-v4` is the
  cheaper realtime model. The size of that gap is unknown and is the one number here
  nobody can guess.

If all three arms fail, that is the result and it is recorded as such. There is no
fallback arm, and no arm will be added after seeing a number.

## 7. Honesty constraints that travel with every number

- **Confidence is conditional on emission.** 22.8% of edit operations in the probe's
  scored region were deletions the signal cannot see. A good AUROC on the gold set is
  not a claim about WER on the benchmark.
- The gold-set probe is 6 meetings in 6 cities with meeting and city fully confounded.
  It motivated this experiment; it does not license any claim here.
- `stt-rt-v4` confidence is not `stt-async-v5` confidence, and W-rt is not W. Nothing
  measured here transports to the shipped fusion stack without a paid re-run.
- Agreement-with-OpenCouncil, not fidelity-to-audio. Scored with
  `eval/controlled_eval/scoring.py`, not the benchmark app's scorer, so these numbers
  are comparable to `exp-2026-08-16-composition-over-selection` and **not** to the
  published leaderboard.
- **Sixth pass over the same 247 windows.** Freezing a design does not remove adaptive
  overfitting pressure across passes.
- Realtime Soniox is not deterministic (the probe measured 97.8% word agreement
  run-to-run on the same model). W-rt is one draw from that distribution and is not
  exactly reproducible.

## 8. Cost

Zero. Free Perplexity temp key on the realtime path, local CPU, no GPU, no paid API
call. If any step would cost money the run stops and reports instead.
