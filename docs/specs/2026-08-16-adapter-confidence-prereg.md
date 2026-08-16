# Preregistration — per-word confidence from `artifact-adapter-fixed`

`exp-2026-08-16-adapter-confidence`. Written **before any outcome statistic was
computed**, and before the scorer existed. Reviewed by Codex at high effort
(job `06a6ad95a3d14af4bd53ba76308a6b66`) before the scoring code was written; every
numbered item below that says *(Codex)* exists because that review asked for it.

Companion to [`2026-08-16-soniox-confidence-probe`](../reports/2026-08-16-soniox-confidence-probe.md).
Definitions are **inherited from it wherever one exists**, so that the two systems are
measured the same way. Inherited items are marked *(inherited)* and their being
defensible rests on their having been fixed for a different system, before this one.

## Question

faster-whisper returns a real per-word probability when `word_timestamps=True`. Our
serving code already exposes it (`serve/oc-asr/oc_asr_server.py:172`). The 247-window
benchmark decode did not ask for it. Three things follow, in order:

1. **Gate.** Does asking for it change the transcript? If it does, confidences belong
   to a re-run and cannot be attached retroactively to the frozen fusion input W.
2. **Calibration.** Does the probability predict which emitted words are wrong?
3. **Head-to-head.** Is it the *same* signal as Soniox confidence, or a complementary
   one? With no Scribe credential there are 2 of 3 voters, and whether those two
   signals are independent bounds what a confidence-weighted vote can gain.

## Substrates — kept strictly separate

| | gold set | 247 windows |
|---|---|---|
| metric | **fidelity-to-audio** (a human listened) | **agreement-with-OpenCouncil** (our own published text) |
| n | 27 scored 15 s cores, 6 meetings, 6 cities | 247 two-minute windows, 144 meetings, 10 cities |
| adapter decode | already cached **with** word probabilities, `beam_size=2`, CPU int8, local server | requires the re-decode below |
| Soniox confidences | exist (probe re-run) | **do not exist** and will not be created |
| decides | the primary claim | descriptive only |

These are different quantities and are never merged (CLAUDE.md). The head-to-head is
possible **only** on the gold set.

City and meeting are fully confounded on the gold set — one meeting per city.

## The re-decode (247 windows)

Frozen config = `notebooks/decode_ablation.CONTROL`, which is the run's own
`decode.json`. **Exactly one field changes**: `word_timestamps`. Two passes, both
local, CPU int8, `MODEL_DIR=/home/harold/oc-asr-serve/ct2-fixed`
(`artifact-ct2-fixed`), per-window CTranslate2 seed `decode_ablation.seed_for("A", wid)`
— the control arm's own seed, i.e. common random numbers, so any difference is
attributable to `word_timestamps` and not to a different sampling draw.

Codex judged 37 windows insufficient to carry a benchmark-wide magnitude, so **both
passes cover all 247 windows** rather than reusing the 37-window overlap alone.

### Three paired contrasts — *not* additive components (Codex)

WER is nonlinear; the two contrasts below do not sum to the third. They are named as
contrasts, not as a decomposition.

| contrast | what it measures |
|---|---|
| cached GPU `wt=F` vs local CPU `wt=F` | **stack contrast** (RunPod GPU fp16 vs local CPU int8) |
| local CPU `wt=F` vs local CPU `wt=T` | **the gate**, `word_timestamps` isolated |
| cached GPU `wt=F` vs local CPU `wt=T` | end-to-end |

"Stack contrast" is only a fair name if model artifact, library versions and every
decode argument are otherwise identical; the report must state what was verified and
what was not.

Reported per contrast: raw-identical rate, normalized-token-identical rate, **pooled
WER in both directions with the denominator named**, a symmetric normalized edit
distance, and the per-window distribution.

### Amendment, 2026-08-16 20:05 EEST — randomized order and a wall-clock stop

Made **before any contrast statistic was computed or looked at**. The only 247-window
outputs inspected up to this point were decoded-window *counts* and the scorer's own
"PARTIAL coverage" guard.

Measured after launch: this box is shared with two other agents, and the paired pass
runs at RTF ≈ 1.38 (`wt=T`) and ≈ 0.89 (`wt=F`), i.e. **≈ 14 h of wall clock** for the
full paired design. That is not a defensible cost for a probe whose primary claim rests
on the gold set. So:

- the **remaining** decode queue is randomized with a fixed seed (`random.Random(21)`),
  applied to the **full 247-item substrate** and then filtered, so **both passes walk
  the identical order** and the slower pass's completed set is nested inside the faster
  one's;
- decoding stops at a wall-clock deadline fixed here in advance:
  **2026-08-16 23:45 EEST**;
- the analysis set is the windows completed in **both** passes.

Consequence, disclosed rather than hidden: the first 9 windows of `wt=T` and 11 of
`wt=F` were decoded in substrate order before the amendment, and substrate order is
city-clustered (all of them are `argos`). The analysis set is therefore **that
city-clustered head plus a uniform random sample of the remainder**, not a uniform
sample of all 247. The report states the realized n and the city composition. The
sample size is outcome-independent because the deadline was fixed before any contrast
was read.

Codex's objection to the 37-window eval-freeze overlap was precision, not principle;
this amendment answers it with a larger and *randomized* sample rather than with the
pre-existing, purposively-selected 37.

### Gate criterion (frozen now)

Confidences may be attached to the frozen W substrate **only if the normalized token
sequence is identical in every window of the analysis set** (all 247 before the
amendment above; the completed-in-both set after it). A single mismatch fails it —
the criterion is a conjunction, so a smaller sample can fail the gate but cannot pass
it as convincingly. Diagnostics reported
alongside: raw text identity, segment-boundary identity (the segment proxy
additionally needs a compatible segment mapping). **No analysis may afterwards be
restricted to the subset of windows that happened to be stable** — that would be
selection on the outcome.

### Disposition of the two smoke windows (Codex)

A 2-window smoke was run **before** this document, to size the job. It showed that
local CPU `wt=F` reproduces the cached control arm bit-exactly, and that `wt=T`
changed the text on both windows (segment count 17 → 19 on one). These are recorded as
**development observations**: the gate *criterion* is frozen here outcome-blind, but
the gate *outcome* was partially visible beforehand and the report must say so rather
than claim otherwise.

## Analysis unit (Codex)

Not "an emitted word". The unit is a **normalized hypothesis-token instance**:

- an emitted word is mapped through the frozen normalizer `wtoks`;
- a word yielding **zero** tokens (pure punctuation) is excluded, and counted;
- a word yielding **more than one** token contributes one row per token, each carrying
  the *same* probability — duplicated probabilities stay duplicated, which gives
  multi-token words extra weight. This is the inherited Soniox behaviour and is kept
  for comparability, not because it is ideal;
- `M` is the negative class, `S` and `I` are positive, `D` produces **no row**.

Counts reported: emitted words, normalized tokens, and how many words produced 0 / 1 /
>1 tokens.

## Frozen statistics

- **Score**: faster-whisper's word `probability`, `p_word`. There is no min/mean choice
  here — unlike Soniox subtokens, faster-whisper emits one probability per word.
- **PRIMARY**: equal-weight mean of the **within-meeting** AUROCs of `1 - p_word` for
  predicting that a normalized hypothesis token is an error; ties worth 0.5; null
  **0.5** *(inherited)*.
- **Informative threshold 0.60** *(inherited)* — the Soniox probe's GO threshold,
  restated for comparability. **It is not a promotion gate here.**
- A meeting whose rows are all positive or all negative has an **undefined** AUROC: it
  is excluded from the macro mean and counted (Codex).
- **Region**: `core_envelope` primary; `core_strict` and `clip` as sensitivity
  *(inherited from the gold-set prereg)*.
- **Alignment**: `eval.gold_set_score.align_ops` (tie-break S > D > I) primary; all 6
  op priorities × forward/reversed reported as a **complete 12-way envelope**, never
  selected among (Codex). The head-to-head uses the **primary alignment only**.
- **Permutation null**: permute `p_word` at the **originating emitted-word block**
  within meeting — not at the token row, because normalizer expansions share one
  probability and independent row permutation would violate exchangeability (Codex).
  2,000 permutations, seed 21, one-sided (`AUROC > 0.5`), p = (b+1)/(B+1). For the
  **segment proxy** the permutation unit is the whole segment block.
- **Meeting-cluster bootstrap**: 2,000 resamples of the 6 meetings, seed 21. Six
  clusters: **descriptive, not significance**. Replicates with no informative meeting
  are dropped and counted.
- **Domination diagnostic**: the six leave-one-meeting-out macro AUROCs, their range,
  and whether every one stays above 0.5 and above 0.60 (Codex); plus each meeting's
  share of the errors.
- **Calibration**: equal-width bins 0.0–1.0 step 0.1, and sample deciles. Ties are
  **never split to force equal bin sizes**; fewer than ten effective bins is allowed
  and realized counts are always shown (Codex).
- **Deletion coverage**, stated before the result: `(S + I) / (S + I + D)`, computed
  **for the adapter itself** — Soniox's 22.8% is not transferable (Codex). This is a
  bound on edit-operation *coverage*, not a ceiling on AUROC.
- **Splits** *(inherited)*: overlap vs non-overlap; insertions-vs-match;
  substitutions-vs-match. The insertion arm is inherited from the Soniox probe, where
  it was already known to be the weakest — it is not a new hypothesis found here.

## Segment proxy

`exp(avg_logprob)`, the number `oc_asr_server.py:175` already returns, attached to
every word of its segment and run through the identical pipeline as a **secondary**
score. It is a **ranking** score and no calibrated word probability is claimed for it.
"Usable proxy" carries **no threshold**: the finding is reported as the paired
per-meeting AUROC differences against `p_word`.

## Head-to-head with Soniox (gold set only)

**Shared column**: a reference token index at which **both** systems have an emitted
word aligned to it (`M` or `S` under each system's own primary alignment to the same
gold reference sequence). Insertions have no reference index and are excluded; so is
every deletion by either system. The head-to-head is therefore a **conditioned
subset** and is not a comparison of overall confidence quality — this limitation is
printed next to the table.

Frozen:

- **Ranks** are computed among shared eligible columns **within each meeting**,
  midranks for ties, scaled to [0, 1]; an all-tied or single-observation group is
  rank 0.5.
- **Combination confidence** = `(rank_adapter + rank_soniox) / 2`; uncertainty is
  `1 −` that. Mean-rank rather than min-rank: mean-rank encodes compensatory
  consensus, min-rank encodes an "either system warns" OR rule, and they answer
  different operational questions (Codex). No weight is fitted. No logistic model is
  fitted — that would be tuning on the gold set.
- **The label each combination AUROC predicts is frozen separately** (Codex), because
  the two systems have different error labels:
  1. combination vs adapter uncertainty, predicting **adapter** error;
  2. combination vs Soniox uncertainty, predicting **Soniox** error.
  No "either system is wrong" label is introduced after seeing results.
- **Error co-occurrence** 2×2: neither / adapter only / Soniox only / both, with
  counts.
- **Spearman** with midranks, per meeting, pooled reported as **secondary** (a pooled
  correlation can be driven by between-meeting differences). Tie diagnostics — share
  of tied scores and number of unique confidence values — reported for both systems.
- **Bottom-decile flag**, by rank within meeting: all observations tied at the cutoff
  are included and the **realized** flag rate is reported; ties are not broken to force
  exactly 10% (Codex).

## 247-window descriptive arm

Same pipeline; the label is disagreement with our own published corrected text, not
fidelity to audio. Clustered by meeting (144). No overlap split — there is no
diarization there. If the gate fails, these confidences belong to **this re-run** and
cannot be attached to the frozen fusion input W, and the report says so.

## Honesty constraints

- 27 cells / 6 meetings cannot support a population claim or any promotion gate. A
  6-cluster interval excluding a null is not significance.
- The gold reference is one listener, one pass, no adjudication: conclusions are
  phrased as fidelity to the human-verified reference **under its transcription
  rules**, not as unconstrained audio truth (Codex).
- Nothing is tuned on the gold set.
- If the signal is uninformative, that is the result. No variant hunting.

## Implementation tests required before any number is believed (Codex)

1. **Alignment/accounting fixture** — synthetic M/S/I/D plus a punctuation-only word
   and a word splitting into two tokens: every normalized token gets exactly one
   M/S/I label, deletions produce no row, duplicated tokens inherit the intended
   probability, each shared reference index is unique per system.
2. **Known-AUROC fixture** — perfect ordering → 1.0, reversed → 0.0, all-equal → 0.5,
   through the macro path, the percentile-rank combination and the tie handling.
3. **Reconciliation assertion** — the token sequence rebuilt from emitted words equals
   the sequence handed to `align_ops`; `0 ≤ p_word ≤ 1`; no eligible word missing a
   probability; the segment proxy constant within each segment.
