# Does our own adapter's per-word confidence predict its errors — and is it the same signal as Soniox's?

2026-08-16, completed 2026-08-17. `exp-2026-08-16-adapter-confidence`. Zero GPU, zero
paid API, local CPU only.

**Answer: yes it predicts, about as strongly as Soniox's does — and the two signals are
not the same signal. But our confidence can reach a much smaller share of our damage,
because 41% of our edit operations are deletions, which a per-word confidence cannot
see by construction. And the gate fails: turning word probabilities on changes the
transcript in 101 of 102 paired windows, so these confidences belong to their own decode
and cannot be retrofitted to the frozen fusion input W.**

Preregistered in
[`docs/specs/2026-08-16-adapter-confidence-prereg.md`](../specs/2026-08-16-adapter-confidence-prereg.md),
written before the scorer existed and before any outcome statistic was computed.
Definitions are inherited from
[the Soniox probe](2026-08-16-soniox-confidence-probe.md) wherever one existed, so the
two systems are measured the same way.

## What was measured, and on what

| | gold set | 247 windows |
|---|---|---|
| metric | **fidelity-to-audio** (a human listened) | **agreement-with-OpenCouncil** |
| n | 27 scored 15 s cores, 6 meetings, 6 cities | **102** of 247 two-minute windows, 82 meetings — the decode was stopped |
| Soniox confidences exist? | yes | **no** |
| carries | the primary claim and the whole head-to-head | descriptive only |

These are different quantities and are never merged (CLAUDE.md). **The head-to-head —
the reason this ticket exists — is a 27-cell measurement**, because Soniox per-word
confidence exists only on the gold set: the 247 benchmark Soniox output was produced by
the paid `stt-async-v5` with no confidences, and re-running it would change the text
under every frozen number in this project.

## The lucky half: the gold set needed no re-decode

The local serving path (`serve/oc-asr/oc_asr_server.py`) has **always** called
faster-whisper with `word_timestamps=True`. The gold-set adapter hypotheses came
through that path, so **the per-word probabilities were already cached, attached to
exactly the text `exp-2026-08-16-gold-set` scored**. No re-run, no gate risk, nothing
to reconcile. That is why the primary claim below rests on the gold set and not on the
247 windows.

The price is that this decode is `beam_size=2`, CPU int8 — **not** the 247-window
benchmark config (beam 5, RunPod GPU). Nothing here transports to the benchmark decode
without the same kind of test the 247 arm below is.

Two cross-checks that the machinery is measuring what it claims:

- `(S+D+I)/(M+S+D)` = **0.282** against the gold-set report's 0.284. Consistent.
- Running the *Soniox* system through this scorer's own unit builder reproduces the
  published probe exactly — M = 859, S = 97, D = 43, I = 49, n = 1005, 146 errors, mean
  within-meeting AUROC **0.8167**. The two systems really are going through one
  pipeline, so the comparison below is not two pipelines being compared.

## The ceiling, stated before the result

In `core_envelope`: **M = 749, S = 134, D = 116, I = 32.**

**Deletions are 41.1% of edit operations.** Edit-operation coverage —
`(S+I)/(S+I+D)`, the share a per-word confidence can even attach to — is **58.9%**.

Soniox on the same recordings was 22.8% / 77.2%. These are *different sets of errors*,
each system's own; the contrast is structural, not a performance comparison. It says
our model's failure mode is much more deletion-heavy, and that is exactly the part
confidence is blind to. This is the standing finding of
`exp-2026-08-12-serving-stack-ladder` — the deletions live in the weights — arriving
again from a new direction.

## Result: the word probability discriminates errors

n = 915 normalized token rows, 166 errors, prevalence 0.1814.

| | mean within-meeting AUROC | pooled AUROC | average precision (null 0.181) |
|---|---|---|---|
| **`p_word`** (preregistered) | **0.8151** | 0.8778 | 0.607 |
| `exp(avg_logprob)` segment proxy | 0.6110 | 0.7861 | 0.519 |

Per meeting (`p_word`):

| meeting | AUROC | n | errors |
|---|---|---|---|
| sparta jul1 | 0.892 | 118 | 10 |
| chalandri jul29 | 0.861 | 130 | 12 |
| samothraki jul6 | 0.790 | 199 | 101 |
| zografou jun18 | 0.707 | 31 | 2 |
| chania jun24 | 0.854 | 277 | 24 |
| xylokastro jun29 | 0.787 | 160 | 17 |

No meeting had an undefined AUROC. Leave-one-meeting-out stays in **0.7997–0.8367**,
every value above 0.60. Meeting-cluster 95% [0.764, 0.862] — 6 clusters, descriptive,
not significance.

**Within-meeting permutation null**, blocking on the emitted word (normalizer
expansions share one probability, so permuting token rows independently would violate
exchangeability): null mean 0.4999, 95% [0.414, 0.587], observed 0.8151, one-sided
p = 0.0005. The observed value sits outside the entire null band.

**Single-meeting domination**: samothraki jul6 supplies **101 of 166 errors (61%)** —
more concentrated than the Soniox probe's 47%. Its own AUROC is 0.790, at the low end,
and removing it *raises* the mean to 0.820. So jul6 is not inflating the primary
statistic. It does dominate everything prevalence-dependent — the pooled AUROC, the
average precision, the calibration table, the bottom-decile yield and any statement
about the error mix — and those must be read with it in mind.

Region sensitivity: `core_strict` 0.845 (n = 421, deletion share 50.3%), `clip` 0.852
(n = 1011). Alignment envelope, all 12 variants (6 op priorities × forward/reversed):
macro 0.8136–0.8152, S ∈ [124, 134], I ∈ [32, 37]. The alignment ambiguity is
negligible.

### Calibration — discrimination, not calibrated probabilities

| bucket | n | errors | error rate | 95% Wilson |
|---|---|---|---|---|
| [0.0, 0.1) | 3 | 3 | 1.000 | [0.44, 1.00] |
| [0.1, 0.2) | 9 | 7 | 0.778 | [0.45, 0.93] |
| [0.2, 0.3) | 13 | 8 | 0.615 | [0.36, 0.82] |
| [0.3, 0.4) | 10 | 4 | 0.400 | [0.17, 0.69] |
| [0.4, 0.5) | 16 | 11 | 0.688 | [0.44, 0.86] |
| [0.5, 0.6) | 23 | 19 | 0.826 | [0.63, 0.93] |
| [0.6, 0.7) | 34 | 18 | 0.529 | [0.37, 0.69] |
| [0.7, 0.8) | 62 | 29 | 0.468 | [0.35, 0.59] |
| [0.8, 0.9) | 77 | 31 | 0.403 | [0.30, 0.52] |
| [0.9, 1.0) | 668 | 36 | 0.054 | [0.04, 0.07] |

Error enrichment generally increases toward lower confidence, but **the sparse middle
bins are locally non-monotone**: [0.5, 0.6) at 0.826 sits above [0.4, 0.5) at 0.688 and
[0.3, 0.4) at 0.400. The 0.826 − 0.688 gap is 0.139 with a naive SE of 0.140 — readily
sampling noise. Nothing is claimed from the shape.

**No calibrated error probability is claimed.** Several buckets show error rates far
above their nominal `1 − p`, 668 of 915 rows sit in one wide bucket, and the sample
deciles run 0.707 / 0.484 / 0.289 / 0.097 / 0.110 / 0.056 / 0.022 / 0.046 / 0.010 /
0.000. This is a **ranking** signal.

Bottom decile (realized 9.9%, n = 91): error rate **0.703** against prevalence 0.181 —
**3.88× lift**.

### Splits

| | n | prevalence | mean WM AUROC |
|---|---|---|---|
| overlap | 179 | 0.240 | 0.834 (5 informative meetings) |
| non-overlap | 736 | 0.167 | 0.811 |
| substitutions vs match | 883 | 0.152 | 0.845 |
| **insertions vs match** | 781 | 0.041 | **0.725** |

**Insertion detection is our weakest arm too** (0.725, on only 32 insertion errors),
and insertions are exactly what kills every occupancy fusion arm
(`exp-2026-08-16-char-vote-homophones`: occupancy columns hold 14.2% of the remaining
oracle gap and fail the insertion gate). Soniox's separately estimated insertion AUROC
was 0.773; that uses a different system-specific risk set and **does not establish that
Soniox is better at this**.

Confidence does not degrade inside overlap. It also does not solve overlap, which is a
*recall* problem — deletions, which confidence cannot see.

### The cheap segment number is not a substitute

`exp(avg_logprob)` is what `oc_asr_server.py:175` returns today. Paired per-meeting
differences (`p_word` − `seg_conf`) are **positive in all six meetings**: +0.342,
+0.144, +0.161, +0.060, +0.209, +0.307. Macro 0.611 vs 0.815; its own segment-blocked
permutation gives p = 0.0040, so it is not noise — it is just much weaker. If anything
downstream wants a confidence signal, it has to ask for `word_timestamps=True`.

## The head-to-head: are the two signals the same signal?

Shared columns only — reference indices where **both** systems emitted an aligned word.
This excludes every deletion by either system and every insertion, so it is a fair
same-row comparison and **not** a comparison of overall confidence quality. n = 866
columns, 6 meetings.

**The errors overlap strongly. The confidences do not.**

| | count |
|---|---|
| neither system wrong | 706 |
| adapter only | 74 |
| Soniox only | 30 |
| **both** | **56** |

56 joint errors against a meeting-stratified independence expectation of **21.0**
(pooled expectation 12.9); odds ratio 17.8. 43% of adapter errors and 65% of Soniox
errors are shared.

But the confidence *rankings* are only weakly associated — Spearman on midranks, within
meeting: **0.157, 0.169, 0.175, 0.250, 0.347, 0.383** (pooled 0.351, secondary). Tie
structure differs a lot: 540 unique adapter values across 866 rows against 244 unique
Soniox values (71.8% tied).

And the two bottom-decile flags barely overlap at all:

| | count |
|---|---|
| both flagged | **8** |
| adapter only | 78 |
| Soniox only | 78 |
| neither | 702 |

86 flagged each (realized 9.93%). The meeting-stratified independence expectation for
"both" is **8.54**. Observed 8. **The two systems' least-confident columns are, to
measurement precision, independent draws.**

These are three different properties — binary error association, full-ranking
association, and overlap at one tail cutoff — and they need not agree. There is no
paradox. The pattern is consistent with common error *difficulty* plus system-specific
confidence ranking: they get the same columns wrong, and are unsure about different
ones.

### Does combining them help? Not established.

The preregistered parameter-free combination is the mean of the two within-meeting
percentile ranks. Its label was frozen separately for each system, because the two have
different error labels.

| target | own signal | other system's signal | mean-rank combination |
|---|---|---|---|
| adapter error | 0.8462 | 0.7487 | 0.8656 |
| Soniox error | 0.8298 | 0.7259 | 0.8581 |

Read the point estimates and stop. The **paired per-meeting differences** say the gain
is not established:

| target | mean gain | per-meeting range | meetings positive | LOO mean range | 6-cluster 95% (descriptive) |
|---|---|---|---|---|---|
| adapter error | +0.019 | −0.031 … +0.089 | 4 / 6 | +0.005 … +0.029 | [−0.013, +0.053] |
| Soniox error | +0.028 | −0.039 … +0.101 | 4 / 6 | +0.014 … +0.042 | [−0.015, +0.072] |

Both intervals include zero and both directions occur across meetings. **The
combination does not beat either signal on this evidence.** What survives is the
weaker, more useful statement: each system's confidence carries information about the
*other's* errors (0.749 and 0.726, well above 0.5), the rankings correlate only weakly,
and the flagged tails are independent. That is descriptive evidence of **non-redundant
ranking information** — which is what a confidence-weighted vote would need — without a
demonstrated gain from the one combination rule that was preregistered.

## The 247 windows and the gate

**Status: stopped, not finished. 102 of 247 windows are paired.** Completed 2026-08-17
on what exists; the decodes will not be resumed. Nothing in the sections above depends
on them.

The question is whether asking faster-whisper for word probabilities changes the
transcript. If it does, the confidences belong to a re-run and **cannot** be attached
retroactively to the frozen fusion input W.

Design: two full local passes over all 247 windows, CPU int8, `artifact-ct2-fixed`, the
run's own frozen `decode.json` with **exactly one field changed** — `word_timestamps` —
and the control arm's own per-window CTranslate2 seed (common random numbers). Codex
judged the 37-window eval-freeze overlap insufficient to carry a benchmark-wide
magnitude, so both passes cover all 247 rather than reusing the overlap alone.

Three **paired contrasts**, not additive components — WER is nonlinear and these do not
sum:

| contrast | measures |
|---|---|
| cached GPU `wt=F` vs local CPU `wt=F` | stack contrast (RunPod GPU vs local CPU int8) |
| local CPU `wt=F` vs local CPU `wt=T` | **the gate**, isolated |
| cached GPU `wt=F` vs local CPU `wt=T` | end-to-end |

**Gate criterion, frozen before the passes finished**: attachment to W is permitted
only if the normalized token sequence is identical in **all 247** windows, and no
analysis may afterwards be restricted to whichever windows happened to be stable.

**Disclosure.** A 2-window smoke was run before the preregistration, to size the job.
It showed (a) local CPU `wt=F` reproducing the cached control arm bit-exactly on both
windows, and (b) `wt=True` **changing the text on both windows** (segment count 17 → 19
on one). The gate *criterion* was frozen outcome-blind; the gate *outcome* was partially
visible beforehand. Recorded as a development observation rather than claimed otherwise.

> **Erratum, 2026-08-17.** That smoke was read too widely. This report previously said
> our decode *is* deterministic on this stack. On the 18 windows where a cached local
> `wt=F` control also exists, the re-run reproduces it in **16 of 18**, not 18 of 18.
> The claim "the cached CPU decode is bit-exactly reproducible" is withdrawn. What
> survives is measured below, next to the number it is needed for.

### The stop rule was violated, and the sample is a protocol deviation

The amendment froze a wall-clock stop at **23:45 EEST on 2026-08-16** and armed a
watchdog to enforce it. **The watchdog process died when the session hit its API limit,
and the decodes ran on to roughly 00:44 before being killed by hand — about an hour
past the frozen stop.** Per-window completion timestamps were never recorded, so the
realized set **cannot** be truncated back to the 23:45 boundary. There is no way to
reconstruct the sample the preregistration actually specified.

Realized: `wt=True` 102 windows, `wt=False` 133, verified **perfectly nested**
(intersection exactly 102). The analysis set is those **102 of 247**, spanning 82
meetings and all 10 cities (argos 15, athens 12, chalandri 8, chania 19, orestiada 3,
samothraki 5, sparta 16, vrilissia 8, xylokastro 7, zografou 9).

What survived and what did not:

- **The frozen stop rule was mechanically violated.** State it that way. The realized
  102-window sample is a protocol deviation and every quantitative summary computed on
  it below is **descriptive**, not the preregistered estimate.
- **Blindness survived.** The queue order was fixed by seed, applied to the full
  247-item substrate, before any contrast statistic existed; which windows landed in
  the set was never influenced by any outcome, and nobody chose n = 102 after seeing a
  result. The stop slipping changed the sample **size**, not its blindness.
- **The binary gate conclusion is invariant to the overrun, and this is checkable
  rather than argued.** Nine paired windows were decoded *before* the 20:05 amendment,
  so they are inside the 23:45 deadline set whatever time decoding stopped. Exactly
  **one of those nine** is normalized-identical. The deadline set therefore contained at
  least eight gate violations, and the gate is a conjunction. **The overrun cannot have
  created the failure.**
- **What the overrun does damage** is the precision and the framing of every
  *magnitude* here — the 7.7% change rate, the per-window distribution, the descriptive
  AUROC. An unenforced time stop can select on runtime, and runtime is not independent
  of what a window contains: slow, segmentation-heavy windows are the ones least likely
  to have finished. Combined with the disclosed city-clustered `argos` head, **the 102
  are not a uniform random sample of the 247** and nothing here is an unbiased estimate
  for the remaining 145.
- Had the gate **passed**, this caveat would bite much harder: a pass is a claim about
  windows one did not look at, and a runtime-conditioned, non-uniform sample is a bad
  place to make it from.

### Gate result: fails, decisively

**One measurement correction first, because it moves the headline number.** The decode
script stores `text` as `"".join(segment.text)`, and faster-whisper does not always put
a leading space on a segment — **505 of 1,677 segment boundaries here have none**. The
joined string therefore fuses the last word of a segment with the first word of the
next, and `word_timestamps` *moves segment boundaries*, so scoring the joined string
charges the gate for fusions the join invented at exactly the boundaries under test.
Tokens are built **per segment** below. On the joined string the same contrast reads
0.0804; the fusion-free number is **0.0770**. Both fail. The corrected pipeline now
reconciles exactly with the descriptive arm — same 29,437 tokens, same 2,946 errors —
which the joined-string version did not (a discrepancy Codex caught).

| | value |
|---|---|
| raw text identical | **0 / 102** |
| normalized token sequence identical | **1 / 102** |
| — and that one window differs only in punctuation/spacing | yes |
| pooled WER, `wt=False` as denominator | **0.0770** |
| symmetric edit rate | 0.0769 |
| op mix between the two settings (S / D / I) | 996 / 586 / 678 over 29,345 tokens |

Per-window change rate: min 0, p25 0.029, **p50 0.053**, p75 0.102, max 0.352.

| per-window change rate | windows |
|---|---|
| exactly 0 | 1 |
| (0, 0.02] | 17 |
| (0.02, 0.05] | 32 |
| (0.05, 0.10] | 24 |
| (0.10, 0.20] | 19 |
| (0.20, 1] | 9 |

Bins fixed before they were computed. The 10 most-changed windows carry **32.1%** of
all edits, so the effect is broad *and* right-skewed: it is not one pathological window,
and it is not evenly spread either.

**The gate fails.** 101 of 102 paired outputs changed after normalization, at a pooled
7.7% between the two settings. This is decisive for what the gate was for — these
confidences **cannot** be attached retroactively to the frozen fusion input W or to the
cached benchmark text. It is *not* a claim that all 247 windows would change, nor that
7.7% is the benchmark-wide magnitude.

### Is it `word_timestamps`, or is it just a different run?

18 of the 102 windows also carry a cached local CPU `wt=False` decode from
`exp-2026-08-12-decode-ablation` (arm A) — same config, same per-window seed, earlier
run. That file stores only the joined string, so both rows below are scored on the
joined string and are comparable only to each other.

| on the same 18 windows | raw identical | symmetric edit rate |
|---|---|---|
| re-run, same config (`wt=F` vs cached arm A) | 16 / 18 | **0.0139** |
| the gate (`wt=F` vs `wt=T`) | 0 / 18 | **0.0800** |

So our decode is **not** bit-deterministic across runs on this stack, and the earlier
"bit-exact" claim is withdrawn. The `word_timestamps` contrast is nevertheless **5.7×
the observed re-run discrepancy on matched windows**, which makes ordinary run-to-run
variability an implausible explanation for most of it. That is directional evidence, not
a decomposition: n = 18, one historical re-run, arm A's CPU thread count was never
recorded (this pass used 6), and the design cannot separate a causal `word_timestamps`
effect from an interaction between the setting and nondeterminism. **For the gate it
does not matter**: whatever the mechanism, the decode that produces the confidences
demonstrably does not reproduce W.

For context, the other two frozen contrasts on the same 102 windows. Both compare
against the cached benchmark text, which exists only as a stored string, so each is
reported twice — local side per segment, and both sides on the stored string:

| | symmetric edit rate | normalized-identical |
|---|---|---|
| **stack** (cached GPU `wt=F` vs local CPU int8 `wt=F`) | 0.0926 / 0.0956 | 2 / 0 of 102 |
| **end-to-end** (cached GPU `wt=F` vs local CPU `wt=T`) | 0.0979 / 0.1001 | 0 / 0 of 102 |

Neither tokenization is unambiguously right here and the choice does not change what
they say: **moving from the RunPod GPU to local CPU int8 already changes the text about
as much as `word_timestamps` does**, so this pass was never a way to obtain confidences
for the benchmark decode either. These do not sum — WER is nonlinear and they were never
a decomposition.

### Does asking for word timestamps delete more speech?

The direction this project cares about most. Reference-anchored, so
**agreement-with-OpenCouncil**, on the same 102 windows / 30,272 reference tokens:

| | WER | deletions | insertions | substitutions |
|---|---|---|---|---|
| `wt=False` | 0.1474 | 0.05173 | 0.02111 | 0.07456 |
| `wt=True` | 0.1489 | 0.05157 | 0.02398 | 0.07334 |

Paired deltas (`wt=True` − `wt=False`), 2,000 meeting-cluster bootstrap resamples over
the 82 meetings, seed 21, ratio-of-sums — **descriptive**, and four unadjusted
quantities:

| | delta | 95% (descriptive) |
|---|---|---|
| WER | +0.00149 | [−0.00562, +0.00935] |
| deletions | −0.00017 | [−0.00727, +0.00808] |
| insertions | **+0.00287** | [+0.00003, +0.00621] |
| substitutions | −0.00122 | [−0.00389, +0.00120] |

Nothing here establishes that `word_timestamps` improves or worsens WER. The insertion
interval only just clears zero, at a lower bound of +0.00003, and it is one of four
unadjusted quantities on a protocol-deviating sample — suggestive of a small insertion
cost, not established.

**And the near-zero deletion delta is cancellation, not stability.** Per-window
`D_true − D_false` is **nonzero in 82 of 102 windows**, `|Δ| ≥ 10` in **18**, and the
absolute values sum to **613** against a net sum of **−5**, ranging −32 to +48. In all
18 large-swing windows the emitted-token count moves the opposite way by nearly the same
amount — +48 deletions against 51 fewer emitted tokens, −32 deletions against 33 more —
so these are whole passages appearing and disappearing between the two settings, not
deletions being redistributed word by word. **The aggregate deletion burden is nearly
unchanged in this sample. Deletion behaviour is not stable, and the interval does not
establish equivalence.**

### Are the differences systematic?

Punctuation is already excluded — the normalizer keeps only `\w+` — so these are word
differences, and exactly **one** window's disagreement was punctuation-only. Segment
count differs in **82 of 102** windows (mean −0.28, range −11 to +9; more segments under
`wt=True` in 36, fewer in 46). Combined with the passage-level deletion swings, the
picture is that `word_timestamps` re-cuts the segmentation, the re-cut segments hit the
temperature-fallback and threshold ladder differently, and whole chunks of text move.
It is not a punctuation or tokenization technicality, and it is not scattered
single-word jitter either.

### Descriptive arm — agreement-with-OpenCouncil, never merged with the gold set

The label here is disagreement with **our own published corrected text**, not fidelity
to audio, and these confidences belong to the `wt=True` re-run, not to the cached
benchmark text. 102 windows, 82 meetings, 29,437 normalized token rows, 2,946 errors,
prevalence 0.1001.

| | mean within-meeting AUROC | pooled |
|---|---|---|
| `p_word` | **0.8412** | 0.8622 |
| `exp(avg_logprob)` segment proxy | 0.6048 | 0.7223 |

Average precision 0.405 against a 0.100 null. All 82 meetings have a defined AUROC;
leave-one-meeting-out stays in 0.8400–0.8430, which is what 82 clusters buys you.
Insertions-vs-match 0.8751, substitutions-vs-match 0.8342. Deletions are **34.6%** of
edit operations here, so edit-operation coverage is 65.4% — better than the gold set's
58.9%, on a different metric against a different reference.

**What must not be said about 0.8412 and the gold set's 0.8151.** They are close. That
is not evidence of anything. Different labels, different targets, different sampling
frames, different decode substrates — and this one's errors include places where
OpenCouncil's published text is itself wrong or edited. Their proximity is **not**
replication, not external validation, not equivalence, not transport from
agreement-with-OpenCouncil to fidelity-to-audio, and not robustness across substrates.
Both are above chance on their own task. That is the whole of it.

## Limits that must travel with every number here

- **27 cells, 6 meetings, 6 cities — one meeting per city, so meeting and city are
  fully confounded.** Clustering is by meeting. **This cannot support a population
  claim or any promotion gate.** A 6-cluster interval excluding a null is not
  significance.
- The gold reference is **one listener, one pass, no adjudication**. Conclusions are
  fidelity to the human-verified reference *under its transcription rules*, not
  unconstrained audio truth. 19% of core tokens were excluded as text-uncertain with
  their time masked out, and 17 spans the human judged as lost speech were never
  written into the reference.
- **The gold-set decode is `beam_size=2` CPU int8, not the benchmark's beam 5 on GPU.**
  Nothing here is evidence about the benchmark decode's confidences.
- **AUROC covers substitutions and insertions among emitted words only.** 41.1% of edit
  operations in this region are deletions, invisible by construction.
- **0.8151 and Soniox's 0.8167 are separate within-system estimands** over different
  emitted-token and error populations. "Nearly identical descriptive point estimates on
  the same recordings" is supported; "the adapter is as good as Soniox" is an
  equivalence claim and is not.
- **samothraki jul6 supplies 61% of the errors.** The macro average protects the
  primary discrimination estimate from it; nothing prevalence-dependent is protected.
- **The head-to-head is a conditioned subset** — no deletions, no insertions. It cannot
  speak to overall confidence quality.
- **Nothing was tuned on the gold set.** The primary statistic, the region, the
  alignment, the buckets, the nulls, the combination rule and its two labels were all
  written down before any outcome was computed. The paired-gain analysis, the
  meeting-stratified independence expectations and the Wilson intervals were added
  **after** the first numbers, on Codex's findings review, and are labelled here as
  what they are: uncertainty quantification on already-frozen estimands, not new
  estimands.
- The secondary analyses are sensitivity checks on one dataset, not independent
  replications. zografou jun18 contributes 2 errors and its 0.707 means almost nothing.
- **The 247-window arm's stop rule was violated and it never finished.** 102 of 247
  paired, and the 102 are not a uniform sample. Every magnitude from it is descriptive.
  The binary gate failure is the one conclusion that does not depend on any of this.
- **The gate failure is not a claim about the other 145 windows.** It is a claim that
  the attachment precondition is not met, which one counterexample is enough to settle.

## What this changes

- **Per-word confidence from `artifact-adapter-fixed` cannot be retrofitted.** Asking
  for it changes the transcript, so the probabilities belong to the decode that produced
  them. Any consumer that wants them has to re-decode and accept a different text — and
  that text is a different substrate from the frozen benchmark and from W. This is the
  same wall the Soniox work hit from the other side, where a confidence-bearing re-run
  forced the parallel **W-rt** substrate rather than being merged into W.
- A confidence-weighted fusion vote does have **two** real per-word signals rather than
  one, and they are measurably not the same signal — the errors co-occur strongly while
  the flagged tails overlap at chance. That is a precondition a weighted vote needed. It
  is **not** a demonstrated gain: the one preregistered combination rule did not beat
  its parts, with both intervals including zero.
- **And the precondition has since been tested and did not pay.**
  `exp-2026-08-16-w-rt-confidence` closed NEGATIVE: on the full 247-window W-rt
  substrate the Soniox occupancy arm and the minority-override arm each fitted a
  threshold that **never fires** in all ten leave-one-city-out folds, and the weighted
  identity vote gave −0.00035 with intervals including zero. A detector at AUROC ~0.82
  bought nothing there. Post-hoc, its AUROC on the decisions a fusion arm actually makes
  is 0.587–0.703 — weakest exactly where the mass is.
- So the honest reading is narrow. **A strong marginal error-detection AUROC does not by
  itself produce actionable fusion gain, and no downstream improvement from either
  confidence source is established.** Panel asymmetry (one voter of three carrying a
  confidence) and weak discrimination *conditional on the decisions fusion actually
  makes* are both plausible limiting mechanisms; neither has been isolated, and the
  0.587–0.703 figure makes "the signal quality is fine, the panel is the problem"
  specifically hard to defend. Do not name a binding constraint. Neither experiment was
  designed to identify one.
- Whatever a vote gains, it cannot touch the 41% of our edit operations that are
  deletions. Confidence is not a route to the deletion problem.
- If any serving-time consumer wants confidence, `exp(avg_logprob)` will not do.

## Did this arm deliver what it was for?

No. It existed to decide whether per-word confidences could be attached to the frozen
fusion substrate; the answer is that they cannot, and the arm that would have quantified
the cost of re-decoding was stopped at 41% coverage with its stop rule broken. What it
did deliver is a clean negative on the attachment question, at zero cost, plus a
correction to a determinism claim this project had started to lean on. The primary
claim of the ticket never depended on it and stands on the gold set.

## Cost and reviews

**Zero.** Local CPU only, no GPU pod, no paid API call.

Three Codex reviews at high effort:

- Job `06a6ad95a3d14af4bd53ba76308a6b66`, **before any scoring code**: renamed the
  analysis unit from "emitted word" to the normalized token instance it actually is,
  forced the paired contrasts to stop being called an additive decomposition, moved the
  permutation unit from the token row to the emitted-word block, required the deletion
  coverage to be recomputed for the adapter instead of inheriting Soniox's 22.8%,
  required undefined meeting AUROCs to be counted, banned splitting ties to force equal
  bins, required the two combination labels to be frozen separately, and required the
  second full 247-window pass rather than leaning on the 37-window overlap.
- Job `6dd131920dfd4afb998c62d8059e96e2`, **on the findings, before any claim was
  written**: killed "the combination beats each signal" (the paired per-meeting
  differences were then computed and the intervals include zero), replaced the pooled
  independence expectations with meeting-stratified ones (joint errors 12.9 → 21.0, so
  the enrichment is 2.7× and not 4.3×), forbade "exactly the chance rate", forbade
  calling the insertion gap a head-to-head advantage over Soniox, forbade an
  equivalence reading of 0.8151 vs 0.8167, and required Wilson intervals before the
  calibration curve was described.
- Job `f2d5d496bb414420a6aa490e1bb5a878`, **on the closing findings, before any gate
  claim was written**, asked directly whether the overrun invalidates the arm. It
  endorsed the blindness argument but called it insufficient on its own, and supplied
  the check that makes the binary conclusion invariant (the nine pre-amendment windows,
  of which exactly one is identical). It required "protocol-deviating, quantitative
  summaries descriptive" to be said explicitly; forbade calling 0.0139 a measured noise
  floor or 5.7× a causal effect ratio; required the cancellation finding to be supported
  by alignments rather than by the sum-of-absolutes alone; forbade every reading of the
  0.8412/0.8151 proximity; refused "the binding constraint is the asymmetry of the voter
  panel" as over-identified; and caught an accounting inconsistency — 29,437 rows
  against 29,377 tokens — that turned out to be the joined-string segment fusion
  corrected above. After the fix the two paths reconcile exactly.
