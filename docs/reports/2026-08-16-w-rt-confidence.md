# Soniox per-word confidence on all 247 windows: no arm met its criteria

2026-08-16 · `exp-2026-08-16-w-rt-confidence` ·
prereg [`2026-08-16-w-rt-confidence-prereg.md`](../specs/2026-08-16-w-rt-confidence-prereg.md)
(revision 2) · sixth pass over the same 247 windows · **zero GPU, zero paid API**

**None of the three confirmatory confidence-based fusion rules met the preregistered
success criteria.** Arms O and M selected the "never fire" threshold in every
leave-one-city-out fold and changed nothing. Arm A's Bonferroni-adjusted interval
included zero and both error-component gates failed. Confidence did show
permutation-attributable signal inside A, and the non-promotable variant O2 produced
an exploratory interval below zero, but neither supports promotion.

This was a **development run by design** (prereg §1 and §9, on Codex correction 1): the arm
classes were chosen by a human who had already seen five reference-informed passes
over these windows, so no arm ships on this run whatever it scores.

Everything below is **agreement-with-OpenCouncil**, not fidelity-to-audio. "Correct",
"real" and "oracle" all mean *agreeing with the human-corrected reference*.

## The substrate had to be rebuilt, and that is half the story

The cached Soniox text for these windows came from the paid `stt-async-v5` and carries
**no confidence values** — the client discarded them. The only free path is the
realtime `stt-rt-v4`, a different model, so its text differs. Soniox is one of W's
three voters, so re-running it changes W itself. Grafting new confidences onto the old
text was ruled out: the earlier probe found only 14 of 27 gold-set cells reproducing
token-identically between two runs of the *same* model, so a token-by-token graft
across two *different* models has no defensible alignment.

So a parallel substrate, **W-rt**, was built. Same 247 windows (post-removal of the 6
sealed temporal-holdout windows, which stay sealed and were never transcribed), same
144 meetings, 10 cities, 74,917 reference tokens, same references. `scribe-v2-clean`
and `oc-runpod-fixed-2026-08-10` byte-identical from cache. Only the Soniox arm is new.
Nothing frozen was modified: new cache root, new result file, no edit to `fusion_lab`,
`msa`, `column_classes` or `scoring`.

### Acquisition

247/247 windows, **0 failures, 0 retries**, 2,041 s wall clock (34 minutes) at **18
concurrent** free realtime WebSocket sessions — one session per ~140 s window, so no
segmentation or merging was needed. Protocol frozen before the batch: `stt-rt-v4`, PCM
s16le mono 24 kHz, 100 ms frames at ~1x, `endpoint_detection=False`, `--lang el`, first
protocol-valid success retained unconditionally, temp key re-read per session.
**Silence trimming stayed OFF** (measured at ~11% of real words lost — this project's
exact failure mode). One disclosed deviation: concurrency was ramped 12 → 18 after a
12-window pilot came back 12/12 clean; those 12 windows are kept under the
first-success rule. The `maxJobs = 18` figure inherited from the user's `soniox-core`
held on the first attempt.

The token stream is clean: 78,425 words → **79,007 normalized units**, **zero** missing
a timestamp, **zero** missing a confidence, **zero** invalid confidences, zero residual
non-final tokens, 4 words dropped by normalization, 483 words splitting into more than
one normalized token.

### The model swap, descriptive only

**This is not an experimental comparison.** It confounds model, decode path, pacing and
non-determinism, and it is unpaired in any causal sense. It is here because it prices
how much of W depended on `stt-async-v5`.

| | WER | del | ins | sub |
|---|---:|---:|---:|---:|
| old W (`stt-async-v5`) | 0.10046 | 0.02032 | 0.03743 | 0.04271 |
| **W-rt** (`stt-rt-v4`) | **0.09931** | 0.02050 | **0.03557** | 0.04323 |
| old whole-window vote V | 0.12012 | | | |
| W-rt whole-window vote V | 0.12466 | | | |
| old column oracle | 0.04748 | | | |
| W-rt column oracle | 0.04799 | | | |

On these 247 windows W-rt's observed WER was **0.00115 lower** than old W's, almost
entirely through insertions. **This descriptive model-swap comparison does not
establish that `stt-rt-v4` is generally non-inferior to `stt-async-v5` as a fusion
voter** — and it points the other way in two places: as a whole-window *selection* the
realtime trio is 0.0045 **worse**, and the alignment-conditional column oracle is
0.0005 **worse**, i.e. the ceiling moved up. The prediction written before the run
(W-rt would be worse) was wrong.

The W-rt column census, which is **its own** and not the old one:

| class | columns | share |
|---|---:|---:|
| `agree` | 62,615 | 77.93% |
| `exact_2_of_3` | 6,960 | 8.66% |
| `two_present_same` | 4,491 | 5.59% |
| `singleton` | 4,301 | 5.35% |
| `unresolved_three` | 1,092 | 1.36% |
| `unresolved_two` | 888 | 1.11% |
| total | 80,347 | |

3,167 of the 4,301 singleton columns are ones where **Soniox alone** heard a word — the
realtime model emits a great deal the other two systems do not.

## The arms

Leave-one-city-out, only held-out outputs scored, paired meeting-clustered percentile
bootstrap, 10,000 replicates, seed 7, no refitting inside replicates. Confirmatory
family = {O, A, M}, Bonferroni-adjusted central two-sided **98.333%** interval
(quantiles 0.8333% / 99.1667%).

| arm | eligible cols | fired | WER | ΔWER vs W-rt | CI95 | adjusted CI | del gate | ins gate | outcome |
|---|---:|---:|---:|---:|---|---|---|---|---|
| **W-rt** | | | 0.09931 | — | — | — | 0.02050 | 0.03557 | baseline |
| **O** occupancy | 3,167 | **0** | 0.09931 | 0.00000 | [0, 0] | [0, 0] | pass | pass | **no change** |
| **M** majority override | 1,515 | **0** | 0.09931 | 0.00000 | [0, 0] | [0, 0] | pass | pass | **no change** |
| **A** weighted vote | 1,889 | 750 | 0.09896 | −0.00035 | [−0.00105, +0.00027] | [−0.00124, +0.00039] | **fail** | **fail** | **fails** |
| O2 *(variant, not in family)* | 811 | 289 | 0.09903 | −0.00028 | [−0.00051, −0.00006] | n/a | **fail** | **fail** | not promotable |

### O and M did not fail a gate. They refused to fire.

In every one of the ten leave-one-city-out folds, both O and M selected the operational
**"never fire" threshold, τ = 1.01**. Zero firings, zero windows changed, ΔWER exactly
0.0. On nine cities of training reference text, no threshold anywhere on the
preregistered grid beat doing nothing.

The prereg predicted O would fail the *insertion* gate. That prediction was right about
the mechanism and wrong about where it would surface: the fitting procedure never got
as far as the gate.

The **ungated control O-all** shows the trade-off it was avoiding. Firing on every
eligible Soniox-only singleton:

| | WER | del | ins |
|---|---:|---:|---:|
| W-rt | 0.09931 | 0.02050 | 0.03557 |
| O-all | 0.12624 | **0.01285** | **0.07020** |

The deletion rate falls about **37%** — the occupancy material is genuinely there — but
the insertion rate nearly doubles and WER rises by 0.02694 (CI excludes zero, on the
wrong side). 210 of 247 windows get worse. Separately, the hindsight column oracle
favoured the Soniox candidate in **547** of the 3,167 eligible columns; in rounded
counts O-all bought about **573 fewer deletion errors for about 2,594 more insertion
errors**.

**M-all** is the same story on identity: WER 0.11212, ΔWER +0.01281, substitutions
0.04323 → 0.05608. Overriding 2-of-3 majorities remains a losing move — the fourth
independent time this substrate has said so — and confidence does not rescue it at any
threshold on the grid.

### A: attributable signal, no result

Arm A gives each present system a weighted vote in the 1,889 `unresolved_two` /
`unresolved_three` columns where Soniox has a token: Soniox's weight is its
`conf_min_lex`, and the other two systems get the preregistered constant **k = 0.5**,
chosen so Soniox wins an otherwise 1-against-1 contest exactly when its confidence
exceeds the production operating point 0.5. **The vote is asymmetric — only one of
three systems has a confidence signal — and that is its central weakness**, stated
before the run and not resolved by it.

It fired 750 times, changed 150 of 247 windows (48 better, 154 tied, 45 worse), and
landed ΔWER −0.00035 with a 95% CI of [−0.00105, +0.00027] and an adjusted CI of
[−0.00124, +0.00039]. Both include zero. Both rate gates fail, narrowly: deletions
0.020503 → 0.020583, insertions 0.035573 → 0.035653. Five cities better, five worse;
the exact paired city sign test gives p = 1.0. Of its 750 firings, 169 match the column
oracle and **301 go against it**.

A **did** meet the preregistered permutation-attribution criterion. With confidence
permuted within (meeting × eligibility) and the whole procedure rerun, its observed
ΔWER was at least as favourable as **every one of 200** replicates (one-sided Monte
Carlo p = 1/201 = 0.00498; null p05 = −0.000200). So the confidence values are doing
something a shuffle does not. **This does not rescue the confirmatory result:** the
adjusted interval included zero, both component gates failed, and the sensitivity
instability clause fired.

That clause: the declared `k` envelope moves the effect from **+0.000013** at k = 0.3 to
**−0.00068** at k = 0.7, whose 95% CI [−0.00127, −0.00015] excludes zero. The sign
changes across the envelope, so under the preregistration nothing k = 0.7 did may be
promoted, and A does not ship on any k.

For O and M the permutation null is **degenerate** — every one of the 200 replicates
returns exactly 0.0, because the fitting procedure declines to fire under any
reassignment of confidence. The attribution test is therefore uninformative for those
two arms, and no attribution claim is made from it.

### O2, the variant that came closest and cannot be used

O2 takes the Soniox token in `unresolved_two` columns when confidence clears τ. It fit
**τ = 0.95 in all ten folds**, fired on 289 of 811 eligible columns, changed 44 windows
(20 better, 221 tied, 6 worse), and of its firings **74 match the column oracle against
8 that go against it** — by far the cleanest firing profile of any arm here. Five cities
better, none worse (exact sign test p = 0.0625); the top meeting supplies 19% of the
improvement, so the domination check does not fire.

Among O, M, A and O2, **only O2 had a reported interval entirely below zero**: its
exploratory 95% CI was [−0.00051, −0.00006]. It cannot be promoted, for three separate
reasons written down before it was run:

1. the preregistration placed it **outside the confirmatory family** as a declared
   variant, so it has no adjusted interval and no path to promotion;
2. **both rate gates fail** — by two tokens each (deletions 0.020503 → 0.020529,
   insertions 0.035573 → 0.035599), which is the "a pure substitution rule can still
   move the scorer's alignment" effect the char-vote experiment already documented;
3. its interval is **not on the same footing** as the family's: 95% exploratory versus
   the family's adjusted 98.333%.

## Why: what confidence can actually discriminate here

**Post-hoc, reporting only.** Written after the arms were scored, never a gate, never
visible to `fit` or `apply`, and the column oracle it reads is hindsight. For each arm's
eligible columns: label 1 = the oracle wants the Soniox candidate, label 0 = the oracle
wants what W-rt already has, and columns where the oracle wants a **third** thing are
excluded because they are not the decision the arm makes.

| arm | oracle wants Soniox | oracle wants W-rt | oracle wants a third thing | prevalence | AUROC of confidence |
|---|---:|---:|---:|---:|---:|
| O (occupancy) | 547 | 2,620 | 0 | 0.173 | **0.618** |
| M (majority override) | 203 | 1,312 | 0 | 0.134 | **0.587** |
| A (identity, tie set) | 405 | 408 | 1,076 | 0.498 | **0.703** |
| O2 (identity, `unresolved_two`) | 138 | 49 | 624 | 0.738 | **0.673** |

In this diagnostic, AUROC ranges from **0.587 to 0.703**, below the **0.8167** the
gold-set probe measured for confidence predicting that an *emitted* Soniox word is an
error. **These are not directly comparable estimates of degradation** — different task,
different population, different labels, different exclusions. What they do say is that
on the four decisions a fusion arm actually has to make here, confidence carries
somewhere between weak and moderate discrimination, and the two decisions with the most
theoretical room (occupancy at 3,167 eligible columns, majority override at 1,515) are
the two where it discriminates least. M is the weakest at 0.587; O is next at 0.618, on
the largest absolute oracle-positive pool (547 columns).

That is enough to explain the whole result. At 0.618 discrimination and 17.3%
prevalence there is no threshold on the grid at which restoring Soniox-only words pays,
which is precisely what ten independent folds concluded without being told.

## Cost

**Zero.** Free Perplexity temp key on the realtime path, local CPU, no GPU, no paid
API call. 34 minutes of wall clock for 9.6 audio hours. The ~$0.82 `stt-async-v5` run
that `exp-2026-08-16-soniox-confidence` proposed was **not** spent, and this result is
an argument against spending it before the deletion side of the problem is attacked
some other way.

## Limits that travel with every number here

- **Development run, not confirmatory.** The arm classes and eligibility rules were
  chosen after five reference-informed passes over these same 247 windows.
  Leave-one-city-out prices *parameter* overfitting only. This is the sixth pass.
- **The bootstrap is conditional on the ten fitted thresholds**, with no refitting
  inside replicates. It is not an interval for a deployable trained arm and it does not
  include threshold-selection variability. That O2's threshold came out identically
  (0.95) in all ten folds is descriptively reassuring and does not turn its interval
  into full-pipeline uncertainty. Nested refitting was declined in the prereg with a
  stated reason; the honest substitutes are the permutation null and the second-draw
  obligation below.
- **The estimand is the finite frozen benchmark**, plus weakly future meetings inside
  these ten cities. Ten cities are too few for a city-level bootstrap, so per-city
  deltas and an exact paired city sign test are reported instead, both low-powered.
- The rate gates are **point-estimate operational constraints**, not evidence that the
  population deletion or insertion rates are non-inferior.
- `stt-rt-v4` confidence is **not** `stt-async-v5` confidence, and W-rt is **not** W.
  Nothing here transports to the shipped fusion stack without a paid re-run.
- Realtime Soniox is not deterministic (97.8% run-to-run word agreement measured by the
  probe). **W-rt is one draw.** The prereg's confirmation obligation — a second
  independently collected substrate under the identical protocol, applying the frozen
  rules without retuning — was **not triggered**, because no arm passed the gates. If
  any arm is ever revived, that draw is owed first.
- Agreement-with-OpenCouncil, scored with `eval/controlled_eval/scoring.py`. Comparable
  to `exp-2026-08-16-composition-over-selection`, **not** to the published leaderboard.
- The 6 sealed temporal-holdout windows were removed by the same explicit filter before
  anything was computed and were never transcribed. They stay sealed.
- `soniox-tools` at `/home/harold/projects/soniox-tools` is **not a git repository**, so
  the `on_token` / `--json` tooling this depends on has no commit or revision to cite.

## Reviews

Two Codex passes at high effort, each before the thing it governed.

- Job `b71f2dca0cad451db62cfb8f65e9d08e`, **on the preregistration before any arm was
  implemented**. It reclassified the whole run as development rather than confirmatory
  and attached a confirmation obligation; it required the ungated O-all / M-all controls
  and the confidence-permutation null with the frozen attribution criterion, without
  which a τ = 0 win would have been mislabelled a confidence result; it forced the
  estimand and clustering statement and the city-level sign test; it replaced the
  intersection stop rule with all-or-nothing; it made the acquisition protocol fully
  explicit including the retention rule; it banned transferring old-W census facts to
  W-rt; it demanded the complete confidence-to-column mapping and the invalid-confidence
  rule; it disambiguated O2; it **killed the false argument** that an added wrong token
  is an insertion by construction (global alignment can convert a deletion or a
  substitution instead); and it supplied the acceptance-test list now in
  `eval/tests/test_w_rt.py`. Declined, with reasons recorded in the prereg: nested
  bootstrap refitting, and discarding the failed-window characterisation.
- Job `3b441ec3b47b48dbbaf49caa0ce1ed84`, **on these findings before any claim was
  written**. Of seven sentences submitted it rejected six. It replaced "confidence does
  not beat the vote" with "did not meet the preregistered success criteria" (failure to
  demonstrate is not demonstration of failure); it caught a **factual error** — M at
  0.587, not O at 0.618, is the weakest AUROC; it forbade "collapses" for the 0.82 → 0.6
  comparison as not directly comparable; it forbade "real words" for oracle-positive
  columns as an audio-fidelity claim this experiment cannot make, and corrected 2,600
  from a total to an increase; it cut "not worse as a fusion voter" back to a
  descriptive statement about these 247 windows; it required that A's permutation result
  be stated as attribution against a shuffle rather than as a WER benefit; and it noted
  that "the only arm whose CI excluded zero" was false, since O-all's interval also
  excludes zero, harmfully.

## Deliverables

- [`eval/controlled_eval/w_rt.py`](../../eval/controlled_eval/w_rt.py) — the W-rt
  substrate builder and the confidence-to-column mapping.
- [`eval/controlled_eval/exp_w_rt_confidence.py`](../../eval/controlled_eval/exp_w_rt_confidence.py)
  — arms, controls, permutation nulls, diagnostics.
- [`scripts/run_soniox_rt_bench.py`](../../scripts/run_soniox_rt_bench.py) — the
  resumable free-path transcription runner.
- [`eval/tests/test_w_rt.py`](../../eval/tests/test_w_rt.py) — 46 acceptance tests.
- `eval/controlled_eval/results_w_rt_confidence.json` — every number above.
- Caches, outside git: `~/.cache/oc-public/composition-rt-2026-08/`
  (`soniox-tokens/`, `manifest.json` with per-window SHA-256, `run_log.json`,
  `align_f399c791caa2c159.json`).
