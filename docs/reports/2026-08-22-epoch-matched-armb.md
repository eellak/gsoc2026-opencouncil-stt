# The epoch-matched rerun: contiguous packs win; the spliced arm was never epoch-matched

2026-08-22 · `exp-2026-08-22-epoch-matched-armb` · CLOSED

The previous comparison failed on its own design: with a fixed *step* budget the two
arms trained to wildly different points on their curves (control 0.50 epochs, packs
5.82). This is that comparison run again with the budget matched in epochs.

## What was run

619 steps (≈2 epochs of the contiguous pack), one RTX A4000, three arms × three seeds
(13/29/47). Every adapter — including the incumbent — was scored by the same external
scorer on the same 39 frozen validation windows (32 meetings, 11,911 reference tokens),
with the frozen `CONTROL` decode config and a per-window ctranslate2 seed derived from
the window id.

**The incumbent was re-scored on this stack.** Its previously recorded 0.1542 came from
a different stack and is not used here; on this one it scores 0.16002. Comparing across
stacks is how this project has produced false results before.

## The rule, fixed before any number existed

Ship a challenger only if the **upper** bound of the paired, meeting-clustered bootstrap
interval against the incumbent falls below zero, Bonferroni-corrected across the three
arms (98.33%). Seed 13 was pre-declared as the shipping seed, so no seed selection
occurs. Step 619 is confirmatory; intermediate checkpoints are exploratory.

## Result

| adapter | WER | del | ins | sub | Δ vs incumbent | Bonferroni 98.33% CI | gate |
|---|---:|---:|---:|---:|---:|---|---|
| incumbent (`artifact-adapter-fixed`) | 0.16002 | 0.06196 | 0.02149 | 0.07657 | — | — | — |
| contiguous s13 *(ship seed)* | 0.14566 | 0.04450 | 0.02376 | 0.07741 | −0.01436 | [−0.02671, −0.00171] | **pass** |
| contiguous s29 | 0.13995 | 0.03980 | 0.02334 | 0.07682 | −0.02007 | [−0.03747, −0.00595] | **pass** |
| contiguous s47 | 0.13895 | 0.03971 | 0.02107 | 0.07816 | −0.02107 | [−0.03895, −0.00736] | **pass** |
| combined s13 | 0.15255 | 0.03896 | 0.03350 | 0.08009 | −0.00747 | [−0.02742, +0.01173] | fail |
| combined s29 | 0.15028 | 0.04047 | 0.02964 | 0.08018 | −0.00974 | [−0.03092, +0.00752] | fail |
| combined s47 | 0.14524 | 0.03753 | 0.02896 | 0.07875 | −0.01478 | [−0.03109, −0.00060] | marginal |

**The contiguous arm passes in all three seeds.** Relative improvement over the
incumbent is 9.0% (s13) to 13.2% (s47).

**It wins by deleting less.** The incumbent omits 6.20% of reference tokens; the
contiguous adapters omit 3.97–4.45%, roughly a third fewer, with substitutions flat.
The standing warning in this project is that a model which lowers WER by skipping hard
passages looks better and is worse. Here the opposite happened: the gain *is* recovered
speech.

Robustness, all pre-specified: 29 of 39 windows improve; the largest single window
contributes 24–28% of the effect, clear of the 67% domination that has misled this
project before; the interval is clustered by meeting, not by window.

## The spliced arm: a failed arm, but the cause is NOT established

Two of three combined seeds cross zero, and the third clears it by 0.0006. An arm that
passes in one seed out of three has not passed — quoting `comb_s47` alone would be
exactly the seed selection the protocol forbids.

Head-to-head against its own contiguous twin, the combined arm is **worse**: +0.00688
for seed 13 (95% CI [−0.00212, +0.01587]) and +0.01033 for seed 29 (95% CI [+0.00168,
+0.01919], excluding zero).

**But this comparison is confounded, and the confound is the same one this experiment
was built to fix.** Both arms got 619 optimizer steps at effective batch 8, i.e. 4,952
samples. The contiguous pack has 2,475 rows and the combined pack has 6,984:

| arm | rows | epochs at 619 steps |
|---|---:|---:|
| contiguous | 2,475 | **2.00** |
| combined | 6,984 | **0.71** |

The two arms were matched in **steps**, not epochs — the exact error recorded against
the previous run. The contiguous arm is epoch-matched to ~2 epochs and the combined arm
is not matched to anything; it is undertrained relative to its own data.

So the observed insertion rise (2.3% → 3.0–3.4% with deletions flat) is **not**
attributable to the synthetic joins. Undertraining produces its own error profile, and
nothing here separates the two explanations. The honest statement is:

- **At an equal number of optimizer updates, adding spliced packs did not help.** That is
  a real and useful budget result: given a fixed compute budget, spend it on contiguous
  packs.
- **Whether jitter-splicing helps or hurts at a matched epoch budget is unmeasured.** It
  would need ~1,746 steps on the combined pack to reach 2 epochs. That experiment was not
  run.

This was caught by an external review after the report was first written, not by the
design. The mechanism claim in the first version of this report was wrong and is
withdrawn.

## What this does not license

These 39 windows are the **validation** set, from the held-out cities argos and
orestiada, and they have been scored repeatedly across this project. A validation win
justifies promoting a candidate to testing. It is not a product claim.

Two design caveats are unchanged by this result. Only 2.5% of pack utterances are
corrections, so the packs dilute the correction signal ~1:8 relative to the incumbent's
training set. And the incumbent's own recipe was never re-run epoch-matched on its own
data, so part of the gap could be the epoch budget rather than the data.
