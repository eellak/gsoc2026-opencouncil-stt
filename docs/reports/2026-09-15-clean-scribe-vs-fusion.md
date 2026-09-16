# Scribe in `no_verbatim` mode beats the fusion: 0.09294 against 0.09913

Run `2026-09-15-clean-scribe-baseline-vs-fusion-no-verba`, a clone of the 391-window
benchmark with one provider added: `scribe-v2-clean`, ElevenLabs Scribe v2 called with
`no_verbatim: true`. 391/391 windows, 0 errors. Every other column was copied from the
source run, so the comparison is on identical windows and identical references.

Until this run, every "Scribe" column this project ever published was Scribe in its
**default verbatim mode**. The production pipeline that produces the reference has always
called it with `no_verbatim: true`.

## What was measured

| system | WER | sub | del | ins |
|---|---:|---:|---:|---:|
| **Scribe v2 `no_verbatim`** | **0.09294** | 0.04862 | 0.02118 | 0.02314 |
| fusion-rules (temp=0) | 0.09913 | 0.04634 | 0.01866 | 0.03413 |
| Scribe v2 verbatim | 0.14023 | 0.05546 | 0.00955 | 0.07523 |
| Soniox `stt-async-v5` | 0.15019 | 0.06348 | 0.01229 | 0.07442 |

Paired, clean minus fusion, on all 391 windows: **−0.00619**, 219 better / 26 tied / 146
worse, zero leave-one-out sign reversals, largest single window **12.0%** of the net edit
delta. On `wer-nofillers` the gap is −0.00606, so it is not explained by the scorer's
hesitation lexicon. Both training-disjoint cities agree: argos 0.07985 against 0.08720,
orestiada 0.05286 against 0.05967.

## The fusion under test was never the fusion we would ship

`eval/live_fusion/providers.py` declared `no_verbatim: True` in `SCRIBE_PARAMS` and then
built the multipart body by hand without that field. Every fused window in every published
number therefore consumed **verbatim** Scribe. The TypeScript production port has always
sent it.

Worse than the missing field: `params_sha` hashes `SCRIBE_PARAMS`, so the cached responses
sit at exactly the sha a corrected request now computes. A naive rerun would have served
verbatim recordings as clean ones. Fixed 2026-09-16 — the body is derived from the
parameter dict instead of retyped, the scribe cache moved to `scribe-nv`, and a test
asserts field parity between the request and its own cache key.

So the honest statement of this result is: **clean Scribe beats the fusion that was built
on verbatim Scribe.** The clean-input rematch has not been run.

## "The fusion helps on hard windows" does not survive an independent sort

Binning the 391 windows by *clean Scribe's own* WER showed fusion winning the hardest
quartile. That sort is not admissible: the last quartile is where the baseline got
unlucky, and regression to the mean hands any other system an advantage there.

Re-sorting the same windows by **Soniox's** WER — independent of both contestants — flips
the finding:

| quartile (by Soniox WER) | Soniox | clean | fusion | fusion − clean |
|---|---:|---:|---:|---:|
| Q1 (easiest) | 0.05058 | 0.03854 | **0.03649** | **−0.00205** |
| Q2 | 0.10272 | 0.06682 | 0.06812 | +0.00130 |
| Q3 | 0.15956 | 0.08865 | 0.09900 | +0.01035 |
| Q4 (hardest) | 0.30292 | 0.18602 | 0.20248 | **+0.01646** |

Fusion wins the **easiest** quartile and loses worst on the hardest. Under the biased sort
the same table read Q1 +0.01343 … Q4 −0.00223 — the exact opposite.

A perfect quartile router, sending windows to fusion only where fusion wins, would buy
**0.00054** pooled. Difficulty routing is dead on value, whatever feature drives it.

## Headroom, and what it is worth

Per-window oracle over {clean, fusion} = 0.08411, i.e. **−0.00883** against clean. Adding
Soniox moves it to 0.08385; all six providers give 0.08297. The oracle has no null: it is
the minimum of two noisy numbers, and no test-retest of clean Scribe exists to say how much
of it is noise.

## What this does not license

- **It is agreement-with-OpenCouncil.** The reference is our own published text, produced
  by a pipeline whose transcription step is this same vendor in this same mode. No fraction
  of the −0.00619 can be attributed to transcription quality from these data alone. The
  `wer-nofillers` result rules out one narrow mechanism, not shared normalization, broader
  disfluency suppression, fixer edits, or correlated recognition errors.
- **"Clean transcribes Greek better" is not supported.** Nothing here was scored against a
  human who listened.
- **"Fusion wins on substitutions" is not supported.** Against verbatim Scribe, the
  substitution gain (−0.00912) and the deletion loss (+0.00911) cancel to four decimals:
  that is alignment reclassification, not a set of executable patches.
- **The pool is contaminated.** 105 of 203 benchmark meetings are in the training set.

## The number that decides what happens next

Clean Scribe's deletion rate is **0.02118** against verbatim's **0.00955** — roughly 1,290
reference words that verbatim emits, clean drops, and the reference kept. That is both the
fidelity red flag in this result and the cheapest remaining opportunity: the donor is the
same vendor in the same request shape, and both outputs already exist on disk for all 391
windows. It is tracked as `exp-2026-09-16-deletion-recovery`.

Reviewed before any of this was written up: Codex Astra (`a81dbae2`) and Claude Fable 5.1,
independently, on the same brief. Three arithmetic corrections above are theirs; the
Soniox re-sort and the 0.00054 routing ceiling came out of that review.
