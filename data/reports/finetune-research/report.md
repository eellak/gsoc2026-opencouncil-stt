# Tiny-Whisper CPU auto-research — findings (Track 2)

_2026-06-23. whisper-base (72.6M, LoRA r=8 on q/v, encoder frozen), CPU-only on
the mini PC. Two time-boxed rounds, ~1 h wall-clock total. Method after
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): vary one
data/training axis, fine-tune, score val, keep/discard, log, repeat._

## TL;DR

1. **Fine-tuning on ~17 min of corrections cuts ordinary-speech WER hard and
   reliably** — `val_reg` (reviewed-unchanged utterances) drops from **0.674 →
   ~0.43** normalized WER (~24 pts) on *every* fine-tune. That is real domain
   adaptation, well outside noise.
2. **The hard residual correction cases barely move.** `val_corr` sits at
   **~0.61 ± 0.06** regardless of data mixture — and the seed-to-seed spread of a
   *single* config (0.062) is as large as the spread across *all* configs
   (0.064). So **no data axis is statistically distinguishable on `val_corr` at
   this scale**; the two "keeps" are inside the noise floor.
3. **What actually transfers to the large-v3 run:** the *composition* lever
   (corrections + a no-edit backbone) and "don't starve ordinary speech", plus a
   hard methodological lesson — **we need a bigger, lower-variance validation set
   before any data-mixture ranking is trustworthy.** lr/steps don't transfer.

## What this was

An automated fine-tuning experiment loop on the GPU-less mini PC. The lever is
the **data/training config**, not the architecture (Codex: data-mixture findings
transfer to a bigger model; model-rank sweeps do not). The recipe is held fixed
(whisper-base + LoRA r=8 q/v, encoder frozen, 40 optimizer steps, grad-accum 4,
greedy `language=el` decoding identical across every run) so val-WER deltas are
attributable to the data choice.

The point was never a deployable model — 1.8 h of corrections-scale data overfits
a tiny model. The point was **which knobs move the needle**, to aim the eventual
large-v3 GPU run.

## Setup

- **Data (frozen):** `data/asr/manifest.jsonl`, 900 clips / 20 high-yield
  meetings, built from `/api/export` corrections + meeting-JSON no-edit backbone,
  no-edit gated on `humanReview=true`. Decode-once-then-slice, 16 kHz mono,
  ±0.2 s padding, composite-id dedup, clip validation. Splits:

  | split | clips | min | what |
  |---|---|---|---|
  | `train` | 279 | 16.9 | corrections, 8 train cities |
  | `train_noedit` | 452 | 19.5 | no-edit backbone (for the composition axis) |
  | `val_corr` | 56 | 4.4 | corrections in orestiada+argos (held-out cities) |
  | `val_reg` | 113→70 | 4.9 | no-edit in the same held-out cities (regression guard) |

- **Baseline (zero-shot whisper-base, no FT):** `val_corr` 0.6071 · `val_reg`
  0.6736 (normalized WER). Every run is compared to this.
- **Selection rule:** *keep* only if `val_corr` improves **and** `val_reg` does
  not regress beyond +0.01.

## Results — all 16 fine-tune runs

Round 1 = one-axis sweep from a corrections-only FT baseline. Round 2 = combine
the round-1 keepers + remaining axes. Sorted by `val_corr` (lower = better).

| config | n_train | val_corr (Δ) | val_reg (Δ) | status |
|---|---|---|---|---|
| `sample_capped_oversample` | 180 | **0.6041** (−0.003) | 0.4321 (−0.242) | keep |
| `comp_backbone_1x` | 558 | 0.6056 (−0.002) | 0.4642 (−0.209) | keep |
| `comp_backbone_3x` | 731 | 0.6100 (+0.003) | **0.4283** (−0.245) | discard |
| `focus_acoustic` | 452 | 0.6115 (+0.004) | 0.4660 (−0.208) | discard |
| `sample_cat_balanced` | 2600 | 0.6145 (+0.007) | 0.4509 (−0.223) | discard |
| `base_ft` (corrections only) | 279 | 0.6174 (+0.010) | 0.4377 (−0.236) | discard |
| `r2_bb1x_balanced` | 2879 | 0.6189 (+0.012) | 0.4642 (−0.209) | discard |
| `r2_bb1x_capped` | 459 | 0.6248 (+0.018) | 0.4434 (−0.230) | discard |
| `r2_bb1x_capped_acoustic` | 639 | 0.6248 (+0.018) | 0.4321 (−0.242) | discard |
| `r2_bb1x_acoustic` | 731 | 0.6278 (+0.021) | 0.4453 (−0.228) | discard |
| `filters_strict` | 216 | 0.6381 (+0.031) | 0.4509 (−0.223) | discard |
| `lr_5e-5` | 279 | 0.6588 (+0.052) | 0.6057 (−0.068) | discard |
| `r2_bb1x_lr5e-5` | 558 | 0.6677 (+0.061) | 0.5377 (−0.136) | discard |

Finalist reseed (`sample_capped_oversample`, 3 seeds): `val_corr` =
**[0.6041, 0.6086, 0.6662]**, mean 0.626, spread **0.062**.

## Observations

- **`val_reg` is the robust, dominant signal.** Every fine-tune (except the
  under-trained `lr_5e-5`) improves ordinary-speech WER by 21–25 points, and the
  differences *between* mixtures (0.428–0.466) are small relative to the gain.
  Fine-tuning teaches the model the *domain* (council vocabulary, formatting,
  acoustics); that helps the easy majority far more than the hard tail.
- **`val_corr` is noise-limited.** Cross-config spread 0.064 ≈ single-config seed
  spread 0.062. The leaderboard order on `val_corr` is therefore **not
  trustworthy** — `capped_oversample`'s −0.003 "win" is a coin-flip. This is the
  single most important methodological takeaway: **56 val_corr clips is too few.**
- **Combinations did not stack.** Pairing the two round-1 keepers
  (`bb1x_capped*`) landed at 0.625, *worse* than either alone — exactly what you
  expect when the underlying differences are noise.
- **The one unambiguous result is negative:** `lr=5e-5` at 40 steps is clearly
  worse on both sets (undertrained). It confirms `lr=1e-4` for this step budget —
  and lr/steps don't transfer to large-v3 anyway, so this is a dead end by design.
- **No mixture beat the regression guard meaningfully**, but none *failed* it
  either (except low-LR): fine-tuning here is "safe" for ordinary speech, which
  was Codex's central worry (correction-bias degrading general ASR). At this
  scale, **it does not degrade — it helps.**

### Finalist per-category `val_corr` (directional only, tiny n)

| category | n | wer_norm |
|---|---|---|
| punctuation_capitalization | 12 | 0.577 |
| homophone | 6 | 0.597 |
| substitution_phonetic | 27 | 0.601 |
| place_name | 6 | 0.613 |
| noun_case | 9 | 0.616 |
| person_name | 3 | 0.714 |
| word_boundary | 8 | 0.762 |
| verb_inflection | 5 | 0.809 |

Phonetic substitutions and punctuation (the bulk of corrections) sit near the
mean; morphology-heavy categories (verb inflection, word boundary) are hardest —
but every cell has n ≤ 27, so read this as a hypothesis, not a measurement.

## Direction — what to carry into the large-v3 GPU run

1. **Keep the composition recipe:** corrections **+ a no-edit backbone**. It is
   the lever with a coherent story (helps `val_reg`, doesn't hurt `val_corr`) and
   it is exactly the kind of data-mixture choice that transfers to a bigger model.
   Start at ~1× backbone; 3× pushed ordinary-speech WER lowest but is not needed.
2. **Capped oversampling of dominant categories** is a reasonable default (it was
   the nominal best, and it's cheap), but **do not trust its ranking** until the
   val set is bigger.
3. **Drop** error-type focus, strict filtering, and low-LR — none helped.
4. Re-tune lr / steps / epochs on the GPU; they do not transfer from a 40-step
   CPU run.

## Next steps (in priority order)

1. **Fix the validation set first — this is the bottleneck.** Use the *full*
   human-corrected pool in orestiada+argos (~9.9 k utts) for `val_corr`, not 56
   clips, and add a `val_reg_other_meeting` view (no-edit from held-out-city
   meetings that contributed no corrections). Target seed-spread < 0.01.
2. **Add meeting-level bootstrap CIs and ≥3 seeds per config** by default;
   optionally meeting-level k-fold (3–5) across all cities for a lower-variance
   read. Then re-run this exact sweep — with a real val, the mixture ranking
   becomes meaningful.
3. **Scale the train data** (more meetings + the backbone) toward the 30–50 h
   "first useful checkpoint" tier before reading mixture effects as real.
4. **Then** run the large-v3 LoRA on GPU (Track 1 / Kaggle) with the composition
   recipe, and compare `val_reg`/`val_corr`/per-category to this baseline.

## Caveats / what was capped for the 1 h budget

- 20 meetings (not all 272); `val_corr` 56 clips; single seed per axis (finalist
  reseeded to 3); 40-step CPU runs; `val_reg` drawn from the same meetings as
  `val_corr` (same-meeting regression view — correlated acoustics, so the
  `val_reg` gain is partly "same rooms/speakers"). All of this inflates variance
  and is the reason the `val_corr` signal is inconclusive — by design, to fit the
  hour. Nothing dropped was silent; see the logs.
- **Reproducibility:** frozen `manifest.jsonl` (+ `dataset_stats.json`, source
  mp3 hashes), full traces in `loop.log`, every run in `leaderboard.jsonl`,
  `results.tsv`. Code: `eval/autoresearch/{prepare_asr,experiment,loop,round2}.py`.

```
build:  python eval/autoresearch/prepare_asr.py
round1: python eval/autoresearch/loop.py   --budget-min 36
round2: python eval/autoresearch/round2.py --budget-min 13
```
