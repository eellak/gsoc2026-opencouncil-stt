# Validation set: subsampling the held-out cities, and a leakage check

Date: 2026-06-24

## What this is about

The split mechanics were already settled: TEST is the temporal hold-out (meetings
reviewed from June 2026 onward), the split unit is the whole meeting, and membership
is speaker-exclusive. What stayed open was the validation set. We had it as two whole
held-out cities — Orestiada and Argos, all their speakers — which is what the first
LoRA run used.

Two things came up when we looked closer:

1. Christos's point was really about validation *efficiency*: running val on every
   hour of those two cities is slow. Could we instead validate on a few specific
   speakers from there with enough speaking time?
2. A leakage worry: the included corrections from Orestiada/Argos were hand-picked
   during review. Did any of those already-corrected utterances end up in training,
   which would inflate the LoRA numbers?

Both were checked against the code. Notes below.

## The leakage check: clean

The split is strictly city-disjoint. In `eval/autoresearch/prepare_asr.py:55`,
`VAL_CITIES = {"orestiada", "argos"}`, and the notebook (cell 5) assigns:

```
train_src = [r for r in rows if r["city_id"] not in VAL_CITIES ...]
val_src   = [r for r in rows if r["city_id"] in     VAL_CITIES ...]
```

So every Orestiada/Argos row — including the 102 + 91 hand-picked includes — went to
validation, never to train. The model never saw them during fine-tuning. The −20% /
−36% deltas are not inflated by training on already-corrected text.

There is still a real caveat, just a different one: `val_corr` (191 clips) *is* our
curated includes from those two cities. It didn't leak, but it's a hand-selected,
non-representative slice — fine as a "hard cases" probe, not a number to quote as the
headline. The clean signal is `val_reg` (280 clips of random reviewed no-edit speech
from the same cities), which has no curation and improved more (−36% WER). That's why
the re-test plan already says: headline = normalized WER, and grow val toward the full
held-out pool.

## The efficiency idea: validate on selected speakers, not all hours

Running val on the full held-out pool (~9,900 utterances in Orestiada + Argos) is slow
on every checkpoint. The fix is to validate on a fixed, seeded subset of whole
speakers with enough speaking time. Speaker-exclusivity is automatic — whole cities
are held out, so none of their speakers are in train anyway. The only choice is which
speakers to evaluate on, for speed and a stable signal.

Two validation sets, each for a different job:

| set | what it is | used for |
| --- | --- | --- |
| fast val | seeded ~10-speaker subset of the held-out cities (see thresholds below) | early stopping, checkpoint choice (runs in minutes) |
| full held-out pool | all of Orestiada + Argos | periodic / final solid number with seeds + CIs |

## Setting the thresholds from the data

`scripts/val_speaker_minutes.py` computes per-speaker speaking minutes from the cached
meeting JSON (grouping by `speakerTag.personId`). Current cache: 14 Orestiada + 32
Argos meetings.

| | Orestiada | Argos |
| --- | --- | --- |
| total cached speech | 1022 min | 1000 min |
| distinct speakers | 44 | 131 |
| speakers < 1 min (junk tail) | 15 | 78 |
| speakers ≥ 3 min | 22 | 37 |
| largest single speaker | 172 min | 363 min |

The thresholds below were settled by cross-checking the distribution against the ASR
literature (Grok, sourced) and a Codex second opinion; both converged on the same
numbers.

| parameter | value | why |
| --- | --- | --- |
| reliability floor | ≥ 3 min/speaker | ~3 min ≈ 400 words — where per-speaker WER stabilizes (WER variance ~1/N). Also drops the un-named diarization tail. |
| fast-val budget | 60 min/city → 120 min total | ≈ 20k words, above the ~10k-word (~1 h) per-language significance rule, yet ~16× faster than the ~2000-min full pool. |
| speakers | 12/city (24 total) | diversity without being slow; both cities have enough eligible speakers (22 and 37 ≥ 3 min). |
| per-speaker cap | 5 min target, 8 min hard cap | keeps any one speaker under ~13% of a city's val, so the 363-min Argos speaker (and 172-min Orestiada one) can't drive checkpoint choice. |
| buckets | val_corr and val_reg separate | val_reg (random no-edit) is the **primary** early-stopping/checkpoint metric; val_corr (curated includes) is a diagnostic alongside, not mixed into the score. |

Using val_reg as the decision metric also answers the curation-bias worry: training
decisions ride on the non-hand-picked speech, not on the includes we selected.

Selection rule: apply the 3-min floor, then seeded-pick 12 speakers per city up to the
60-min budget, sampling ~5 min per speaker with an 8-min hard cap. Reproducible by
seed, no hand-picking — consistent with the "no human picking" principle, the same way
the `<5%` edit-fraction cutoff was set from an observed gap rather than a guess.

**Main risk** (flagged by Codex): the 3-min floor removes the messy short-utterance
tail that real deployment audio has, so early stopping can underweight it. Mitigation:
keep the full held-out pool with seeded bootstrap CIs for the periodic/final
measurement, which catches any regression in that tail.

Early-stopping mechanics (from the literature): evaluate every ~50–500 steps, monitor
val **WER** not just loss (in ASR, loss can fall while WER rises), patience 2–3,
`load_best_model_at_end=True`. Report final numbers with bootstrap CIs clustered by
meeting/speaker (Bisani & Ney 2004; blockwise bootstrap, Liu 2020).

## Sources (the general ML question)

Cross-checked with live web search (Grok, sourced) and a Codex second opinion; both
agreed whole-domain hold-out for val is sound when a separate temporal test already
measures unseen-city generalization, and that a tiny single-domain val is a noisy
early-stopping signal — which is exactly what the speaker subset + cap is meant to
steady.

- speaker-exclusive as the ASR standard: HF audio-dataset best practices
  (https://discuss.huggingface.co/t/best-practices-to-create-an-audio-dataset/174312),
  arXiv 2605.31469 (https://arxiv.org/html/2605.31469v1)
- whole-group vs stratified hold-out (LOSO / GroupKFold vs leave-one-domain-out):
  scikit-learn cross-validation (https://scikit-learn.org/stable/modules/cross_validation.html),
  LOSO-CV (https://pmc.ncbi.nlm.nih.gov/articles/PMC8671136/)
- temporal split benefits and pitfalls: kumo.ai temporal split
  (https://kumo.ai/pyg/concepts/temporal-split/)

## Follow-ups

- [ ] Add the seeded speaker-subset val builder to the split program (≥3-min floor,
  12 speakers/city, 60-min/city budget, 5-min target / 8-min hard cap per speaker),
  emitting a frozen speaker list. Primary metric = val_reg WER.
- [ ] Keep the full held-out pool for the periodic / final measurement with seeds + CIs.
- [ ] Close the open question in `docs/decisions/data.md` (whole-city val → speaker
  subset of held-out cities, plus the leakage-check result).
