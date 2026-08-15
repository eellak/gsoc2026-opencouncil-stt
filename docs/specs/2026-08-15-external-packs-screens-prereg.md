# Preregistration — external-pack stage-1 screens (STOMA / CV / EuroSpeech)

2026-08-15. Status: **frozen design, not started.** No GPU may be created until the
open items in §6 are closed. Ledger record: `exp-2026-08-14-external-packs`.

> **User simplification, 2026-08-15 (supersedes §3's five arms): TWO trainings.**
> Arm A: stage-2 only, on the in-domain set *including the newly verified gap
> corrections*. Arm B: stage-1 on the **combined balanced** external packs
> (stoma-v2 + cv-v2 + eurospeech-v2), then the identical stage-2. Same seed, same
> stage-2 data/steps/decode in both. What this buys: one clean answer to "do the
> externals help on top of our best corrections set", at two runs instead of five.
> What it gives up, recorded so nobody re-narrates it later: (a) no per-pack
> attribution — if B wins or loses we cannot say which source did it; (b) the new
> gap corrections have no control of their own — this pair cannot say what the gap
> rows contributed relative to the old stage-2. Gates, dose rule, promotion
> honesty (single seed = screen) and §6 checklist stay exactly as below.

Builds on [`2026-08-14-hparl-stage1-prereg.md`](2026-08-14-hparl-stage1-prereg.md);
where the two disagree, this file governs the external-pack screens.

## 1. Question

Which, if any, of the external packs — `stoma` (~23 h studio read speech), `cv`
(~16 h validated read prompts), `eurospeech` (~30 h parliamentary, preselected +
Soniox-verified), or a combined balanced mix — improves the frozen two-stage recipe
when used as stage-1 adaptation, measured as agreement-WER on the 39 validation
windows, without a deletion-rate regression?

## 2. Prior expectation (stated before any run)

Small or zero. `exp-2026-08-11-wer-levers-research`: ~1300 h buys ~0.5 points; the
dominant residual error is homophone orthography. The plausible gain is robustness.
The specific risk is a **deletion regression** from short-utterance packs teaching
early EOS (median clip: STOMA ~4 s, CV ~4 s; EuroSpeech ~15 s). This experiment can
lose, and the screens exist to lose cheaply.

## 3. Arms (screen phase, ONE shared seed)

| arm | stage 1 | stage 2 |
|---|---|---|
| A baseline | — | frozen recipe on in-domain corrections |
| B stoma | LoRA on `stoma-v1` | same, identical data/steps/LR |
| C cv | LoRA on `cv-v1` | same |
| D eurospeech | LoRA on `eurospeech-v1` | same |
| E combined | LoRA on balanced mix of the three (equal sampling probability per source, not raw concatenation) | same |

- **Dose equalization**: identical stage-1 optimizer updates across B–E (capped by
  the smallest pack; larger packs subsample with the run seed). Otherwise pack
  identity is confounded with dose.
- Stage boundary: stage 2 **continues the same LoRA adapter** (no merge, no
  re-init); optimizer and scheduler are re-initialized identically in all arms —
  including A, whose "stage 1" is zero updates.
- Same base model, stage-2 data, decode config as
  `exp-2026-08-13-targeted-deletion-training`'s frozen recipe.

## 4. Frozen before any number

- Decode config: copied verbatim from the corrected-adapter benchmark runs.
- Substrate: 2026-08 freeze, 39 validation windows; the 7 temporal holdouts stay
  sealed. Normalizer unchanged.
- Stage-1 catastrophic check (before any stage 2 spends GPU time): the stage-1
  checkpoint must produce non-empty transcriptions on a 5-window smoke set drawn
  from **training** meetings (never the 39). Empty output or EOS collapse → the arm
  stops there.

## 5. Metrics and decision rules

Screens are labelled **screens** everywhere they are quoted — measured per-seed
spread is 2.1 WER points; one seed decides nothing.

- Screen promotion rule (decided now): promote an arm to confirmation only if, on
  the shared screen seed, its WER delta vs arm A is **< 0** AND its deletion-rate
  delta is **≤ 0**. Promote at most **two** arms. No re-runs with new mixes if all
  fail — that outcome closes the record with a negative result.
- Confirmation: 3 fresh seeds, arm A re-run per seed (paired deltas). Gates on the
  seed-mean deltas, unchanged from the HParl prereg: WER < 0 with CI excluding
  zero; deletions ≤ 0 (hard fail); insertions and substitutions each < +0.0005.
- Always reported: single-item domination check, DS-WER on domain terms, per-arm
  deletion/insertion/substitution decomposition.

## 6. Open before any pod is created

- [ ] Packs built from the finished full passes and registered in the ledger with
      `train_jsonl_sha256` (`stoma-v1`, `cv-v1`, `eurospeech-v1`).
- [ ] EuroSpeech↔hparl2 overlap check (normalized-text hash intersection) recorded.
- [ ] Licences recorded in each `meta.json`. Research use proceeds; **commercial
      shipping stays blocked** for eurospeech (and hparl2) until their legal calls
      are made. STOMA (CC-BY-4.0) and CV (CC0) are clean.
- [ ] Pod plan per `docs/runbooks/runpod-training-pod.md`: cheapest GPU that fits
      whisper-large-v3 LoRA (≥24 GB VRAM), watchdog with hard deadline armed
      **before** upload, pod ID recorded in the ledger record.
- [ ] Cost ceiling written into the ledger record before creation; the screen
      phase (5 arms × stage-1 micro + stage-2) must fit it.

## 7. Stop conditions

Anything failing twice the same way → stop, kill the pod, write down what broke.
Stage-1 loss divergence or the catastrophic check failing → that arm stops before
stage 2.
