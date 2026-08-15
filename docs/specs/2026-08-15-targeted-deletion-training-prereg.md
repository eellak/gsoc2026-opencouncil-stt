# Preregistration: targeted deletion training (exp-2026-08-13-targeted-deletion-training)

Date frozen: 2026-08-15, v2 after Codex review (job a23a41d8, 37 findings; the
material ones are folded in below). Nothing below changes after the first pod
starts; any deviation is recorded in the ledger as a protocol violation before
results are read.

## Question

Does a deletion-targeted training mix lower the deletion rate on the frozen
validation windows without raising insertions, substitutions, or overall WER,
relative to a neutral control trained on the same expanded pool?

## Data (frozen row lists)

All row lists are frozen on disk before training; sha256 of each list goes into
the run manifest. Sampling unit everywhere: **row**, with realized audio-hours
and target-token totals reported per arm (exposure attestation, below).

- **Backbone (42.4h target, 10 cities)** — no_edit rows from trusted meetings
  passing the unified quality filter (duration ≥ 2.0 s AND not
  speaker-change-adjacent AND ≥ 2.0 tokens/s, gap2 normalization):
  `existing_keep.parquet` (12,442 rows / 12.96h) + a per-city seeded random
  subset of `headroom_keep.parquet` up to +5h/city (seed 20260816; caps bind
  athens, chania). Audit basis: 300-row Soniox audit — unfiltered 80.7% pass,
  filtered stratum 48/50 = 96% (n=50; Wilson ~87–99%).
- **FINAL BACKBONE AUDIT GATE (before training):** a 250-row Soniox audit of
  the composed backbone, risk-stratified (oversampling rows near the filter
  thresholds: duration 2–3s, 2.0–2.5 tok/s, and the two capped cities). Gate:
  ≥ 90% pass (same two-sided token gate as the 300-row audit). If it fails,
  tighten the filter, re-compose, re-audit; training does not start until the
  gate passes. Results file: `backbone-final-audit-*.json`.
- **Deletion-hard (~2,950 rows, ~4.6h)** — audio-verified restored-speech set:
  user manual includes + 2,643 auto-accepted rows. Contamination is reported
  **stratified by rule** (rule 1: found_frac ≥ 0.85 ∧ n_added ≥ 5, audited
  97.4%; rule 2: found_frac ≥ 0.85 ∧ n_added 2–4, audited 95.7%). Labels use
  corrected text and the user's adjusted boundaries where present; rows with
  invalid intervals (end ≤ start, duration > 30s, empty target) are dropped at
  manifest build and counted.
- **Names (10%)** and **other (10%)** — from the existing reviewed correction
  pool.
- **Dedup / leakage:** buckets are disjoint by utterance_id AND by
  (meeting_id, overlapping time span). No row from a validation city, a
  validation meeting, or the 135 mined validation-split rows appears in any
  bucket. Speaker overlap with validation is allowed (unavoidable) and
  reported.

## Arms

- **A (control):** uniform row sampling over the union of all four buckets
  (realized bucket shares = pool proportions; reported).
- **B (treatment):** row sampling probabilities 55/25/10/10
  (backbone/deletion-hard/names/other). An epoch = one pass of the sampling
  budget N_rows = |union pool|. Deletion-hard presentations are capped at 2 per
  row per epoch; if the cap binds before 25% is reached, the shortfall goes to
  backbone and the realized shares are reported (the 25% is a target, the cap
  wins).
- Per-seed sample manifests are generated **before** training (seeds 101, 102,
  103 for both arms — paired: seed k uses the same RNG stream for ordering and
  augmentation in A and B) and hashed into the run manifest.
- 3 seeds per arm. Matched across arms: optimizer, LR schedule, update count,
  effective batch size, gradient accumulation, padding policy, precision,
  container image digest, GPU model. Arm-to-pod assignment: A and B for the
  same seed run on the same GPU type; ordering alternates.
- Plain CE loss. Base: whisper-large-v3 + LoRA, config copied numerically from
  `artifact-adapter-fixed` (rank/alpha/dropout/targets recorded in the manifest
  together with the artifact's content hash, not by name alone).
- Checkpoint: the **final training step** is the evaluated checkpoint. No
  checkpoint selection on validation.

## Evaluation

Substrate: the 39 validation windows (31 meetings) of the 2026-08 freeze
(`research/eval-freeze-2026-08/manifest.json`), frozen decode stack, frozen
normalization. The 7 temporal holdout windows stay sealed unless all gates
pass; if unsealed they are confirmatory only (same metrics, same margins, no
re-tuning, no seed selection after unsealing).

Analysis (frozen): for each seed pair k, compute the per-meeting D/I/S/WER
deltas (B_k − A_k, micro-averaged over the meeting's reference tokens, same
alignment for all four metrics). Resample **meetings** (n=31, wild cluster
bootstrap, 10,000 replicates, RNG seed 20260817, percentile-t CIs) within each
seed pair; combine seed pairs by their mean delta with seed-pair variance
included (hierarchical). All margins are **absolute proportions of reference
tokens**.

Gates — one-sided 95% noninferiority upper bounds, intersection-union (B ships
only if ALL hold; individual passes are not separate findings):

1. Deletion-rate delta: UB < 0 (superiority).
2. WER delta: UB < +0.001.
3. Insertion-rate delta: UB < +0.0005.
4. Substitution-rate delta: UB < +0.0005.
5. Leave-one-**meeting**-out: gate 1 must hold in all 31 LOMO reruns.
6. Seed consistency: no B seed shows a point-estimate deletion regression
   > +0.0005 vs its paired A seed.
7. Hallucination guard (frozen guard set, manifest hashed): counts of (a)
   5-gram repeated ≥ 3× within a window, (b) output > 1.5× reference token
   length, (c) non-Greek script runs ≥ 10 chars. Gate: no metric more than
   doubles vs the pooled A arms.

Shipping artifact: the B seed with the **median** deletion delta (decided by
rule, not by inspection); it must itself satisfy gates 2–4 point-wise.

## Run validity

- A seed run is invalid on: NaN/Inf loss, > 0.5% dropped batches, trainable
  parameter count mismatch, incomplete decode. Invalid runs are rerun with the
  same seed; a seed abandoned after 2 failures invalidates the whole
  experiment round (per the stop-on-second-failure rule).
- Exposure attestation after training: realized bucket shares, unique rows,
  presentation-count distribution, audio hours, target tokens per arm·seed;
  tolerance ±2 percentage points on bucket shares, else the run is invalid.
- Outcome wording: if gates fail, the experiment closes **"did not meet
  shipping criteria"** with the failure mode recorded (efficacy / guardrail /
  invalid run / insufficient precision) — not "no benefit". Any
  contamination-related rerun (excluding `username=auto-verifier` rows) is a
  separate, secondary, pre-declared screen and cannot rescue the primary
  result.

## Pods

RunPod. Watchdog with hard deadline armed **before** any upload; pod ID in the
ledger at creation; same-failure-twice → stop and write down what broke.
