# Handoff: two-screen training runs, mid-flight (2026-08-16)

For the agent continuing `exp-2026-08-14-external-packs`. Written at user request.
Read `CLAUDE.md`, then the ledger record, then this. The prereg that governs
interpretation: `docs/specs/2026-08-15-external-packs-screens-prereg.md` (user
simplification: TWO single-seed screens) with data/gates context in
`docs/specs/2026-08-15-targeted-deletion-training-prereg.md`.

## State right now

**RUN 1 (in-domain targeted-deletion mix) — TRAINING DONE, EVAL PENDING.**
- Pod `t6ugwl9f4efu23` is KILLED (adapter safely downloaded first).
- Artifacts: `~/.cache/oc-public/train-screens-2026-08/run1-artifacts/`
  (adapter/, train.log, build.log, attestation). Adapter sha256
  `2202f426a7ecdd1e…`. Trainer internal finals: train_loss 0.528;
  val_corr eval_wer_norm 19.84; val_reg eval_wer_norm 2.68 (these are
  training-meeting slices — NOT comparable to the frozen-window numbers).
- Realized exposure: 42,204 presentations, shares 68.9/18.6/2.6/10.0
  (backbone/deletion-hard/names/other), MAX_STEPS 10552, seed 101.

**RUN 2 (externals stage-1 → identical stage-2) — STAGE-1 RUNNING.**
- Pod `hwydnokhc60y2f`, RTX A5000 secure $0.27/h, SSH `root@69.30.85.32 -p 22027`
  (StrictHostKeyChecking off). Stage-1: 5,190 steps, was ~48% at 21:15 UTC
  2026-08-15, ~3.1 s/step → ends roughly 01:30 UTC 2026-08-16.
- **WATCHDOG DEADLINE RISK:** `scripts/pod_hard_deadline.sh hwydnokhc60y2f 32
  run2-screen` (local PID: `pgrep -f pod_hard_deadline`) kills the pod ~19:26 UTC
  2026-08-16. Stage-2 needs ~14-16h on the A5000 after stage-1 → the deadline is
  TIGHT. When stage-2 is confirmed healthy (loss decreasing, no crash), kill the
  old watchdog and re-arm with a larger deadline (e.g. 46h total ≈ $12.4, still
  under the $22 ceiling with RUN1's ~$6.2 actual) and record the re-arm in the
  ledger. NEVER leave the pod without a watchdog.
- Everything needed is already on the pod: `/workspace/{train_runpod.py
  (48kHz-resample fix included), screen_arms.py, stage1_gate.py,
  stage1-pack-manifest.jsonl, stage2-presentations-seed101.jsonl,
  run2-stage2-data/, training-sets/}`. `evaluate`+`jiwer` installed.

## RUN 2: exact remaining sequence (from the launch plan, Codex-hardened)

1. Wait for stage-1 to finish (`/workspace/stage1.log`, `pgrep -f train_runpod`).
2. Stage-2 clip build:
   `SMOKE=0 SEED=101 DATA_DIR=/workspace/run2-stage2-data BUILD_AND_EXIT=1 WORK_DIR=/workspace/stage2 python /workspace/train_runpod.py`
   (~1-3h; tolerated per-meeting "audio fail" lines are OK, incl. one known 403).
3. Catastrophic gate (MANDATORY before stage-2 training):
   `python /workspace/stage1_gate.py --adapter /workspace/stage1/adapter --work /workspace/stage2`
   — explicit `STAGE1 GATE: PASS/FAIL`; on FAIL: stop, download stage-1 adapter +
   logs, kill pod, record in ledger. Do not proceed.
4. `python /workspace/screen_arms.py emit --work /workspace/stage2 --presentations /workspace/stage2-presentations-seed101.jsonl --out /workspace/stage2/train_manifest.json`
   → use its printed 2-epoch MAX_STEPS (expect ~10,552 minus attrition).
5. Stage-2 training:
   `SMOKE=0 SEED=101 DATA_DIR=/workspace/run2-stage2-data WORK_DIR=/workspace/stage2 INIT_ADAPTER=/workspace/stage1/adapter TRAIN_MANIFEST=/workspace/stage2/train_manifest.json MAX_STEPS=<emit> python /workspace/train_runpod.py`
   (INIT_ADAPTER hardening asserts trained stage-1 weights; a FATAL there is a
   real finding, not noise). Always `nohup … > log 2>&1 < /dev/null &`.
6. Download BOTH adapters (`/workspace/stage1/adapter`, `/workspace/stage2/adapter`)
   + logs + attestations to `~/.cache/oc-public/train-screens-2026-08/run2-artifacts/`,
   THEN `runpodctl remove pod hwydnokhc60y2f` and kill its watchdog. Record cost.

## Evaluation (both runs; local minipc frozen stack ONLY)

Path proven for `artifact-adapter-correction-only`; reuse it:
1. Merge+convert each adapter: `serve/oc-asr/build_model.sh` pattern (PEFT
   `merge_and_unload` → `ct2-transformers-converter --quantization int8_float32`).
   sha256-fingerprint each CT2 dir (recorded) so the two runs provably decode
   different weights.
2. Decode + score the 39 frozen windows via the `notebooks/correction_only_score.py`
   path (`SC=~/.cache/oc-public .venv-eval/bin/python … decode|score` with
   ARM_MODEL pointed at the new CT2 dir). CPU int8, 16 threads. NEVER compare to
   GPU-stack numbers.
3. DS-WER v2: `scripts/ds_wer_local.py <decode.json>` — primary = v2 **entities**
   cut (terms `research/ds_wer/terms/{argos,orestiada}.v2.json`); report v1 too.
4. Reference numbers (this local stack): control `artifact-adapter-fixed` WER
   0.1589 / DS-WER v1 0.512 / v2-entities 0.5365. Deletion/insertion/substitution
   decomposition comes from the score step.

## Pre-declared decision tree (frozen before any number; do not re-derive)

- RUN2 vs RUN1 winner only if paired per-window bootstrap CI of the delta
  excludes zero; else BOTH advance. Conclusions worded "RUN2 recipe vs RUN1"
  (dose+data confound, GPU-type difference A40-vs-A5000 is a recorded screen
  caveat).
- vs control: primary = deletion-rate drop without insertion/WER regression.
  Both worse on deletions → no blind retries; error analysis first.
  Deletions better but WER >+1pt → ONE rebalance screen (more backbone), max.
  WER >+2pts vs control → suspect pipeline bug (fingerprints/manifests) before
  interpreting.
- Winner (or both, on tie) → 3-seed confirmation with the FROZEN gates in
  `2026-08-15-targeted-deletion-training-prereg.md`, matched GPUs, paired seeds.
  Single-seed screen results decide nothing by themselves (2.1-pt/seed spread).
- DS-WER never decides alone (274 entity occurrences; wide CIs).

## Finish protocol reminders

Update the `exp-2026-08-14-external-packs` ledger record in the same change as
results; run `python3 scripts/check-research-state.py`; report actual $ spent vs
the $22 ceiling; screens are labelled screens everywhere they are quoted.
The 7 temporal holdout windows stay sealed. Transcript text/audio never in git.

## Open loose ends (not blocking, in priority order)

1. `gap4`/`gap5` review queues on the VM are stale after the wave — harmless.
2. BOUNDARY tier (4,043 rows) recoverable via boundary-shift for a future data
   round; MIDDLE (985) untouched.
3. CodeRabbit flagged (in passing): ledger cost estimate wording for the A5000,
   EuroSpeech country-level licence note, and a suggested re-audit of the
   35,427-row backbone actually in the manifest (the 250-row audit sampled the
   pre-leakage-fix composition; direction of bias is conservative). Judgement:
   optional.
4. Monitors in the old session die with it; re-arm your own (SSH tail pattern in
   this repo's session or simply poll).
