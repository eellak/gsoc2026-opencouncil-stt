# Mini-PC auto-research: what improves Whisper fine-tuning (Track 2)

Status: **spec for an AI agent to implement on the mini PC** (2026-06-23). Companion
to [finetuning-dryrun-plan.md](finetuning-dryrun-plan.md) (Track 1 = the Kaggle
large-v3 run). Reviewed against the Codex notes in that doc.

## Goal

The mini PC has **no GPU**, so it can't train large-v3 — but it CAN run a **tiny
Whisper** (`base` ~74M, or `small` ~244M) on CPU fast enough for an **automated
experiment loop**. The point is **not a deployable model**; it's to find *which
data/training choices move validation WER* so those findings transfer to the
large-v3 Kaggle run. (Codex: **data-mixture findings transfer; model-rank sweeps
do not** — so sweep data, not architecture.)

Mirror the existing prompt loop `eval/improve_loop.py`: configs → run → score →
leaderboard → report.

## Data (reuse Track 1's source — build once, cache)

Same source as the notebook, built once on the box:
1. Fetch included corrections: `GET https://79-76-114-184.sslip.io/api/export`
   (1,905 utts; each has `audio_url,start,end,final_after_text,city_id,meeting_id,
   error_categories`).
2. Build the **no-edit backbone** + the **regression val** from the meeting JSON
   (`/api/cities/{city}/meetings/{meeting}` → utterances with `lastModifiedBy`):
   - no-edit utts from **`humanReview=true`** meetings (use `data/eval/meeting_meta.json`).
3. Audio: download each meeting mp3 **once**, **decode to 16 kHz mono PCM once**,
   then slice per utterance (Codex: no repeated compressed seeks). Cache clips to
   disk (`data/asr/clips/`), keyed by utterance_id, so every experiment reuses them.
4. **Frozen manifest** (`data/asr/manifest.jsonl`): one row per clip with
   `utterance_id, split, city, meeting, start, end, audio_sha, text, source(tier),
   error_categories, dur`. Version it; record source hashes. Splits:
   - `train` = the 8 cities (corrections + optional no-edit backbone).
   - `val_corr` = orestiada+argos corrections.
   - `val_reg` = no-edit utts from orestiada+argos reviewed meetings (regression).

## Split (fixed, same as Track 1)

- **TEST = future data** (not here). **VAL = orestiada + argos** (whole, disjoint).
- Never split utterances randomly; group by meeting (and speaker where known).
- Also support **meeting-level k-fold (3–5)** across all cities for a
  lower-variance read (Codex), as an optional eval mode.

## The sweep (highest-signal, transferable dims — Codex)

Each experiment = one config dict. Loop over these axes (start one-at-a-time, then
combine the winners):

1. **Data composition:** `corrections_only` vs `corrections + no_edit_backbone`
   (and the backbone ratio: 0 / 1× / 3× the corrections volume).
2. **Sampling / weighting:** `uniform` vs `error_category_balanced` vs
   `capped_oversample` (cap rare categories so they don't dominate).
3. **Error-type focus:** restrict/oversample by `error_categories`
   (named_entity / homophone / number_date / …) — does focusing on acoustic
   categories help more than style ones?
4. **Learning rate / effective batch** (lr ∈ {1e-4, 5e-5}; grad-accum for batch).
5. **Segment-quality filters:** duration window, speech-ratio / silence filter,
   timestamp-padding, text↔audio duration-mismatch drop.

(Do **not** sweep LoRA rank / model size as a research goal — low transfer.)

## Loop mechanics

```
for cfg in configs:
    ds = build_split(manifest, cfg)          # apply composition/sampling/filters
    model = whisper-base (fresh each run)     # tiny, CPU
    finetune(model, ds.train, steps=cfg.steps, lr=cfg.lr)   # cap steps for CPU
    scores = eval(model, ds.val_corr, ds.val_reg)           # WER raw+norm, CER, per-cat
    leaderboard.append({cfg, scores, delta_vs_baseline})
report()   # rank by val_corr WER gain WITHOUT val_reg regression
```

- **Baseline first:** no-finetune tiny-model WER on val (the reference every run
  compares to), with **identical decoding params** across all runs.
- **Selection rule:** a config "wins" only if it improves `val_corr` WER **without
  regressing `val_reg`** (ordinary speech) beyond a small tolerance — that's the
  whole point of the regression set.
- **Seeds:** ≥3 per finalist config; report mean/range, not one best run.
- **Budget:** tiny model + few hundred steps per run; cap total runs; `log()` what
  was skipped. Use the on-box `claude`/`codex` only if a config-proposer step is
  wanted (optional; the fixed grid above is enough to start).

## Outputs

- `data/asr/manifest.jsonl` (frozen, versioned) + dataset stats.
- `data/reports/finetune-research/leaderboard.jsonl` + `report.md` (ranked configs,
  per-category deltas, regression check, CIs by meeting).
- A short "what transfers to large-v3" summary → feeds the Kaggle run's config.

## Metrics (same as Track 1)

Raw WER + Greek-normalized WER + CER + S/D/I + per-category; bootstrap CIs grouped
by meeting (utterances within a meeting are not independent). Report `val_reg`
separately as the regression guard.

## Reproducibility

Frozen manifest with row IDs + audio hashes + split; record code commit, package
versions, model revision, seed, dataset hash per run; log audio failures instead
of silently dropping rows.

## Suggested phases for the agent

- **P0:** env (venv, torch CPU, transformers/datasets/peft/evaluate/jiwer/faster-
  whisper/librosa); verify ffmpeg.
- **P1:** build + freeze the manifest and the clip cache; audit a stratified
  sample of clips by ear/length before trusting them.
- **P2:** baseline tiny-model WER on val.
- **P3:** run the one-axis sweeps → leaderboard.
- **P4:** combine winners, ≥3 seeds, write `report.md` + the "transfers to large-v3"
  summary.

## Results — first dry run (2026-06-23)

Implemented in `eval/autoresearch/{prepare_asr,experiment,loop,round2}.py`; full
write-up in [`data/reports/finetune-research/report.md`](../../data/reports/finetune-research/report.md).
16 fine-tune runs, ~1 h wall-clock, whisper-base CPU. Headline:

- Fine-tuning on ~17 min of corrections drops ordinary-speech WER reliably
  (`val_reg` 0.674 → ~0.43, ~24 pts on every run) — domain adaptation works and
  does **not** degrade general ASR (the correction-bias worry didn't materialise).
- The hard residual cases (`val_corr`) are noise-limited: cross-config spread
  (0.064) ≈ single-config seed spread (0.062) on 56 val clips → **no data-mixture
  axis is statistically separable yet.** Composition (corrections + no-edit
  backbone) is the transferable lever; lr/steps are not.
- **Blocker for the next round:** enlarge `val_corr` (full ~9.9k orestiada+argos
  pool, not 56) + bootstrap CIs before trusting any mixture ranking.
