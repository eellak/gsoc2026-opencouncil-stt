# Spec: Whisper-large-v3 hyperparameter sweep (self-contained Kaggle notebook)

Status: approved design — 2026-06-26
Owner: Άγγελος
Artifact: `notebooks/whisper_sweep_kaggle.ipynb` (new; does not modify `whisper_finetune_kaggle.ipynb`)

## Goal

Find good training hyperparameters for the real large-v3 LoRA fine-tune by running a
small, self-contained sweep on Kaggle GPU. Output a leaderboard the human (or an LLM
on the Mini PC, later) reads to pick/reject configs. No LLM and no API key are involved
in the run itself — Kaggle only trains and emits metrics.

This is a **test / first pass**, not the definitive tuning run. Use the leaderboard to
*reject* clearly-bad configs, not to crown a winner on a 0.1 WER difference.

## Why a sweep, not an LLM-agent loop

The user's "auto research" goal is hyperparameter tuning over a defined space, not
open-ended code editing. A structured sweep is reproducible, needs no API key, and
fits the existing notebook's building blocks. An LLM layer (reads the leaderboard,
proposes the next batch) is an optional future addition — explicitly out of scope here.

## Why Kaggle, self-contained, single notebook

- Mini PC is an AMD APU (no real CUDA GPU) → large-v3 LoRA would crawl there. Kaggle
  is the GPU executor, as in the existing notebook.
- One notebook that runs the whole sweep in a single session (≤12h, safe ~9h budget)
  and writes one leaderboard CSV. No Kaggle-API orchestration, no token, no polling.
  The notebook is written so a future Mini-PC `kaggle kernels push/output` wrapper can
  drive it unchanged, but that path is out of scope for this spec.

## Decisions (locked)

| Axis | Value |
|---|---|
| Method | Hyperparameter sweep, no LLM in the loop |
| Orchestration | Single self-contained Kaggle notebook → leaderboard CSV |
| Search space | LR × LoRA-rank grid; epochs measured per-epoch (free axis) |
| Data recipe | Fixed: corrections (training cities) + 1× no-edit backbone |
| Budget | 9 configs, single seed, ≤12h Kaggle session (per-epoch flush + wall-clock guard) |
| Base model | Sweep on `openai/whisper-large-v3-turbo` (fast decoder); apply the chosen LR/rank to the `large-v3` final run. LoRA on `q_proj`/`v_proj`, encoder frozen |

### Search space (concrete)

- LR ∈ {5e-5, 1e-4, 2e-4}
- LoRA rank ∈ {8, 16, 32}, alpha = 2 × rank, dropout = 0.05
- 3 × 3 = **9 configs**
- epochs: train each config to max **3**, eval after every epoch (epochs is not a
  separate run — it is read off the per-epoch curve; smoke showed epoch 4 overfitting)
- seed: single fixed seed

## Architecture — three phases, one notebook

### Phase 0 — one-time setup (the expensive part; never repeated per config)

1. **Data build** reuses the existing notebook's memory-safe pipeline verbatim:
   - decode exactly one meeting mp3 at a time, free PCM + delete file before the next
     (an earlier all-in-RAM version OOM'd);
   - smoke speed is dominated by the number of **distinct meetings** whose full mp3 is
     downloaded/decoded — cap meetings, not just clips;
   - signature-based cache guard in `/kaggle/working` so a kernel restart skips the
     ~70-min rebuild; rebuild only when the signature changes;
   - denylist filter for unreviewed (<5% human-edit) meetings as defense in depth.
2. **Data recipe fixed**: corrections from training cities + 1× no-edit backbone.
   Val sets fixed: `val_corr` + `val_reg` (Orestiada/Argos), **subsampled to ~70 clips
   each** so per-epoch eval stays bounded.
3. **Feature extraction once**: log-mel `input_features` + tokenized `labels` do not
   depend on the config → build `ds_train` / `ds_valc` / `ds_valr` once (Arrow on disk)
   and reuse for all 9 configs. This is the main speedup.

### Phase 1 — sweep loop (9 configs)

For each `(lr, rank)`:
1. **Reload base fp16 from the local cache** (~20–40 s) for clean isolation — avoids
   adapter-state bleed when rank changes between configs.
2. Fresh `LoraConfig(r=rank, lora_alpha=2*rank, lora_dropout=0.05,
   target_modules=["q_proj","v_proj"])`; freeze encoder; gradient checkpointing;
   `use_cache=False`.
3. Train to 4 epochs with `eval_strategy="epoch"`; Trainer logs per-epoch
   `wer / wer_norm / cer` on `val_corr` in `state.log_history`.
4. **Per-epoch eval on `val_reg`** via a `TrainerCallback` (`on_epoch_end` →
   `trainer.evaluate(ds_valr)`), recording the regression-guard metric per epoch.
5. Free model + `torch.cuda.empty_cache()` before the next config.

### Phase 2 — leaderboard output

- One row per (config × epoch):
  `lr, rank, alpha, epoch, val_corr_wer_norm, val_reg_wer, val_corr_cer, train_loss, wall_s`.
- Also record the baseline (epoch 0, pre-train) `val_corr` / `val_reg` once, so the
  regression delta is meaningful.
- Write `leaderboard.csv` + `leaderboard.md` to `/kaggle/working` (downloadable as
  notebook output), sorted by `val_corr_wer_norm` ascending, with a regression column
  (Δ `val_reg` vs baseline).
- "Best" is the lowest `val_corr_wer_norm` row **whose `val_reg` does not regress beyond
  a threshold** (default: Δ ≤ +1.0 WER absolute vs baseline; tunable).

## Lessons carried verbatim from `whisper_finetune_kaggle.ipynb`

These are non-negotiable — they were each a bug that was fixed once:

- Decode with `clean_up_tokenization_spaces=False`; collapse whitespace identically on
  predictions and references (the default `True` corrupts BPE-tokenizer text and
  distorts WER/CER on punctuation/casing).
- Collator casts `input_features` to **float16** to match the fp16 model, or
  `generate()` crashes in encoder conv1 ("Input type (float) and bias type (Half)").
- Strip the leading BOS from labels when every row starts with it.
- Greek-normalized WER: NFD → strip combining marks → `ς`→`σ` → strip punctuation.
- `pip uninstall -y torchao` (Kaggle ships 0.10.0, which PEFT's LoRA dispatcher rejects).
- One mp3 in RAM at a time; cap meetings; signature cache guard.
- `val_reg` regression is the metric that matters most — guard on it explicitly.

## Honest caveats

1. **Eval dominates wall-time** (Whisper `generate()` is slow): per-epoch eval × 2 sets
   × 9 configs. Mitigated by subsampling val to ~70 clips. If the run is too slow, cut
   val sets or eval `val_reg` only at the best `val_corr` epoch.
2. **Single seed + small sample**: prior CPU research showed seed-variance (≈0.062)
   exceeded config-ranking differences (≈0.064). The leaderboard is **indicative, not
   definitive**. Multiple seeds belong to the later, larger run.
3. **Kaggle quota** ~30h GPU/week, 12h/session, ~2 concurrent GPU sessions.
4. **12h hard-kill is real (learned the hard way)**: a `large-v3` run hit ~2.8h/config →
   9 configs ≈ 25h, killed at 12h (exit 137) and — because the leaderboard was written
   only at the end — produced **zero output**. Two fixes: (a) the sweep now runs on
   `large-v3-turbo` (fast decoder) so the full grid fits; (b) the loop writes
   `leaderboard.csv/.md` **after every epoch** and stops ~40min before the limit
   (`MAX_RUNTIME_S`, anchored at `KERNEL_T0` before the data build), so a timeout always
   leaves the finished configs on disk.
5. **Turbo-vs-large-v3 transfer**: hyperparameters are tuned on `large-v3-turbo` as a
   fast proxy. LR/rank usually transfer, but confirm the winner on `large-v3` in the
   final run rather than assuming identical optima.

## Out of scope (this spec)

- LLM-driven config selection / karpathy-style agent editing code.
- Mini-PC `kaggle kernels push/output` orchestration (notebook is written to allow it
  later, but the wrapper is not built here).
- Data-recipe axes (reviewed vs random corrections, composition, scale) — deferred;
  this run holds the recipe fixed.
- Multi-seed variance estimation and bootstrap CIs — later, larger run.

## Success criteria

- Notebook runs end-to-end on Kaggle GPU in one session, ≤ ~4h, no OOM.
- Produces `leaderboard.csv` + `leaderboard.md` with 9 configs × per-epoch rows and a
  baseline row, plus a regression-guarded "best" pick.
- Reuses the existing data-build and metric code without reintroducing any of the
  fixed bugs listed above.

## How to run (Kaggle)

1. Upload `notebooks/whisper_sweep_kaggle.ipynb` to a new Kaggle notebook.
2. Accelerator: **GPU T4 x2**. Internet: **On**.
3. Smoke first: leave `SMOKE = True`, **Save & Run All (Commit)**. Confirm it finishes,
   `leaderboard.csv` appears in Output, and the table has rows. This proves the harness
   end-to-end before spending the GPU budget.
4. Real run: set `SMOKE = False`, Commit again (~3–4h, one session).
5. Download `leaderboard.csv` / `leaderboard.md` from the notebook Output.

## Reading the leaderboard

- Sort key is `val_corr_wer_norm` (lower better). `reg_delta` is the change in `val_reg`
  WER vs baseline; positive = ordinary speech got worse.
- The "Best" line already excludes configs that regress `val_reg` beyond `MAX_REG_DELTA`.
- Treat differences smaller than the known seed-variance (~a few WER points at this
  scale) as noise — use the board to reject bad LR/rank, not to crown a 0.1 winner.
