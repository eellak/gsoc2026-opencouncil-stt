# Whisper-large-v3 Hyperparameter Sweep — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Kaggle notebook that sweeps 9 training configs (3 LR × 3 LoRA-rank) for large-v3 LoRA fine-tuning and emits a regression-guarded leaderboard.

**Architecture:** Pure, testable helpers (grid, val subsample, leaderboard assembly, best-pick) live in `eval/sweep/sweep_utils.py` with local unit tests. The Kaggle notebook `notebooks/whisper_sweep_kaggle.ipynb` reuses the existing notebook's memory-safe data build + metrics, extracts features once, loops the 9 configs reloading the fp16 base per config, and inlines the verified helpers to produce `leaderboard.csv` / `leaderboard.md`.

**Tech Stack:** Python, PyTorch, HF transformers/peft/datasets, jiwer/evaluate, Kaggle GPU (T4 x2). Tests: pytest.

Spec: `docs/specs/whisper-hyperparam-sweep.md`. Reference notebook (do not modify): `notebooks/whisper_finetune_kaggle.ipynb`.

---

## File Structure

- Create: `eval/sweep/__init__.py` — empty package marker.
- Create: `eval/sweep/sweep_utils.py` — pure helpers: `make_grid`, `subsample`, `build_leaderboard`, `pick_best`, `render_markdown`. No torch/HF imports — must import on a laptop.
- Create: `eval/tests/test_sweep_utils.py` — unit tests for the helpers.
- Create: `notebooks/whisper_sweep_kaggle.ipynb` — the sweep notebook (inlines a copy of the verified helpers).

The split: all logic that can be wrong *and* tested without a GPU lives in `sweep_utils.py`. The notebook is orchestration of GPU code adapted from the reference notebook plus a paste of the verified helpers.

---

## Task 1: Pure helpers — grid + deterministic subsample

**Files:**
- Create: `eval/sweep/__init__.py`
- Create: `eval/sweep/sweep_utils.py`
- Test: `eval/tests/test_sweep_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# eval/tests/test_sweep_utils.py
from eval.sweep.sweep_utils import make_grid, subsample


def test_make_grid_is_full_cartesian_with_stable_ids():
    grid = make_grid(lrs=[5e-5, 1e-4, 2e-4], ranks=[8, 16, 32], seed=13)
    assert len(grid) == 9
    # alpha is always 2 * rank
    assert all(c["alpha"] == 2 * c["rank"] for c in grid)
    # every config carries the seed and a unique, stable id
    ids = [c["config_id"] for c in grid]
    assert len(set(ids)) == 9
    assert all(c["seed"] == 13 for c in grid)
    assert "lr0.0001_r16" in ids


def test_subsample_is_deterministic_and_bounded():
    records = [{"i": i} for i in range(200)]
    a = subsample(records, n=70, seed=13)
    b = subsample(records, n=70, seed=13)
    assert len(a) == 70
    assert a == b                      # deterministic for a fixed seed
    assert a != subsample(records, n=70, seed=99)  # seed actually matters


def test_subsample_returns_all_when_n_none_or_larger():
    records = [{"i": i} for i in range(10)]
    assert subsample(records, n=None, seed=1) == records
    assert subsample(records, n=50, seed=1) == records
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest eval/tests/test_sweep_utils.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.sweep'`

- [ ] **Step 3: Write minimal implementation**

```python
# eval/sweep/__init__.py
# (empty package marker)
```

```python
# eval/sweep/sweep_utils.py
"""Pure helpers for the Whisper hyperparameter sweep.

No torch / transformers imports here on purpose: this module must import and run
on a laptop so the leaderboard logic can be unit-tested without a GPU. The Kaggle
notebook inlines a copy of these functions (Kaggle kernels are self-contained).
"""
import random


def make_grid(lrs, ranks, seed):
    """Full Cartesian product of learning rates x LoRA ranks.

    alpha is fixed at 2 * rank. config_id is stable and human-readable.
    """
    grid = []
    for lr in lrs:
        for rank in ranks:
            grid.append({
                "config_id": f"lr{lr:g}_r{rank}",
                "lr": lr,
                "rank": rank,
                "alpha": 2 * rank,
                "seed": seed,
            })
    return grid


def subsample(records, n, seed):
    """Deterministic subsample of at most n records, preserving original order.

    Returns the full list (new list) when n is None or >= len(records).
    """
    records = list(records)
    if n is None or len(records) <= n:
        return records
    rnd = random.Random(seed)
    idx = sorted(rnd.sample(range(len(records)), n))
    return [records[i] for i in idx]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest eval/tests/test_sweep_utils.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/sweep/__init__.py eval/sweep/sweep_utils.py eval/tests/test_sweep_utils.py
git commit -m "feat(sweep): grid + deterministic val subsample helpers"
```

---

## Task 2: Leaderboard assembly with regression delta

**Files:**
- Modify: `eval/sweep/sweep_utils.py`
- Test: `eval/tests/test_sweep_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# append to eval/tests/test_sweep_utils.py
from eval.sweep.sweep_utils import build_leaderboard


def test_build_leaderboard_sorts_and_adds_regression_delta():
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0}
    rows = [
        {"config_id": "a", "lr": 1e-4, "rank": 16, "alpha": 32, "epoch": 2,
         "val_corr_wer_norm": 26.0, "val_reg_wer": 17.0, "val_corr_cer": 10.0,
         "train_loss": 0.4, "wall_s": 600},
        {"config_id": "b", "lr": 2e-4, "rank": 32, "alpha": 64, "epoch": 4,
         "val_corr_wer_norm": 24.0, "val_reg_wer": 30.0, "val_corr_cer": 9.0,
         "train_loss": 0.2, "wall_s": 900},
    ]
    lb = build_leaderboard(rows, baseline)
    # sorted ascending by val_corr_wer_norm -> 'b' (24.0) first
    assert [r["config_id"] for r in lb] == ["b", "a"]
    # reg_delta = val_reg_wer - baseline val_reg_wer
    assert lb[0]["reg_delta"] == 3.0    # b regressed: 30 - 27
    assert lb[1]["reg_delta"] == -10.0  # a improved: 17 - 27
    # original rows are not mutated
    assert "reg_delta" not in rows[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest eval/tests/test_sweep_utils.py::test_build_leaderboard_sorts_and_adds_regression_delta -v`
Expected: FAIL with `ImportError: cannot import name 'build_leaderboard'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to eval/sweep/sweep_utils.py
def build_leaderboard(rows, baseline):
    """Return a new list of rows sorted by val_corr_wer_norm ascending, each with
    reg_delta = val_reg_wer - baseline['val_reg_wer'] (positive = regression).

    Input rows are not mutated.
    """
    out = []
    for r in rows:
        rr = dict(r)
        rr["reg_delta"] = round(r["val_reg_wer"] - baseline["val_reg_wer"], 3)
        out.append(rr)
    out.sort(key=lambda x: x["val_corr_wer_norm"])
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest eval/tests/test_sweep_utils.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/sweep/sweep_utils.py eval/tests/test_sweep_utils.py
git commit -m "feat(sweep): leaderboard assembly with regression delta"
```

---

## Task 3: Regression-guarded best-pick + markdown render

**Files:**
- Modify: `eval/sweep/sweep_utils.py`
- Test: `eval/tests/test_sweep_utils.py`

- [ ] **Step 1: Write the failing test**

```python
# append to eval/tests/test_sweep_utils.py
from eval.sweep.sweep_utils import pick_best, render_markdown


def _sample_lb():
    baseline = {"val_corr_wer_norm": 33.0, "val_reg_wer": 27.0}
    rows = [
        {"config_id": "b", "lr": 2e-4, "rank": 32, "alpha": 64, "epoch": 4,
         "val_corr_wer_norm": 24.0, "val_reg_wer": 30.0, "val_corr_cer": 9.0,
         "train_loss": 0.2, "wall_s": 900},
        {"config_id": "a", "lr": 1e-4, "rank": 16, "alpha": 32, "epoch": 2,
         "val_corr_wer_norm": 26.0, "val_reg_wer": 17.0, "val_corr_cer": 10.0,
         "train_loss": 0.4, "wall_s": 600},
    ]
    return build_leaderboard(rows, baseline)


def test_pick_best_skips_configs_that_regress_val_reg():
    lb = _sample_lb()
    # 'b' has the lowest val_corr_wer_norm but reg_delta +3.0 > 1.0 -> skipped
    best = pick_best(lb, max_reg_delta=1.0)
    assert best["config_id"] == "a"


def test_pick_best_returns_none_when_all_regress():
    lb = _sample_lb()
    assert pick_best(lb, max_reg_delta=-100.0) is None


def test_render_markdown_contains_table_and_best_line():
    lb = _sample_lb()
    md = render_markdown(lb, pick_best(lb, max_reg_delta=1.0))
    assert "| config_id |" in md
    assert "lr1e-04_r16" not in md   # ids in fixture are 'a'/'b'
    assert "**Best (regression-guarded):** a" in md
    assert "24.0" in md              # values rendered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest eval/tests/test_sweep_utils.py -k "pick_best or render" -v`
Expected: FAIL with `ImportError: cannot import name 'pick_best'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to eval/sweep/sweep_utils.py
def pick_best(sorted_rows, max_reg_delta=1.0):
    """Lowest val_corr_wer_norm row whose reg_delta <= max_reg_delta.

    Returns None if every row regresses val_reg beyond the threshold.
    """
    for r in sorted_rows:
        if r["reg_delta"] <= max_reg_delta:
            return r
    return None


_COLS = ["config_id", "lr", "rank", "alpha", "epoch",
         "val_corr_wer_norm", "val_reg_wer", "reg_delta", "val_corr_cer",
         "train_loss", "wall_s"]


def render_markdown(sorted_rows, best):
    """Render the leaderboard as a Markdown table plus a best-pick line."""
    header = "| " + " | ".join(_COLS) + " |"
    sep = "| " + " | ".join("---" for _ in _COLS) + " |"
    lines = [header, sep]
    for r in sorted_rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in _COLS) + " |")
    best_line = (f"**Best (regression-guarded):** {best['config_id']} "
                 f"(epoch {best['epoch']}, val_corr_wer_norm {best['val_corr_wer_norm']}, "
                 f"reg_delta {best['reg_delta']})"
                 if best else
                 "**Best (regression-guarded):** none — every config regressed val_reg")
    return "\n".join(lines) + "\n\n" + best_line + "\n"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest eval/tests/test_sweep_utils.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add eval/sweep/sweep_utils.py eval/tests/test_sweep_utils.py
git commit -m "feat(sweep): regression-guarded best-pick + markdown render"
```

---

## Task 4: Build the sweep notebook

Build `notebooks/whisper_sweep_kaggle.ipynb` by adapting the reference notebook cell-by-cell. The notebook is authored as a Python script and converted with `jupytext`, or written directly as `nbformat` JSON. Use whichever is already available; the steps below specify content, not tooling.

**Files:**
- Create: `notebooks/whisper_sweep_kaggle.ipynb`
- Reference (read, do not modify): `notebooks/whisper_finetune_kaggle.ipynb`

- [ ] **Step 1: Phase 0 cells — reused verbatim from the reference notebook**

Copy these cells unchanged in behavior (install + torchao uninstall; fetch export + meeting JSON + denylist; memory-safe audio helpers; signature-cached memory-safe build; lazy HF datasets + feature extraction; collator + metrics). Keep every fixed-bug guard:
- `clean_up_tokenization_spaces=False` + identical whitespace collapse on preds/refs,
- collator casts `input_features` to `torch.float16`,
- strip leading BOS from labels,
- Greek-normalized WER (`gnorm`),
- one mp3 in RAM at a time, cap meetings, signature cache guard.

Change only the config cell: replace the single-config block with sweep config —

```python
# Sweep config
EXPORT_URL = "https://79-76-114-184.sslip.io/api/export"
MEETING_API = "https://opencouncil.gr/api/cities/{city}/meetings/{meeting}"
MODEL_ID   = "openai/whisper-large-v3"
LANGUAGE, TASK = "greek", "transcribe"
VAL_CITIES = {"orestiada", "argos"}
SMOKE      = True            # True = tiny sanity sweep; False = full ~3-4h Standard run
SEED       = 13
# Search space
LRS   = [5e-5, 1e-4, 2e-4]
RANKS = [8, 16, 32]
EPOCHS = 4
TRAIN_BS, GRAD_ACC, EVAL_BS = 2, 4, 4
LORA_DROPOUT = 0.05
VAL_SUBSAMPLE = 70           # clips per val set kept for eval (keeps per-epoch eval bounded)
MAX_REG_DELTA = 1.0          # best-pick guard: skip configs whose val_reg regresses > this
# Data-build sizing (mirror reference notebook; SMOKE caps distinct meetings)
SAMPLE_N = 60 if SMOKE else None
SMOKE_TRAIN_MEETINGS = 4 if SMOKE else None
SMOKE_VAL_MEETINGS   = 2 if SMOKE else None
VAL_REG_PER_MEETING = 8
SR = 16000; PAD_S = 0.2; MIN_DUR, MAX_DUR = 0.3, 30.0
OUT_DIR = "/kaggle/working/whisper-sweep"
import os, random, numpy as np
os.makedirs(OUT_DIR, exist_ok=True); random.seed(SEED); np.random.seed(SEED)
```

After the datasets are built (Phase 0 end), subsample the val sets deterministically:

```python
# keep val eval bounded; same subsample every run (paste of eval/sweep/sweep_utils.subsample)
ds_valc = ds_valc.select(_subsample_idx(ds_valc.num_rows, VAL_SUBSAMPLE, SEED))
if ds_valr: ds_valr = ds_valr.select(_subsample_idx(ds_valr.num_rows, VAL_SUBSAMPLE, SEED))
```

- [ ] **Step 2: Inline the verified helpers**

Add one cell that pastes the verified functions from `eval/sweep/sweep_utils.py`
(`make_grid`, `subsample`, `build_leaderboard`, `pick_best`, `render_markdown`) plus a
small index helper used above:

```python
# Inlined from eval/sweep/sweep_utils.py (Kaggle kernels are self-contained).
# Keep in sync with that module — it is the tested source of truth.
import random
def make_grid(lrs, ranks, seed):
    grid = []
    for lr in lrs:
        for rank in ranks:
            grid.append({"config_id": f"lr{lr:g}_r{rank}", "lr": lr, "rank": rank,
                         "alpha": 2 * rank, "seed": seed})
    return grid
def _subsample_idx(n_total, n, seed):
    if n is None or n_total <= n:
        return list(range(n_total))
    rnd = random.Random(seed)
    return sorted(rnd.sample(range(n_total), n))
def build_leaderboard(rows, baseline):
    out = []
    for r in rows:
        rr = dict(r); rr["reg_delta"] = round(r["val_reg_wer"] - baseline["val_reg_wer"], 3)
        out.append(rr)
    out.sort(key=lambda x: x["val_corr_wer_norm"]); return out
def pick_best(sorted_rows, max_reg_delta=1.0):
    for r in sorted_rows:
        if r["reg_delta"] <= max_reg_delta:
            return r
    return None
_COLS = ["config_id","lr","rank","alpha","epoch","val_corr_wer_norm","val_reg_wer",
         "reg_delta","val_corr_cer","train_loss","wall_s"]
def render_markdown(sorted_rows, best):
    lines = ["| " + " | ".join(_COLS) + " |", "| " + " | ".join("---" for _ in _COLS) + " |"]
    for r in sorted_rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in _COLS) + " |")
    bl = (f"**Best (regression-guarded):** {best['config_id']} (epoch {best['epoch']}, "
          f"val_corr_wer_norm {best['val_corr_wer_norm']}, reg_delta {best['reg_delta']})"
          if best else "**Best (regression-guarded):** none — every config regressed val_reg")
    return "\n".join(lines) + "\n\n" + bl + "\n"
```

- [ ] **Step 3: Phase 1 — sweep loop cell**

Replace the reference notebook's single model/trainer/train cells (8 + 9 + 10) with the loop. A `TrainerCallback` runs per-epoch eval on `val_reg`; the base model is reloaded fresh per config for clean isolation across ranks.

```python
import time, torch, gc
from transformers import (WhisperForConditionalGeneration, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, TrainerCallback)
from peft import LoraConfig, get_peft_model

class RegEvalCallback(TrainerCallback):
    """After each epoch, eval val_reg and stash wer into the rows list."""
    def __init__(self, trainer, ds_valr, sink):
        self.trainer, self.ds_valr, self.sink = trainer, ds_valr, sink
    def on_epoch_end(self, args, state, control, **kw):
        if self.ds_valr is None: return
        m = self.trainer.evaluate(self.ds_valr, metric_key_prefix="valr")
        self.sink[round(state.epoch)] = m.get("valr_wer", float("nan"))

def build_model(rank, alpha):
    m = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
    m.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    m.config.suppress_tokens = []
    m.generation_config.language, m.generation_config.task = LANGUAGE, TASK
    m.model.encoder.requires_grad_(False)
    m.gradient_checkpointing_enable(); m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=LORA_DROPOUT,
                                     target_modules=["q_proj","v_proj"]))
    return m

# baseline (epoch 0) once, on the untouched base model
_base = build_model(8, 16)
_bargs = Seq2SeqTrainingArguments(output_dir=OUT_DIR+"/_base", per_device_eval_batch_size=EVAL_BS,
    predict_with_generate=True, generation_max_length=225, fp16=True, report_to=[],
    remove_unused_columns=False, label_names=["labels"])
_bt = Seq2SeqTrainer(model=_base, args=_bargs, data_collator=collator,
                     compute_metrics=metrics, processing_class=processor)
baseline = {"val_corr_wer_norm": _bt.evaluate(ds_valc)["eval_wer_norm"],
            "val_reg_wer": (_bt.evaluate(ds_valr)["eval_wer"] if ds_valr else float("nan"))}
print("BASELINE", baseline)
del _base, _bt; gc.collect(); torch.cuda.empty_cache()

rows = []
for cfg in make_grid(LRS, RANKS, SEED):
    t0 = time.time()
    model = build_model(cfg["rank"], cfg["alpha"])
    args = Seq2SeqTrainingArguments(output_dir=f"{OUT_DIR}/{cfg['config_id']}",
        per_device_train_batch_size=TRAIN_BS, gradient_accumulation_steps=GRAD_ACC,
        learning_rate=cfg["lr"], warmup_ratio=0.1, num_train_epochs=EPOCHS, fp16=True,
        predict_with_generate=True, generation_max_length=225, eval_strategy="epoch",
        save_strategy="no", logging_steps=20, report_to=[], remove_unused_columns=False,
        label_names=["labels"], seed=cfg["seed"], per_device_eval_batch_size=EVAL_BS)
    trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=ds_train,
        eval_dataset=ds_valc, data_collator=collator, compute_metrics=metrics,
        processing_class=processor)
    reg_by_epoch = {}
    trainer.add_callback(RegEvalCallback(trainer, ds_valr, reg_by_epoch))
    trainer.train()
    # one row per epoch from the eval entries in log_history
    for h in trainer.state.log_history:
        if "eval_wer_norm" not in h: continue
        ep = round(h["epoch"])
        rows.append({**cfg, "epoch": ep,
            "val_corr_wer_norm": round(h["eval_wer_norm"], 3),
            "val_reg_wer": round(reg_by_epoch.get(ep, float("nan")), 3),
            "val_corr_cer": round(h.get("eval_cer", float("nan")), 3),
            "train_loss": round(h.get("eval_loss", float("nan")), 4),
            "wall_s": int(time.time() - t0)})
    del model, trainer; gc.collect(); torch.cuda.empty_cache()
    print(f"done {cfg['config_id']} in {int(time.time()-t0)}s")
```

- [ ] **Step 4: Phase 2 — write leaderboard**

```python
import csv, json
lb = build_leaderboard(rows, baseline)
best = pick_best(lb, max_reg_delta=MAX_REG_DELTA)
with open(OUT_DIR+"/leaderboard.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=_COLS); w.writeheader()
    for r in lb: w.writerow({c: r.get(c, "") for c in _COLS})
with open(OUT_DIR+"/leaderboard.md", "w") as f:
    f.write(f"# Sweep leaderboard\n\nbaseline: {json.dumps(baseline)}\n\n")
    f.write(render_markdown(lb, best))
print(render_markdown(lb, best))
print("wrote", OUT_DIR+"/leaderboard.csv")
```

- [ ] **Step 5: Validate the notebook file is well-formed**

Run: `python -c "import nbformat; nb=nbformat.read('notebooks/whisper_sweep_kaggle.ipynb', as_version=4); nbformat.validate(nb); print('cells:', len(nb.cells))"`
Expected: prints `cells: N` (no validation error)

- [ ] **Step 6: Commit**

```bash
git add notebooks/whisper_sweep_kaggle.ipynb
git commit -m "feat(sweep): self-contained Kaggle hyperparameter-sweep notebook"
```

---

## Task 5: Run procedure + caveats doc

**Files:**
- Modify: `docs/specs/whisper-hyperparam-sweep.md` (append a "How to run" section)

- [ ] **Step 1: Append the run procedure**

Append to the spec:

```markdown
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
```

- [ ] **Step 2: Commit**

```bash
git add docs/specs/whisper-hyperparam-sweep.md
git commit -m "docs(sweep): Kaggle run procedure + leaderboard reading guide"
```

---

## Self-Review notes

- **Spec coverage:** Phase 0/1/2 → Task 4 steps 1/3/4; pure helpers + tests → Tasks 1–3;
  carried-over bug guards → Task 4 step 1; run procedure + caveats → Task 5. All spec
  sections map to a task.
- **Type consistency:** `_COLS`, `make_grid`, `build_leaderboard`, `pick_best`,
  `render_markdown` signatures are identical in `sweep_utils.py` and the notebook inline
  paste. Row dict keys (`config_id, lr, rank, alpha, epoch, val_corr_wer_norm,
  val_reg_wer, reg_delta, val_corr_cer, train_loss, wall_s`) are consistent across
  Tasks 2–4.
- **Known duplication:** helpers exist both in `sweep_utils.py` (tested) and inline in
  the notebook (Kaggle self-containment). The inline copy is marked "keep in sync".
```
