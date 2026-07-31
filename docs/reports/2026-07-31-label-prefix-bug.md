# The label-prefix bug: every GPU fine-tune trained on shifted targets (2026-07-31)

Status: **fixed in code** (`00d9235`) and **measured** (`eval/ab_label_bug/`, 6 paired
LoRA runs on an A40, 2026-08-01).

**The short answer: the bug did exactly what the diagnosis said, and it cost almost
nothing in WER.** The legacy model learns to emit `<|startoftranscript|>` as its first
token — rank 1 in 50%, 55% and 95% of clips across three seeds, against a base model
that ranks it ~2000th. But at inference Whisper forces the prompt and suppresses that
token, so the habit is mostly hidden: decoded WER on held-out corrected utterances
moves by +0.005, +0.001 and −0.001 across the three seeds — mean **+0.0018, sign not
even consistent**. What *is* consistent is the likelihood: the legacy objective is
worse in 3/3 seeds and far less stable (NLL 0.340 / 0.342 / 0.403 vs a fixed arm that
lands on 0.3167–0.3186 every time).

So: retrain, because the objective was wrong and the corrected one is strictly better
behaved — not because the bug cost measurable accuracy. Skip the sweep re-run; the
effect is smaller than the noise the sweep already could not resolve. [Full numbers
below](#what-it-cost-measured-2026-08-01).

## What was wrong

Whisper's tokenizer returns the full prompt prefix for a plain string:

```
<|startoftranscript|> <|el|> <|transcribe|> <|notimestamps|> …text… <|endoftext|>
     50258             50281      50360           50364                  50257
```

`WhisperForConditionalGeneration` builds `decoder_input_ids` from the labels itself
(`shift_tokens_right`, prepending `decoder_start_token_id` = 50258), so the labels
must not keep their own copy of `<|startoftranscript|>`. The collator tried to drop
it:

```python
if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
    labels = labels[:, 1:]
```

For Whisper, `tokenizer.bos_token_id` is `<|endoftext|>` (**50257**), not
`<|startoftranscript|>` (**50258**) — the tokenizer reuses `<|endoftext|>` as bos,
eos and pad. The condition was never true. The strip never ran. Training therefore
saw:

```
decoder_input_ids: <|sot|> <|sot|> <|el|> <|transcribe|> <|notimestamps|> t1 t2 …
labels:                    <|sot|> <|el|> <|transcribe|> <|notimestamps|> t1 t2 …
```

Two consequences, on every sample of every affected run:

1. the model was trained to emit `<|startoftranscript|>` as its first output token —
   something no caller ever asks for;
2. every content token was learned one decoder position later than it appears at
   inference, so the model was optimised for a world shifted one step from the one
   it is used in.

Nothing warned. The loss curve looked normal, evaluation ran, and the numbers were
plausible.

## Verification

Confirmed locally against the real tokenizer (`transformers` 5.6.2,
`openai/whisper-large-v3`), not from memory:

```
tokenizer.bos_token_id                 -> 50257  (<|endoftext|>)
tokenizer("Καλημέρα σας").input_ids[0] -> 50258  (<|startoftranscript|>)
model.config.decoder_start_token_id    -> 50258
```

## Scope

| Entry point | Affected |
|---|---|
| `notebooks/whisper_finetune_kaggle.ipynb` (first fine-tune) | yes |
| `notebooks/whisper_sweep_kaggle.ipynb` (the hyperparameter sweep) | yes |
| `notebooks/train_smoke.py` | yes |
| `notebooks/train_runpod.py` (the published 2026-07-22 adapter) | yes |
| `eval/autoresearch/experiment.py` (CPU mini-PC sweep) | **no** — compares against `decoder_start_token_id`, always did |

The consequence worth stating plainly: the sweep that produced `r=32`, `lr=1e-4`,
`2 epochs` ran with broken targets. Those values are not "confirmed hyperparameters".
The honest description is *selected under the legacy label schema, never revalidated
under the corrected objective*.

## Two claims corrected along the way

Both were in the docs, both are wrong, and neither is the bug:

- **"Encoder frozen / 0.51% of parameters trainable."** `model.model.encoder.requires_grad_(False)`
  runs *before* `get_peft_model`, and PEFT then injects fresh trainable LoRA adapters
  into every module matching `q_proj`/`v_proj` — encoder included. Read back from the
  published adapter: **128 encoder LoRA tensors, 15.73M trainable parameters**, not
  7.86M with a frozen encoder. The accurate statement is "the base encoder weights are
  frozen, but newly injected encoder LoRA parameters are trainable". The model could
  learn to listen differently; it was not prevented from doing so.
- **"Decoding was broken by `model.config.suppress_tokens = []`."** It was not.
  Current transformers reads generation settings from `generation_config`; that
  assignment wrote to a field nothing reads, so it never disabled the suppression
  list. Removed anyway, but it explains none of the results.

## What the fix does

`00d9235`:

- the collator compares against `decoder_start_token_id`, checks the **raw**
  sequences before padding (a left-padded batch would false-fail an after-padding
  check), rejects left padding explicitly, and **raises** instead of training on a
  prefix it does not recognise;
- generation is configured on `generation_config` only, and the effective values are
  logged;
- every `checkpoint-*` directory is stamped with a run fingerprint that includes the
  label semantics, and resume refuses an unstamped or mismatched checkpoint — so a
  pre-fix checkpoint cannot re-enter a corrected run halfway;
- trainable parameters are logged split by encoder/decoder, so the freeze claim is
  checked rather than assumed;
- `eval/tests/test_whisper_label_prefix.py`: 13 tests, verified in both directions
  (4 of them fail against the pre-fix code — a test never seen red protects nothing).

## What it cost, measured (2026-08-01)

Six paired LoRA runs on one A40: 3 seeds × {fixed, legacy}, identical data, identical
300 steps, arm order alternated. 24 training meetings (1,489 clips, 1.61 epochs),
16 held-out meetings from argos + orestiada (305 corrected clips + 128 no-edit).
Both arms in every pair recorded the **same `batch_order_sha`**, so they provably saw
the same examples in the same order — the collator really was the only variable.
Raw results: `eval/ab_label_bug/results_ab.json`. Cost: ~3.3 h of pod time, ~$1.50.

| run | val_corr WER | CER | val_reg WER | canonical NLL | p(SOT) at pos 0 | SOT rank 1 | p(`<\|el\|>`) |
|---|---|---|---|---|---|---|---|
| base (untrained) | 0.2801 | 0.1433 | 0.1671 | 0.6794 | 1.9e-07 | 0% | 0.758 |
| fixed s13 | 0.2264 | 0.1025 | 0.0652 | 0.3186 | 1.8e-09 | 0% | 0.997 |
| legacy s13 | 0.2314 | 0.1053 | 0.0747 | 0.3398 | 0.467 | 50% | 0.523 |
| fixed s23 | 0.2272 | 0.1024 | 0.0693 | 0.3179 | 1.8e-09 | 0% | 0.997 |
| legacy s23 | 0.2286 | 0.1032 | 0.0774 | 0.3421 | 0.512 | 55% | 0.479 |
| fixed s37 | 0.2286 | 0.1016 | 0.0666 | 0.3167 | 1.7e-09 | 0% | 0.997 |
| legacy s37 | 0.2275 | 0.1039 | 0.0720 | 0.4034 | 0.891 | 95% | 0.100 |

### 1. The mechanism is confirmed outright

`p(SOT)` at decoder position 0 — the position where the legacy target actually puts
the doubled `<|startoftranscript|>` — goes from ~1e-9 in every fixed run to 0.47, 0.51
and 0.89 in the legacy runs, ranking **first** in half to nearly all clips. The
probability of the correct language token collapses from 0.997 to 0.52, 0.48 and 0.10.
This is not an inference from WER; it is the learned behaviour read straight off the
logits. Nothing about the diagnosis is in doubt.

It also cannot be seen from transcripts: large-v3's `generation_config` suppresses
token 50258, so a legacy model's output looks clean. The first version of this
experiment read the probability at the wrong position and would have reported "no
effect" — Codex caught that before the run.

### 2. The WER cost is at the noise floor

val_corr WER, legacy minus fixed, paired **cluster** bootstrap over the 16 held-out
meetings:

| seed | Δ WER | 95% CI | significant | meetings where legacy is worse |
|---|---|---|---|---|
| 13 | +0.0050 | [+0.0006, +0.0103] | yes | 10/16 |
| 23 | +0.0014 | [−0.0073, +0.0091] | no | 8/16 |
| 37 | −0.0011 | [−0.0076, +0.0066] | no | 8/16 |

Mean +0.0018, and the sign does not survive a change of seed. A single-seed run would
have reported seed 13's "+0.5 points, CI excludes zero" as the answer; three seeds
show that is one draw from a distribution straddling zero. On val_reg the direction is
at least consistent (+0.0095, +0.0082, +0.0054, one of three significant), which is
weak evidence of a real but small effect on ordinary speech.

Why so small: the shift damages what the model has learned, but at inference the
prompt is forced and the offending token is suppressed, so decoding walks back onto
the correct track almost immediately.

### 3. The likelihood cost is consistent

Teacher-forced NLL on val_corr, scored under the *correct* label layout for both arms:
fixed lands on 0.3186 / 0.3179 / 0.3167 — a spread of 0.002 — while legacy gives
0.3398 / 0.3421 / 0.4034. Worse in 3/3, and roughly forty times more variable between
seeds. The corrected objective is not just better on average; it is stable, and the
broken one is not. That, rather than the WER delta, is the argument for retraining.

### 4. Unexpected: the fixed adapter clearly beats base whisper here

fixed minus baseline on val_corr: **−0.0537, −0.0530, −0.0515**, every CI excluding
zero, and **0 of 16 meetings worse** in all three seeds. val_reg goes 0.1671 → ~0.067.

This sits badly with the [2026-07-25 postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md),
which found the published adapter tying base whisper and regressing on corrected
utterances. Before anyone celebrates, the differences: this is a different held-out
sample (16 meetings, 305 clips) scored by this script's own normalization, the model
is evaluated in the same HF stack it was trained in rather than through faster-whisper,
and val_reg references are the previous system's output — so part of that 0.167 → 0.067
is the model learning to imitate the old system, not to transcribe better. **This does
not overturn the postmortem.** It is a reason to put both models through the controlled
harness, on the same sample, before making any claim either way.

### What this means for the plan

- **Do not re-run the hyperparameter sweep.** Its picks were made under the broken
  objective, but the objective's effect on WER is smaller than the seed noise the
  sweep already could not resolve — a re-run would buy a differently-noisy answer.
  Keep `r32 / lr 1e-4 / 2 epochs` and describe it as *selected under the legacy label
  schema, never revalidated*.
- **Do retrain and republish**, for the reason in §3 (a correct, stable objective and
  no pathological first-token habit), not for a promised WER gain. Anyone reading
  "we found a training bug" should not expect the new adapter to be much more accurate.
- **Re-open the "does the fine-tune beat base" question** through the controlled
  harness, given §4.

### Caveats

This is a scaled-down recipe — 1,489 clips and 300 steps against the published run's
28,967 clips and 2 epochs — identical for both arms but not a reproduction of it. A
bug that mostly damages calibration could plausibly matter more over 8 hours of
training than over 20 minutes. Three seeds, 16 held-out meetings, one training-set
sample: enough to rule out a large effect, not enough to pin a small one.

## Second, unrelated problem (not touched here)

52% of the training data is not human-verified — it is the previous system's output,
and the held-out "general speech" set uses exactly that as ground truth. Some of the
measured "improvement on ordinary speech" may be the model learning to imitate the old
system. That is a dataset decision, not a bugfix, and it is tracked separately.
