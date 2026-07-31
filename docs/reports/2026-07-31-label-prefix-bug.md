# The label-prefix bug: every GPU fine-tune trained on shifted targets (2026-07-31)

Status: **fixed in code** (`00d9235`), **cost not yet measured** — the A/B that puts
a number on it is `eval/ab_label_bug/run_ab.py`.

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

## What we still do not know

**How much it cost.** The bug is real and affected every sample, but "real" is not a
number. The base model is strong and only a small adapter changed, so moderate damage
is more likely than catastrophic — that is a prior, not a measurement.

`eval/ab_label_bug/run_ab.py` answers it: two LoRA runs, same data, same seed, same
step count, the collator as the only variable, scored with a paired bootstrap over the
same held-out utterances. It also reads the raw logit rank of `<|startoftranscript|>`
at the first free decoding position — the learned habit is invisible in generated text
because large-v3's `generation_config` suppresses that token, so it has to be measured
before the logits processors run.

Results land here when the run completes.

## Second, unrelated problem (not touched here)

52% of the training data is not human-verified — it is the previous system's output,
and the held-out "general speech" set uses exactly that as ground truth. Some of the
measured "improvement on ordinary speech" may be the model learning to imitate the old
system. That is a dataset decision, not a bugfix, and it is tracked separately.
