# Full run with the corrected objective (2026-08-02)

The rerun of the published fine-tune with one thing changed: the labels are right.
Same data, same recipe, same held-out split, same eval as
[the 2026-07-23 run](2026-07-23-fulltrain-run.md), which trained through the
[label-prefix bug](2026-07-31-label-prefix-bug.md).

Adapter: `~/oc-asr-serve/adapter-fixed-2026-08-01` (not yet published). A40, ~12.8 h
of pod time (~$5.6): 1.5 h building 28,967 + 3,157 + 4,722 clips from 440 meetings,
1.3 h of baseline eval, 7.4 h for 7,242 steps, 1.3 h of final eval.

## The comparison is honest

Both baselines reproduce the 2026-07-23 run almost exactly, so data, clips and eval are
the same pipeline and any difference is attributable to the objective rather than to a
changed harness:

| baseline | this run | 2026-07-23 |
|---|---|---|
| val_corr WER / norm / CER | 55.95 / 47.17 / 33.21 | 55.6 / 46.9 / 33.0 |
| val_reg WER / norm / CER | 24.76 / 15.31 / 14.26 | 24.8 / 15.3 / 14.3 |

## Result

| after training | corrected (this run) | buggy (published) | Δ |
|---|---|---|---|
| val_corr WER | **37.37** | 37.74 | −0.37 |
| val_corr WER norm | **29.01** | 29.35 | −0.34 |
| val_corr CER | **17.27** | 17.46 | −0.19 |
| val_reg WER | **10.39** | 10.46 | −0.07 |
| val_reg WER norm | **4.82** | 4.87 | −0.05 |
| val_reg CER | **3.36** | 3.43 | −0.07 |

Better on all six, by very little. This is what the A/B
[predicted](2026-07-31-label-prefix-bug.md#2-the-wer-cost-is-at-the-noise-floor): three
paired seeds at 300 steps put the bug's cost at a mean of +0.0018 WER, and at full scale
it lands at 0.0037 on val_corr — the same order of magnitude and the same sign.

**These are one run against one run.** A 0.37-point difference is well inside
run-to-run variation, and nothing here establishes that the corrected model is better by
a statistically meaningful margin. What the numbers do support is the weaker and more
useful claim: fixing the objective did not cost anything, the improvement is in the
direction the controlled experiment predicted, and the model is now trained against the
target it will actually be asked about at inference.

## What this run does not settle

The training script's own eval is not evidence that this adapter beats base whisper in
production. It scores in the same HF stack the model trained in, and `val_reg`'s
references are the previous system's output, so part of "10.39 vs a 24.76 baseline" is
agreement with the old pipeline rather than transcription quality. The
[2026-07-25 postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md) found the
published adapter tying base whisper once served fairly, and nothing in this run
contradicts that.

Publication is therefore still gated on the controlled harness: same sample, same stack,
base vs this adapter, on the actually-corrected and general subsets. Until that runs, the
right description of this artifact is "the same model, trained correctly", not "a better
model".

## Provenance

`run_meta.json` records `label_semantics: decoder_start_v2`, and every checkpoint carries
the run fingerprint, so this adapter cannot be confused with a pre-fix one or resumed
from one. Trainable parameters logged at load: **15.73M total, 5.24M encoder, 10.49M
decoder** — the published adapter's 384 tensors with 128 in the encoder, which is the
third independent confirmation that "encoder frozen / 7.86M" was never true.
