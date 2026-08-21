# Frozen residual-correction audit

Frozen before computing any stratum WER from the completed 300-row training sample.

## Population and estimand

Use only the 151 `source=correction` rows in the already-frozen sample of
`exp-2026-08-18-training-wer`. Reuse its fixed/base hypotheses, normalizer and
per-segment token concatenation. This measures agreement with training labels, not
fidelity to audio.

Report fixed S/D/I and WER, residual-error contribution, and paired base-minus-fixed
WER for these predeclared dimensions:

- intended clip duration: `<2`, `2-4`, `4-8`, `>=8` seconds;
- `boundary_status`, with source values unchanged;
- total absolute boundary adjustment: `none`, `<=0.25`, `0.25-1`, `>1` seconds;
- overlap flag;
- each declared correction error category, exploded for multi-label rows.

Rows with multiple categories enter each relevant category; category strata therefore
overlap and never sum to the overall population. A stratum is ranking-eligible only at
`n>=10` and `reference_tokens>=50`. Rankings are descriptive and do not select a model.

## Blind listening queue

Take the three eligible strata with highest fixed WER, at most one per dimension.
Within each, deterministically sample up to eight rows with residual errors and four
with zero residual errors, deduplicating rows. The review file exposes audio and the
training label but no model hypothesis, WER, base/fixed identity, or source stratum.
A separate hidden key retains that metadata. Both files stay under
`~/.cache/oc-public/training-residual-audit-2026-08/` and never enter git.

Listening asks only whether the label is faithful and whether its audio boundaries
are usable. No training-data change follows from the CPU aggregate alone.
