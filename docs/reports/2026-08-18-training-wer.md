# Training WER exists now, and its average hides the useful result

2026-08-18. CPU only; no GPU, paid API, test or sealed holdout access.

`artifact-ct2-fixed` was decoded on an outcome-blind simple random sample of 300
rows from the exact 28,967-row training parquet used by `artifact-adapter-fixed`
(SHA-256 `f37d3532...c1e8`). The sample was frozen before scoring by a salted
SHA-256 ordering: 300 clips, 136 meetings, all 9 training cities, 861 seconds of
audio, 151 `correction` and 149 `no_edit` rows, 2,368 normalized reference tokens.
Its fingerprint is `78e5c3e...e5fd1`.

## Result

| slice | WER | substitutions | deletions | insertions |
|---|---:|---:|---:|---:|
| all training sample | **0.1313** | 0.0785 | 0.0144 | 0.0384 |
| `correction` labels | **0.2261** | 0.1459 | 0.0256 | 0.0546 |
| `no_edit` labels | **0.0385** | 0.0125 | 0.0033 | 0.0226 |

The meeting-clustered 95% interval for overall training WER is **[0.1056,
0.1619]**. No item dominates it: the largest clip supplies 4.5% of all errors and
the largest meeting 6.4%.

The average alone says the adapter fits its training labels somewhat better than
validation, but not nearly to zero. The split is the important result: almost all
remaining training error is in the human-correction half. `no_edit` is nearly
reproduced, while `correction` WER is 22.6%.

## What the adapter learned relative to base large-v3

The base `artifact-ct2-base-large-v3` was then decoded on the exact same frozen
sample and stack:

| slice | base WER | fixed-adapter WER | improvement |
|---|---:|---:|---:|
| all | 0.2728 | 0.1313 | **−0.1415** |
| `correction` | 0.4471 | 0.2261 | **−0.2210** |
| `no_edit` | 0.1020 | 0.0385 | **−0.0635** |

The paired overall improvement is 14.15 WER points, meeting-clustered 95% interval
**[11.38, 17.25] points**. Substitutions, deletions and insertions all fall. No row
dominates the paired effect: the largest contributes 3.0% and leave-one-row-out never
reverses its sign.

Therefore the adapter **does learn the corrections strongly**; “the recipe ignores
corrections” is ruled out on this training sample. The remaining 22.6% correction-row
WER may still mix intrinsically hard speech, boundary/label mismatch and incomplete
fit. Training agreement alone cannot separate those mechanisms or prove that the
labels are wrong.

## Validation comparison

The exact matched reference is the re-decoded `artifact-ct2-fixed` arm R from
`served-config-2026-08`: same CPU int8/16-thread stack, frozen beam-5 CONTROL,
`ftoks`, per-segment token concatenation and per-item seed policy. It scores
**0.1548 WER** on the 39 training-disjoint validation windows (31 meetings, 11,911
tokens). Training minus validation is **−0.0235**, descriptive only. Training uses
isolated 0.3–30 s utterances with 0.2 s padding, while validation uses long windows;
the difference confounds in-sample status with acoustic-unit shape, and the training
interval includes the validation point.

Two other numbers are kept separate rather than silently selecting a baseline:

- `0.1589` is the same fixed artifact under the old raw segment-string join, which
  can fuse words and is not representation-matched.
- `0.1556–0.1584` is the range for A_s13/A_s29/A_s47, three distinct training
  artifacts used to calibrate training-run variation, not this fixed artifact.

## What changes next

Every future training run should report this fixed sample and the frozen validation
set at comparable checkpoints, always with S/D/I and the two label-source slices.
The next CPU data audit should concentrate on the **residual** `correction` errors:
stratify their 22.6% WER by boundary status, duration, overlap and correction category,
then listen to a blinded sample of the highest-loss strata. Since learning is already
large on both slices, the first new training mechanism remains the dense-30s arm; the
audit decides whether it also needs a protected label/boundary lane.

The optional July-broken pass was not run. It is not needed for the absolute training
WER question and the artifact remains `KNOWN_BROKEN` by provenance.

An initial 1,000-row runtime pilot was stopped manually after 25 unscored fixed-arm
decodes projected 80–100 minutes per arm. No hypotheses or WER were inspected. The
300-row prefix was then frozen in a new cache before any outcome was scored.

Harness: [`training_wer.py`](../../eval/controlled_eval/training_wer.py). Aggregate:
[`results_training_wer.json`](../../eval/results_training_wer.json). Transcript,
audio and hypothesis text remain under `~/.cache/oc-public/training-wer-2026-08-n300/`.
