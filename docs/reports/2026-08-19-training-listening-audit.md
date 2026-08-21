# Blind listening audit: hard rows unresolved, dense outlier references incomplete

## Result

The reviewer completed all 38 frozen items. In the 36 outcome-enriched training
clips, the frozen answers were:

| Question | yes | no | unsure |
|---|---:|---:|---:|
| Label faithful | 10 | 0 | 26 |
| Boundaries usable | 8 | 1 | 27 |

Jointly, 7 clips were `yes/yes`, one was `yes/no`, and 28 had at least one
`unsure` with no definite `no`. Under the frozen interpretation, only the one
definite failure routes directly to correction. The 28 uncertain clips remain
unresolved; they are neither counted as defects nor certified as clean.

Both insertion-heavy validation windows were judged **`material_omission`**: the
published OpenCouncil reference omits a material amount of clearly spoken content.
They must be audio-faithfully re-referenced before their model insertions can be
classified as hallucinated or recovered speech.

## Interpretation

This sample was deliberately enriched for high residual WER and suspect boundaries,
so its proportions do not estimate dataset-wide label quality. It nevertheless
fails to certify the selected difficult rows as an audited protected lane: only 7
of 36 pass both questions as-is. The reviewer's qualitative summary was that most
clips had issues. Private notes are strongly boundary-oriented, but notes were not a
frozen scored field and therefore guide correction only; they do not turn `unsure`
into `no`.

The two validation judgments explain why the dense screen's insertion regression
was concentrated there and why many independent ASR systems also appear
insertion-heavy. They do not retroactively change the preregistered dense decision:
it remains **`SCREEN — STOP`**.

## Decision

1. Re-reference the two validation windows against audio, preserving the original
   published-reference score as a separate agreement metric.
2. Admit the 7 `yes/yes` rows to an audited-hard lane as-is, route the definite
   failure to correction, and keep the 28 uncertain rows out of both the clean core
   and audited-good lane until corrected or re-reviewed.
3. Do not filter an entire stratum from this selected sample. Build the clean core
   from the independent L2 rules and protect audited names/fast/hard/boundary rows.
4. Freeze the manifest, equal-source-hours control and GPU cost before requesting
   explicit approval to train.

Aggregate: `eval/results_training_listening_audit.json`. Protocol:
`docs/specs/2026-08-19-training-listening-audit.md`. Private answers and notes remain
under `~/.cache/oc-public/training-listening-audit-2026-08/`; answers SHA-256:
`0a49fdc66fc6979e448aa4076828535e0da2c7da0d402fd8589d9bc5c9c0523c`.
