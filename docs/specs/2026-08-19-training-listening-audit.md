# Frozen blind training/reference listening audit

Frozen after the dense screen diagnosis and before reading any human verdict.

## Queue

- 36 training clips already selected by the frozen residual-audit protocol from
  `adjustment>1`, `suspect_cut_start` and `other_lexical` strata.
- The two validation windows with the largest aggregate dense-minus-isolated
  insertion deltas across the three seeds. This is an explicitly diagnostic,
  outcome-selected audit; it cannot estimate a population rate or revise the
  dense screen's `SCREEN — STOP` decision.

The 36 training clips contain 123.9 seconds of audio. The two validation windows
contain 298.9 seconds. Total raw audio is 7.0 minutes; expected human time with
reading and replay is roughly 20–30 minutes.

## Blinding and questions

The reviewer sees audio and one reference/label only. Arm, seed, hypotheses, WER,
S/D/I, source stratum, selection score and window identity are absent from the
served page. The hidden key is stored outside the served directory.

For every training clip, answer exactly:

1. Is the training label faithful to what is audible? `yes/no/unsure`.
2. Are the clip boundaries usable for training? `yes/no/unsure`.

For each validation window, answer exactly whether the published reference omits
clearly spoken words: `complete/minor_omission/material_omission/unsure`. Optional
notes may contain approximate timestamps, but notes never enter the aggregate.

## Frozen interpretation

- A training row with either `label_faithful=no` or `boundaries_usable=no` is routed
  to a protected correction queue, not silently deleted. `unsure` is unresolved,
  never counted as a defect. Because sampling is residual/stratum-enriched, the
  observed proportions do not estimate whole-dataset prevalence and cannot justify
  dropping an entire stratum.
- If either validation window has `minor_omission` or `material_omission`, it must be
  audio-faithfully re-referenced before using it to distinguish genuine insertions
  from recovered speech. If both are `complete`, their extra model words are treated
  as genuine insertion evidence on these two windows only. Any `unsure` leaves the
  mechanism unresolved.
- No answer can retroactively promote the stopped dense arm. The audit only informs
  strict validation construction and the protected lane of the next hybrid-data arm.

The page is built by `eval/controlled_eval/build_training_listening_page.py`; the
aggregate scorer is `eval/controlled_eval/score_training_listening_audit.py`.
