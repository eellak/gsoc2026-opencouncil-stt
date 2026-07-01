# Steps 3-4 — faithfulness buckets + human gate

Calibration items: 320 (320 transcribed, 0 failed/excluded)

## Bucket counts (STARTING gates — NOT final)

- REVIEW: 111
- SHORT-SPECIAL: 79
- AUDIT: 55
- BACKBONE: 38
- DROP: 20
- KEEP: 17

- guard-flagged (rate/clip): 94
- short-special: 79

## CER summary
```
cer_before:
count    320.000000
mean       0.234485
std        0.863814
min        0.004167
25%        0.041524
50%        0.090909
75%        0.250000
max       15.000000

cer_soniox:
count    320.000000
mean       0.299019
std        0.801156
min        0.000000
25%        0.055556
50%        0.139980
75%        0.338235
max       12.000000
```

## Starting gates (Codex guesses — Angelos overrides at the gate)
```
backbone_before = 0.03
backbone_soniox = 0.1
drop_before = 0.08
drop_soniox = 0.18
drop_ratio = 2.5
drop_floor = 0.04
keep_before = 0.12
keep_soniox = 0.15
audit_before = 0.15
audit_soniox = 0.15
rate_lo = 1.0
rate_hi = 4.5
clip_lo = 1.5
clip_hi = 30.0
short_chars = 20
short_words = 5
```

## HUMAN GATE

These buckets use the **starting** CER gates. Hand-audit `calib/calib_audit.csv` (DROP/AUDIT/REVIEW rows first), listen to the clips under `calib/clips/`, and read `calib/cer_distributions.png`. Pick the cut where good vs bad separate, then record the chosen thresholds in the runbook and `docs/decisions/data.md`. Do NOT scale to full bulk (Step 5) until the thresholds are locked.

