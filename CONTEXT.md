# OpenCouncil ASR Research

Canonical language for interpreting training experiments and their evidence.

## Experimental evidence

**Screen**:
An underpowered training comparison that may decide whether replication is worth buying, but cannot establish that an arm wins or loses.
_Avoid_: promising result, negative result, conclusion

**Advancing screen**:
A screen whose point estimates improve validation WER without increasing deletions and whose insertion, length, and dominance guards pass; it licenses only the next experiment stage.
_Avoid_: promising arm, provisional win

**Stopped screen**:
A screen that fails an operational gate and therefore receives no further budget; it does not establish that the underlying idea is generally harmful.
_Avoid_: failed idea, negative finding

**Mechanism candidate**:
One isolated training lever with a causal account of which error source it should change.
_Avoid_: edge, idea bundle, configuration

**Local mechanism preflight**:
A zero-GPU check that can veto a mechanism candidate for infeasibility, leakage, insufficient data, or failure to activate; it cannot establish model quality.
_Avoid_: local proof, winning proxy

**Short-horizon mechanism screen**:
A matched control/treatment training comparison on the real model at a frozen early checkpoint; it decides only whether more training budget is warranted.
_Avoid_: quick win, miniature confirmation

**Confirmed win**:
A preregistered arm whose paired-seed comparison meets the chosen power and interval rule, improves validation WER, and passes every safety guard.
_Avoid_: better run, best seed

**Seed-effect range**:
The maximum minus the minimum effect across paired seeds; it is not a variance or standard deviation.
_Avoid_: seed variance, per-seed spread

**Primary endpoint**:
Validation WER, evaluated on the frozen validation set; training-set WER and benchmark WER do not replace it.
_Avoid_: benchmark score, trainer validation

**Deletion guard**:
A safety constraint under which lower WER accompanied by a higher deletion-rate point estimate is a failed arm, not a trade-off.
_Avoid_: secondary metric
