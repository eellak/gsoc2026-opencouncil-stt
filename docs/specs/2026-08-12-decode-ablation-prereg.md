# Preregistration — decode-threshold ablation

`exp-2026-08-12-decode-ablation` · frozen 2026-08-11, before any arm was decoded ·
plan: [`2026-08-11-endgame-handoff-plan.md`](2026-08-11-endgame-handoff-plan.md)

## Question

Our served decoder deletes 3.3 words for every word it inserts; Scribe v2 deletes 1.6
(`exp-2026-08-11-error-analysis`). The server passes only `beam_size` and
`condition_on_previous_text`; every threshold that can silently drop audio is at a
faster-whisper default nobody in this project has ever chosen. Does moving one of
those thresholds lower the deletion rate without paying for it in WER?

This is a decode-config experiment. The weights never change.

## Evaluation set

[`research/eval-freeze-2026-08/manifest.json`](../../research/eval-freeze-2026-08/manifest.json),
frozen first: **39 windows, 31 meetings, 11,911 reference tokens**, argos +
orestiada, all meetings before 2026-06-01. Holdout: **7 windows, 5 meetings, 2,101
tokens**, touched exactly once in Task 1.4.

The handoff plan describes the 39 as "argos + orestiada minus the 7 temporal-test
windows". That arithmetic is wrong and the manifest records the correction: the
≥2026-06-01 rule selects 7 windows across the whole 260-window benchmark but only
**one** of them is in argos/orestiada. 40 − 1 = 39. The 39 are the same 39 the
`exp-2026-08-10-benchmark-fixed-adapter` caveat re-measured on.

## Software and hardware, fixed for every arm

Python 3.12.3, `faster-whisper` 1.2.1, `ctranslate2` 4.8.1, `.venv-eval`, minipc,
**device `cpu`, compute type `int8`, 16 threads**, `artifact-ct2-fixed`
(`model.bin` sha256[:16] `8a1a3b257d0c1bdb`, verified against the ledger on disk).

Every arm decodes on this one device in this one environment. No number here may be
compared against a GPU-produced number: CPU int8 and CUDA int8 give different tokens
from the same artifact (`exp-2026-08-10-benchmark-fixed-adapter`, caveat 4). That is
why the control is decoded fresh rather than lifted from the 2026-08-10 benchmark
run, which ran on an RTX A4000.

## Arms

One behavioural change each; everything else pinned to the control. Every arm passes
**all** of these options explicitly, and the resolved `TranscriptionOptions` is saved
into the results.

```
language="el", beam_size=5, condition_on_previous_text=False,
word_timestamps=False, vad_filter=False, task="transcribe",
temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
no_speech_threshold=0.6, log_prob_threshold=-1.0,
compression_ratio_threshold=2.4
```

| arm | change | family | role |
|---|---|---|---|
| **A** | none — the control | — | control |
| **B** | `no_speech_threshold=0.8` | anti-deletion | exploratory |
| **C** | `no_speech_threshold=None` | anti-deletion | **primary** |
| **D** | `temperature=[0.0]` (no fallback) | anti-insertion | **primary** |
| **E** | `log_prob_threshold=None` | compound | exploratory |
| **F** | `compression_ratio_threshold=None` | — | exploratory |

### What arm E actually does — a correction to the handoff plan

The plan states that `log_prob_threshold=None` "makes the no-speech gate inert".
Read against the installed faster-whisper 1.2.1 (`transcribe.py:1215-1235`), it does
the opposite:

```python
should_skip = result.no_speech_prob > options.no_speech_threshold
if options.log_prob_threshold is not None and avg_logprob > options.log_prob_threshold:
    should_skip = False          # <- the rescue
```

`log_prob_threshold` is the *rescue* clause: a confident segment survives a high
no-speech probability. Setting it to `None` removes the rescue, so the gate fires on
`no_speech_prob` alone. Arm E therefore **increases** skipping while also disabling
the low-logprob temperature fallback. It is a deletion hazard, not a deletion fix,
and it is preregistered as exploratory with that expectation stated in advance.

The rest of the plan's semantics check out against the same source: the compression
and logprob thresholds trigger a retry at the next fallback temperature and never
discard output on their own (`transcribe.py:1479-1500`), and positive fallback
temperatures switch to `beam_size=1` sampling (`transcribe.py:1432-1444`), which is
why every window is preceded by `ctranslate2.set_random_seed(seed)` with a seed
derived deterministically from `(arm, window_id)`.

`WhisperModel.transcribe` is used, not `BatchedInferencePipeline`. The segment
generator is consumed fully before any result is recorded.

### Deviations from the served configuration, recorded in advance

1. **`beam_size=5`** is the code default in `serve/oc-asr/oc_asr_server.py:46`. The
   mini-PC deployment overrides it to `OC_ASR_BEAM=2` in `asr.env`. Beam is held at
   5 in every arm; beam size is out of scope for this experiment and no arm's result
   transfers to a beam-2 deployment without re-checking.
2. **`word_timestamps=False`**, while the server passes `True`. Held identical
   across arms. Re-verified against the real server in the Task 1.4 ship check.
3. **Device**: the served benchmark row came from a CUDA pod; this runs on CPU.
   Within-experiment only, as above.

## Estimands, defined before any number exists

Scored with the frozen normalizer recorded in the manifest:
`eval.controlled_eval.scoring.wtoks` plus the frozen filler regex. The benchmark's
own `wer-nofillers` normalizer runs server-side and is not importable; this is a
documented stand-in, identical across arms, not byte-identical to the leaderboard.

- **Primary: micro-WER** = `Σ(S+D+I) / Σ(ref tokens)` over the 39 windows.
- **Deletion rate** = `ΣD / Σ(ref tokens)`; **insertion rate** = `ΣI / Σ(ref tokens)`.
- **Delta** = arm − control. Negative is good.

S/D/I come from the global alignment in `eval/controlled_eval/exp_same_stack.py`
(`sdi`), ties broken toward substitution.

## Uncertainty

Paired block bootstrap resampling **meetings**, not windows: 31 blocks, 4000
replicates, seed 7 (`eval.controlled_eval.scoring.cluster_bootstrap`). 95% two-sided
percentile interval. 31 blocks clears the plan's ≥8 threshold, so the intervals are
inferential rather than merely descriptive.

## Gates

A ship candidate must clear its family's gate. Non-inferiority margin is 0.0 — a knob
does not get to raise WER.

- **B / C** (anti-deletion): deletion-rate delta 95% **upper bound < 0**, AND
  micro-WER delta upper bound ≤ 0.
- **D / E / F** (anti-insertion / other): insertion-rate delta upper bound < 0, AND
  deletion-rate and micro-WER deltas both non-inferior (upper bound ≤ 0).
- **C and D are the primary arms.** B, E, F are exploratory and cannot ship on their
  own whatever they show.
- **No post-hoc arm mixing.** If two knobs both pass, their combination is a new
  hypothesis needing its own holdout confirmation, not a free merge.

## Influence check

Leave-one-window-out on each pooled primary delta. Domination is flagged if removing
any single window reverses the sign of the primary conclusion (deletion-rate delta
for B/C, insertion-rate delta for D/E/F). No "% of delta" rule.

## Diagnostics recorded per window per arm

Enough to tell "the knob did nothing" from "the condition never arose":

- number of segments emitted, and total decoded duration;
- how many times the no-speech skip fired, and the `no_speech_prob` at each;
- which fallback temperatures were actually attempted per segment;
- how many segments tripped the compression-ratio and logprob thresholds;
- wall-clock decode seconds.

Hypotheses and per-window scores live under `~/.cache/oc-public/decode-ablation/`.
Only aggregates enter the repo.

## Confirmation and stopping rule

If a primary arm passes, that arm and the control are decoded **once** on the 7
holdout windows. It ships only if both its deltas keep their sign there. If it fails,
that is the result — recorded, not iterated on.

The holdout carries its own limit, recorded in the manifest: 6 of the 7 windows are
in cities that are in the training set. For a within-model decode comparison, where
both arms share weights, that is acceptable. It would not be acceptable for any
model-vs-model claim.

If a step fails twice the same way, the run stops and the failure is written down.
