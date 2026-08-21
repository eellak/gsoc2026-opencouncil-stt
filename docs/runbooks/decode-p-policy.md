# Running the arm P silence-cutting decode policy

Capability: `cap-decode-p-local`. Artifacts: `artifact-decode-policy-p` (the frozen
config), `artifact-ct2-fixed` (the weights). Produced by
`exp-2026-08-18-chunking-aware-decoding` (the harness string in the code and in the
caches is `exp-2026-08-17-...`, a day earlier; same experiment); preregistration
[here](../specs/2026-08-17-chunking-aware-decoding-prereg.md).

## What the evidence is, and what it is not

On the 39 frozen evaluation windows, same decoder and same normalizer:

| arm | WER | deletions | insertions | substitutions |
|---|---:|---:|---:|---:|
| A (control, blind cutting) | 0.15893 | 0.06003 | 0.02015 | 0.07875 |
| V (`vad_filter=True`) | 0.15246 | 0.05911 | 0.01646 | 0.07690 |
| **P (this policy)** | **0.14751** | 0.05575 | 0.01956 | 0.07220 |

P's WER delta against the control is **−0.01142**, cluster-bootstrap CI
**[−0.02025, −0.00195]**, which excludes zero.

**P failed the preregistered gate of that experiment.** The gate was on
*deletions*, and P's deletion delta is −0.00428 with a CI that includes zero. The
WER win comes from *substitutions*. So: P is the best measured arm by WER. It is
not an arm that passed a gate. Do not write it the other way round.

## Smoke check (read-only, instant)

```bash
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_p config
```

Prints the whole policy and its `policy_sha256`. Expect
`5ae98472227696e0...`. A different hash means the policy changed and any number
recorded against the old hash no longer applies.

## Decode one file

```bash
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_p transcribe audio.wav
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_p transcribe audio.wav --json
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_p plan audio.wav
```

`--json` adds the segments, the boundary list with the reason for each boundary,
and the counters. `plan` does the segmentation only — no decode, so a whole
meeting is planned in seconds and you can see where the cuts would land before
paying for the transcription.

## From Python, over many files

```python
from serve.decode_p import DecodeP

decoder = DecodeP.load()            # verifies the CT2 model hash first
for path in paths:
    result = decoder.transcribe(path)
    ...                             # result.text, result.segments, result.boundaries
```

`DecodeP.load()` is the expensive call — load once, decode many. It refuses to run
unless `model.bin` in the directory it is about to load hashes to
`artifact-ct2-fixed`; passing `model_dir` does not get you around that. Only
`verify=False` does, and anything decoded that way has to record which model
produced it.

## What the boundaries tell you

`result.boundaries` gives every internal cut and where it landed:

| `kind` | meaning |
|---|---|
| `vad_silence` | the cut time falls inside a reported silence of at least 0.5 s |
| `forced` | it does not — the 29.5 s ceiling ran out, or the short-tail repair moved the boundary |

This is a *description of where the boundary landed*, deliberately recomputed from
the silence list rather than read from the splitter's own
`cut_silence_duration` field. That field is carried over unchanged when the
short-tail repair **moves** a boundary, so it can claim a silence that is
seconds away from the cut it is attached to. If a transcript looks wrong
somewhere, find the nearest boundary and its kind — a `forced` boundary is a cut
through live speech, and arm P has no context margin to soften it.

## Seeding

Whisper's temperature fallback ladder samples, so without a seed a rerun can
differ. `transcribe(..., seed=N)` calls `ctranslate2.set_random_seed`, which is
**process-global**: it reproduces only if nothing else is decoding in the same
process. With no seed, the RNG is left alone and the output depends on process
history. The experiment used `decode_ablation.seed_for("A", window_id)`.

## Cost

Local CPU int8, 16 threads on the minipc. Measured on the three conformance
windows of ~148 s each: **77 s, 117 s and 68 s** of decode, i.e. **0.46–0.79×
real time** once the model is loaded. The control arm on the same three windows
took 78 s, 206 s and 191 s — P is faster, because a piece that decodes cleanly
never enters the temperature fallback the way a whole 150 s window does. Loading
the model and hashing its 1.5 GB `model.bin` is a fixed cost of a minute or two
on top, paid once per process. No GPU, no money, no external API. Peak RSS is about
1.5 GB for an hour of audio and scales with the length of the file, because the
whole waveform is decoded into memory before planning.

## Whole-meeting audio

Measured on 2026-08-21, see
[the report](../reports/2026-08-21-decode-p-served.md). It works: planning a
64-minute meeting takes ~9 s end to end (3.4 s to decode the mp3, 5.6 s for the
VAD, 0.01 s to split) and produces no tiny pieces and no dropped speech.

The thing to watch is the **forced-cut rate**, which is far more variable on
continuous meetings than on the evaluation windows:

| audio | boundaries | forced |
|---|---:|---:|
| the 39 evaluation windows | 207 | 7.7% |
| `samothraki__jun18_2026` (81.9 min) | 193 | 5% |
| `argos__jun22_2026` (12.7 min) | 31 | 6% |
| `argithea__jun25_2_2026` (64.4 min) | 156 | **41%** |

A meeting with few reported silences gets most of its cuts forced, and a forced
cut is exactly the case P's mechanism does not help with. Run `plan` first and
look at `forced_boundaries` before believing that P will do anything for a
particular meeting.

## Before trusting it on new data

```bash
SC=~/.cache/oc-public .venv-eval/bin/python -m pytest eval/tests/test_decode_p_policy.py -q
```

It asserts that the served constants equal the measured ones, that the policy
digest is the pinned one, and that the policy reproduces the measured arm's
boundaries on a real cached window. If it fails, the served policy has drifted
from the one that was measured and no number attached to it is valid.

The three-window conformance check is a heavier version of the same idea
(~15 CPU-minutes):

```bash
SC=~/.cache/oc-public .venv-eval/bin/python -m eval.decode_p_conformance
```
