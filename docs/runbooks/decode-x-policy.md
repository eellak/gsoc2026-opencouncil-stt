# Running the arm X seam-repair decode policy

Capability: `cap-decode-x-local`. Artifacts: `artifact-decode-policy-x` (the frozen
config), `artifact-ct2-fixed` (the weights). Produced by
`exp-2026-08-20-seam-repair`; preregistration
[here](../specs/2026-08-20-seam-repair-prereg.md).

**Status: exploratory.** This is packaged so it *can* be run over a benchmark, not
because a benchmark has endorsed it. Read the ledger record before quoting a number
from it.

## Smoke check (read-only, instant)

```bash
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_x config
```

Prints the whole policy and its `policy_sha256`. Expect
`3af821c64bfb179c...`. A different hash means the policy changed and any number
recorded against the old hash no longer applies.

## Decode one file

```bash
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_x transcribe audio.wav
SC=~/.cache/oc-public .venv-eval/bin/python -m serve.decode_x transcribe audio.wav --json
```

`--json` adds word timestamps, the boundary list with the reason each boundary was
chosen, and the counters. Without it you get the text only.

## From Python, over many files

```python
from serve.decode_x import DecodeX

decoder = DecodeX.load()            # verifies the CT2 model hash first
for path in paths:
    result = decoder.transcribe(path)
    ...                             # result.text, result.words, result.boundaries
```

`DecodeX.load()` is the expensive call — load once, decode many. Overriding
`model_dir` skips the hash verification, so anything decoded that way must record
which model produced it.

## What the boundaries tell you

`result.boundaries` gives every cut and why it was chosen:

| `kind` | meaning |
|---|---|
| `vad_silence` | a reported silence of at least 0.5 s — the ordinary case |
| `probability_valley` | no silence existed, so the cut went to the lowest-mean Silero valley of at least 128 ms below 0.35 within the last 3 s |
| `blind` | no silence and no valley — genuinely continuous speech |

A `blind` boundary is **not** the failure it is in arm P: the decoder still hears
0.5 s before and 2.0 s after every seam, so nothing is truncated mid-word. On the
pilot windows, blind seams disagreed *less* than silence seams (0.551 vs 0.608).

If a transcript looks wrong somewhere, find the nearest boundary and its kind. That
is the point of recording them.

## Cost

Local CPU int8, 16 threads on the minipc: roughly **145 s of wall time per 150 s of
audio**, so about real time. No GPU, no money, no external API.

## Before trusting it on new data

Two gaps, both real:

1. **Never run on contiguous whole-meeting audio.** Every measurement is on pre-cut
   ~150 s windows, so the policy has never chosen a boundary near a real meeting's
   start or end.
2. **Run the drift test first.**
   ```bash
   SC=~/.cache/oc-public .venv-eval/bin/python -m pytest eval/tests/test_decode_x_policy.py -q
   ```
   It asserts the served constants equal the measured ones and that the policy
   reproduces the measured arm's boundaries on a cached window. If it fails, the
   served policy has drifted from the one that was measured and no number attached
   to it is valid.
