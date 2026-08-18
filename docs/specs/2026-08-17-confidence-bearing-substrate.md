# A confidence-bearing fusion substrate

Status: DRAFT, not yet run
Experiment: `exp-2026-08-18-conf-substrate` (to be opened)
Date: 2026-08-17

## Why

Two arbitration ideas the user selected — routing questions by confidence, and
putting confidence numbers in the LLM's prompt — both need a per-word confidence
attached to the tokens **W is actually built from**. We do not have that, and the
gap is measured, not suspected.

W's substrate is plain text. `hyp[provider]` in the benchmark report is a string:
no segments, no probabilities. Every confidence this project holds was produced by
a *different* decoding pass than the text in W:

| source | why it does not join |
|---|---|
| Soniox per-word | `stt-rt-v4`; W is built on `stt-async-v5`. `artifact-soniox-rt-tokens-2026-08-16` records the prohibition as a caveat. |
| adapter per-word | `word_timestamps=True` re-run, 102/247 windows, and the setting changes the transcript. |
| adapter per-segment | `word_timestamps=False` re-run, 133/247 windows, frozen config. |

The last row looked usable, so it was measured: **0 of 133 windows reproduce the
benchmark text; token agreement 92.1%, worst window 59%.** This matches what
`exp-2026-08-16-adapter-confidence` already recorded from the other direction —
cached RunPod GPU `wt=False` against local CPU int8 `wt=False` is 0.0926 symmetric
WER with 2 of 102 windows identical.

Attaching confidence across that gap is not merely lossy. The columns F1 arbitrates
are the *unstable* ones, so they are enriched inside the 8% that does not match.
Coverage would be low **and biased toward the questions that matter**, which is the
measurement error this project has paid for before.

The fix is not a better join. It is to stop needing one: decode once, and keep the
text and the numbers from the same pass.

## What this produces

A substrate whose adapter row carries its own per-word probability, and a baseline
measured on it. After this, confidence gating and confidence-in-prompt are ordinary
experiments instead of grafts.

This is a **new baseline, not a delta**. The re-decoded adapter row will differ from
the benchmark's by roughly 8%, so the resulting W is a different W. It must be
reported beside the existing W = 0.10046, never as an improvement on it.

## Stages

### 1. `decode`

247 windows — the fusion substrate's window set, with the 16 sealed evaluation
windows excluded exactly as `load_substrate` excludes them.

- model: `artifact-ct2-fixed` (`/home/harold/oc-asr-serve/ct2-fixed`), `model.bin`
  sha256[:16] verified as `8a1a3b257d0c1bdb` before decoding
- config: the frozen CONTROL config with `word_timestamps=True` and nothing else
  changed. This is the existing `RW` arm's semantics.
- local CPU, int8, thread count recorded in the output

Per window store `text`, and per segment `start`, `end`, `text`, `avg_logprob`,
`no_speech_prob`, `temperature`, and `words` as `{w, s, e, p}`. Also store config,
model digest, environment, thread count, seed, and code sha.

Resumable: write incrementally, skip windows already present, and **refuse to
extend** a cache written under a different model, config, or environment rather
than silently mixing passes.

### 2. `build`

Assemble the fusion substrate with the adapter row taken from stage 1 and the other
two rows unchanged from the benchmark report. Align with the existing MSA code.

`eval/controlled_eval/msa.py` must not be edited. Assert the alignment cache key is
still `align_65b1c4d64618a429.json`; a changed key means the 18 MB cache was
invalidated and the stage must fail loudly.

Emit, per aligned column, the adapter's confidence for the token it contributed:
the minimum `p` over the words backing that token. Columns the adapter did not
contribute to get `None`, never a default.

### 3. `measure`

The W-conf baseline on the frozen evaluation normalizer: WER, deletion rate,
insertion rate, each with a bootstrap confidence interval, reported next to the
existing W as a separate baseline.

Also report how many of the 6,645 `exact_2_of_3`-equivalent columns now carry an
adapter confidence, since that number is the ceiling on what confidence gating can
reach.

## Test contract

1. the resolved config equals frozen CONTROL except `word_timestamps=True`
2. `decode` refuses to extend a cache written under a different model, config, or
   environment
3. a two-window smoke emits `words` for every segment, with `0 < p <= 1` for every
   word
4. the MSA alignment cache key is still `align_65b1c4d64618a429.json`
5. per-column confidence is defined for every column the adapter contributed to and
   `None` for every column it did not

## Boundaries

- `eval/controlled_eval/msa.py` must not be touched.
- The 16 sealed windows stay sealed.
- No transcript text or audio in git. All output under
  `~/.cache/oc-public/conf-substrate-2026-08/`.

## Cost

Roughly 5 to 8 hours of local CPU. No GPU, no money.

## What this does not fix

Soniox confidence still does not join — its numbers exist only on `stt-rt-v4` text.
After this work, confidence gating runs on **one of three voters**.

`exp-2026-08-16-adapter-confidence` also fixed a ceiling: deletions are 41.1% of
edit operations in the scored region, and a per-word confidence cannot attach to a
word that was never emitted. Confidence is a route to substitutions, not to the
deletion problem.
