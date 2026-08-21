# Training-window density audit

Status: **FROZEN 2026-08-18, before the audit output was generated**

This is the CPU-only measurement step for the Wayfinder ticket
“Παράθυρα 30 δευτερολέπτων στην ΕΚΠΑΙΔΕΥΣΗ: η πυκνότητα ομιλίας είναι εκτός
κατανομής”. It does not train a model and does not inspect either sealed holdout.

## Inputs

- Existing control-arm training rows: `data/hf-dataset/public/train.parquet`.
  It must contain 28,967 rows and reproduce the 22.47 raw-span hours reported for
  mixture arm A; otherwise the audit fails closed.
- Existing inference cache:
  `~/.cache/oc-public/conf-substrate-2026-08/decode-rw.json`, plus the corresponding
  WAV headers under `~/.cache/oc-public/bench_windows/`. These are the already-read
  247 benchmark windows, not either sealed holdout.
- Existing packing algorithm: `eval/hf_export/build_packs.py`. The feasibility
  simulation is in memory and writes no pack text or audio.

The result committed to the repository contains aggregates only. Transcript text,
row identifiers and audio do not enter the result.

## Frozen definitions

Whisper's encoder window is 30 seconds. The existing trainer cuts on
`start_adj`/`end_adj` when present, falls back to `start`/`end`, adds another 0.2
seconds on each side, starts the waveform at encoder time zero, and right-pads it
to 30 seconds.

For every training row:

- **raw speech-span proxy** = `end - start`. This is the source of the inherited
  “2.79 seconds of speech” statement. It is a timestamp-span proxy, not frame-level
  VAD speech.
- **aligned span** = `end_adj - start_adj`, falling back to the raw span.
- **intended clip audio** = from `max(0, aligned_start - 0.2)` through
  `aligned_end + 0.2`, capped at 30 seconds. The pod-built clip cache is not present
  locally, so this is before the trainer's final clamp to the source recording end.
- **encoder audio occupancy** = intended clip audio / 30.
- **digital-padding share** = 1 minus encoder audio occupancy.
- **position of the raw speech proxy** = the raw `[start, end]` interval expressed
  relative to intended clip start. The clip is never centred; it begins at encoder
  time zero and all remaining padding is on the right.

Duration histograms use fixed edges
`[0, 1.5, 3, 5, 10, 15, 20, 25, 30]` seconds. Position profiles use the six fixed
five-second bins `[0,5), ..., [25,30]`. For audio support, a bin receives its
overlap in seconds. For the raw speech proxy, it receives overlap of the proxy
interval in seconds.

For inference, the audit reuses `eval.window_position.inference_windows` unchanged.
For each reconstructed window:

- **available source audio** = overlap of `[seek, seek+30]` with the WAV duration;
- **source-audio occupancy** = available source audio / 30;
- emitted-word phase is descriptive only and comes from the cached adapter word
  timestamps. It is not treated as ground-truth speech or used by the feasibility
  gate.

The comparison is therefore valid for *audio support inside the encoder*. The two
speech-position proxies are reported separately and are not subtracted from each
other: training uses annotation spans; inference uses model-emitted words.

## Dense-arm feasibility simulation

Run the existing greedy packer unchanged: order selected utterances within each
meeting, use the aligned spans, insert 0.4 seconds of labelled synthetic silence,
and cap a pack at 29.5 seconds. This tests whether the already-used rows can form a
dense arm; it does not claim that the arm would improve WER.

The arm is **construction-feasible** only if all hold:

1. the existing packer's accounting/timing gates pass, including no duplicate
   utterance, at most 0.5 dropped speech-hours, monotone spans and no pack over 30 s;
2. at least 80% of packs are 20--30 seconds long;
3. median pack duration is at least 20 seconds;
4. mean encoder audio occupancy is at least 75%.

No WER, deletion or model-quality conclusion is permitted from this audit. Any GPU
screen remains a separate, explicitly approved experiment under
`docs/decisions/training-evidence.md`.

