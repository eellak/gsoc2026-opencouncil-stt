# Chunking-aware decoding — preregistration

Status: FROZEN 2026-08-17, before any arm was decoded
Experiment: `exp-2026-08-17-chunking-aware-decoding` (to be opened OPEN)

## The question

Do our deletions come partly from **where the audio is cut**, rather than from the
weights?

## Why now

Two facts landed today.

**Measured, `eval/deletion_position.py`.** Deletions inside the 39 frozen windows
form a U across the reference: the middle decile band carries **0.35x** the
expected share, the last decile **1.73x**, and both edges are elevated. The shape
is **identical in `artifact-adapter-fixed` and in RUN2 stage-2** (tail enrichment
1.60 vs 1.54), so it is a property of how audio is cut and decoded, not of either
training run. The same script's preregistered 2.0x tail bar was NOT met, which is
why this spec is about boundaries generally and not about premature EOS.

**Read from indexed source, `schemalabz/opencouncil-tasks` via DeepWiki.**
Production `splitAudioDiarization` cuts audio at silences taken from the
diarization timeline, preferring gaps >= 5 s and falling back to >= 0.5 s, capped
at `MAX_TRANSCRIPTION_SEGMENT_DURATION_SECONDS = 15 * 60`, with **no overlap**;
`combineTranscripts` then concatenates with a timestamp offset and **no
de-duplication**, which is sound precisely because the cuts sit in silence. There
is **no 30-second audio chunking anywhere** — `UTTERANCE_MAX_DURATION_SECONDS = 30`
governs text only — and **no Whisper path in the repo at all**; ElevenLabs Scribe
is the only ASR.

So the pipeline is built on the assumption that the recogniser swallows a
15-minute chunk. Scribe does. Whisper cannot: it works in 30-second windows. Drop
our adapter into step 4 and faster-whisper re-cuts that chunk on its own, blind to
the silences the pipeline already computed, with no overlap and nothing
reconciling the boundaries.

**And this has never been tested.** `exp-2026-08-12-decode-ablation` moved four
thresholds — no-speech, log-prob, temperature, compression ratio — and shipped
nothing. It never touched segmentation. Our frozen config carries
`vad_filter: false`, so every 30-second boundary is placed blind, and
`condition_on_previous_text: false`, so no context crosses one either.

## Arms

All arms: `artifact-ct2-fixed`, local CPU int8, the 39 frozen evaluation windows,
the frozen evaluation normalizer, common random numbers. Only the named field
changes; everything else stays at the frozen CONTROL config.

| arm | change |
|---|---|
| **A** | control, the frozen config as-is (cached, `decode-ablation/eval-A.json`) |
| **V** | `vad_filter=True` at faster-whisper defaults — boundaries placed at detected silence instead of blind |
| **P** | we pre-split each window ourselves at detected silence into pieces of at most 30 s, decode each piece independently, concatenate with a timestamp offset — the production cut policy, applied at the scale Whisper can actually swallow |

Arm P mirrors `splitAudioDiarization`: prefer the longest silence available, never
cut where no silence was found, and add no overlap.

### Amendment 1, 2026-08-18 00:45 — two arms added, still before any number

No arm had produced a score when this was written: arm V was 4 of 39 windows in and
its cache had not been read, arm P had not started, and `score` had never been run.
The gate above is unchanged and applies to these two identically.

| arm | change |
|---|---|
| **Π** | pad the window: at most **25 s of speech** per window, the remaining ~5 s left as digital silence, so the region where deletions concentrate contains nothing real |
| **Ε** | overlapping windows, **keep only the middle**: advance by a stride shorter than the window and take each window's central region, so every second of audio is scored from the middle of some window |

Both come from the same measurement as the rest of this spec. The middle decile band
carries 0.35x its share of deletions while the edges are elevated, so Π removes real
speech from the weak region and Ε covers that region with a neighbour's strong one.
Ε is the stricter test of the same idea: Π discards the edges, Ε reuses them.

Π also tests a **train/decode mismatch** directly. Training clips average 2.79 s of
speech inside a 30 s window — about 91% padding — so the model has never been trained
on a window that is full of speech to the last sample, which is exactly what decoding
hands it. If Π wins, that mismatch is implicated and the next training run's
segmentation follows from it.

Ε needs boundary-merge logic that production does not have today
(`combineTranscripts` concatenates with no de-duplication, which is correct only
because its cuts sit in silence). That logic lives in our serving layer, not in
`opencouncil-tasks`.

## Primary gate, frozen before any number is seen

Primary metric is the **deletion rate**, because that is the failure this project
guards against and the one the U-shape points at. An arm passes only if all four
hold, on a meeting-clustered paired bootstrap:

1. deletion-rate delta vs A is **negative**, and its 95% upper bound is **< 0**
2. WER delta 95% upper bound **<= +0.002**
3. insertion-rate delta 95% upper bound **<= +0.002**
4. the deletion-rate sign does not reverse under leave-one-window-out

A lower WER with a higher deletion rate is a **failure**, not a trade.

## What this cannot show

The 39-window harness measures **agreement with OpenCouncil's published text**, not
fidelity to audio. An arm that recovers real speech the published text omits will
be scored as an insertion and punished. That asymmetry is known
(`exp-2026-08-17-insertion-fidelity`: 23.7% of our scored insertions sit on words
a human heard), and it means a passing arm is credible while a failing arm is
ambiguous.

Single decode per arm; no seeds are involved, since no training happens here.

## Cost

Two new arms x 39 windows on local CPU. No GPU, no money, no external API.
