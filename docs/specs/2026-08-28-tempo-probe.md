# Spec: does feeding Soniox slower recover words lost to crosstalk?

2026-08-28. Record: `exp-2026-08-28-tempo-probe`. Status when written: **preregistered,
no number seen**. Plan reviewed by Codex `db4d8ac8`.

## The question

Harold's question, from a friend's suggestion: council speakers talk over each other and
fast, and the ASR drops words there. If the same audio is played to Soniox **slower**,
does it resolve more of that speech?

This is a **direction pilot**, not a decision. The only human-referenced overlap material
this project has is 15 gold cells over 5 meetings, holding 18 linked episodes with **217
certain** human-transcribed words and 57 more in blocks the listener flagged uncertain.
That cannot settle anything; it can say whether a bigger measurement is worth building.

## What is actually being varied — two different things

| | what changes | what it tests |
|---|---|---|
| **time stretch** (`atempo`) | the acoustics: the same words occupy more seconds | whether the recogniser resolves more syllables when they are longer |
| **feed pacing** (`--speed`) | nothing about the audio; only how fast the client pushes bytes into the realtime socket | whether the streaming model does better with more wall-clock per second of audio |

They are confounded in the naive experiment and are separated here, because if the gain
comes from pacing there is no reason to distort audio at all.

## Arms — five, all through one code path

| arm | audio | fed at | note |
|---|---|---|---|
| `raw` | untouched wav | 1.0× | tells us whether `atempo=1.0` is really a no-op |
| `proc10` | `atempo=1.0` | 1.0× | **the control**; every comparison is against this |
| `t085` | `atempo=0.85` | 1.0× | conservative stretch |
| `t070` | `atempo=0.70` | 1.0× | aggressive stretch |
| `pace070` | untouched wav | 0.7× | pacing only, acoustics unchanged |

`atempo` is WSOLA (`libavfilter/af_atempo.c`), not a phase vocoder, and it preserves
pitch. Its specific risk in crosstalk is that WSOLA aligns to **one** best waveform
match; with two competing speakers it can follow the louder one and smear the quieter.
That is a reason to distrust a `t070` win, not a reason to skip it. `rubberband` is
available in this ffmpeg build and is held back as a **second-implementation check** on
discordant episodes only, if anything survives.

Endpoint detection is OFF in every arm (`file_transcribe.py:118`), because Soniox
endpointing keys on pauses, and stretching changes every pause length. Sample rate,
channels and encoding are held identical across arms.

**Model: `stt-rt-v5`, requested by name.** Until 2026-08-28 this project requested
`stt-rt-v4`, which Soniox removed on 2026-06-30 and routes to v5; every run we have made
is August 2026 and was therefore served by v5 under a v4 label. Requesting v5 explicitly
changes nothing that answers and makes the record true.

## Repetition: three interleaved cycles

Streaming ASR is not deterministic. One control-vs-control repeat is a single draw, not
an estimate of variance (Codex `db4d8ac8`), so **every arm runs three times**, in a
randomized order within each cycle, seed `20260828`, all cycles inside one short window
so server-side drift cannot separate the arms. 15 cells × 5 arms × 3 cycles = 225
sessions of ~35 s each (50 s for the two slow arms). The model is free, so the cost is wall-clock only.

## Scoring

**Episodes, not serial WER.** Two people talking at once have no single correct word
order, so a linear WER on the mixture is meaningless. Blocks are linked through their
`ov_with` field into connected components; each component is an **episode** with a time
span and a bag of human words. A hypothesis token belongs to the episode if its
(rate-corrected) midpoint falls inside the span. Matching is one-to-one on normalized
words (`eval/scoring.py:greek_normalize`), so a repeated word cannot be double-counted
and one hypothesis token cannot satisfy two speakers.

**Timestamps map back by MULTIPLICATION.** At `atempo=r` an event at original time `t`
is returned near `t/r`, so `original = returned × r`. An earlier draft of this spec had
it inverted, which at 0.7 would have placed every word at twice its real time. The
harness asserts the mapping against the cell duration before scoring anything.

| quantity | role |
|---|---|
| certain-reference overlap-episode recall | **primary**, arm − `proc10` |
| the same on `text_unc` blocks | reported separately, never pooled |
| recall outside the episodes, same cells | collateral guard |
| hypothesis words matching nothing | **descriptive only** — see below |

**Why unmatched words cannot be an insertion guard.** The human reference was written
while listening at normal speed. A word the slowdown genuinely recovers may be a word
the human never wrote, and it would score as an insertion — the guard would reject
exactly the phenomenon under test. So unmatched words are counted, not penalised, and
the ones that differ between arms go to blind adjudication (§ below).

## Decision rule, frozen before any number

`ADVANCE` requires all three:

1. the arm's overlap-episode recall gain over `proc10` exceeds the **within-arm
   envelope**, the spread of that arm's own three cycles;
2. at least 4 of the 5 meetings move in the same direction (the substrate is 5, not 6;
   corrected here before any number was seen, and the code's rule -- `max(4, 2N/3)` --
   is unchanged);
3. non-overlap recall falls by no more than 1.0 point.

Anything else is `STOP`. A gain that reverses when the same episodes are re-stretched
with `rubberband` is reported as **transform-specific**, not as evidence that slowing
helps.

## The adjudication, which is Harold's job

The pilot can only measure recovery of words already in the gold set. Whether the
slowdown surfaces **new** true words needs a human, and only for the words that differ
between arms. A page collects every word present in one arm and absent from another,
strips the arm labels, and plays the original audio; Harold marks each **supported /
unsupported / cannot tell**. That is a few dozen judgements, not a re-transcription.

## What this pilot cannot do

- It cannot decide. 217 certain words over 5 meetings, three cycles, is a direction.
- It cannot separate speakers. Slowing does not raise SNR or unmix sources; if the loss
  in crosstalk is source competition, no tempo helps and the prior says so.
- It says nothing about our own adapter. Soniox is a hosted streaming model; a gain here
  would be a **witness-quality** result — better labels for mining — not a decode change
  we can ship.
