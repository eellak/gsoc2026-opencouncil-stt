# Where inside a 30-second window the errors live

2026-08-18. Harness [`eval/window_position.py`](../../eval/window_position.py), run over the
247-window word-timestamped decode of `exp-2026-08-18-conf-substrate`.
Experiment: `exp-2026-08-18-chunking-aware-decoding`.

## Why this measurement did not exist

`eval/deletion_position.py` had already shown that deletions form a U across the reference
text of an evaluation window: the middle decile band carries 0.35× its share and both edges
are elevated. That is a shape in **text position**, and an evaluation window is well over a
minute long, so a U across the whole window is not evidence about Whisper's 30-second
context at all. A within-30-s effect would produce one U per inference window, not one U
per clip.

Nothing in the project had measured the phase directly, because the reference has no
timestamps.

## What was frozen before looking

In the module constants, before any output was read:

- `BIN_SECONDS = 5.0`, six bins across the window
- `WINDOW_SECONDS = 30.0`
- a reference token takes the time of the hypothesis word it aligns to; a **deleted**
  reference token takes the time of the next aligned hypothesis word, or the previous one
  when it is the tail

The alignment uses the same dynamic program and the same tie-breaking as
`exp_same_stack.sdi`, re-emitting the path rather than only the counts, so the ops counted
here are the ops the frozen scorer would count.

`faster-whisper` with `vad_filter=False` seeks sequentially: it decodes `[seek, seek+30]`,
then advances `seek` to the end of the last segment it completed. The reconstruction
replays exactly that from the emitted segment boundaries, so the bins are real inference
windows and not `t mod 30`.

## The profile

247 windows, 64,555 reference tokens.

| bin | ref tokens | S | D | I | (S+D)/N | D/N | ×best |
|---|---|---|---|---|---|---|---|
| 0-5 s | 11,699 | 741 | 833 | 323 | 0.1345 | 0.0712 | 1.94 |
| 5-10 s | 11,461 | 849 | 455 | 277 | 0.1138 | 0.0397 | 1.08 |
| 10-15 s | 10,837 | 799 | 427 | 294 | 0.1131 | 0.0394 | 1.07 |
| 15-20 s | 10,497 | 755 | 402 | 248 | 0.1102 | 0.0383 | 1.04 |
| 20-25 s | 10,097 | 711 | 371 | 306 | 0.1072 | 0.0367 | 1.00 |
| 25-30 s | 9,964 | 748 | 528 | 276 | 0.1281 | 0.0530 | 1.44 |

## The control that decides whether this is real

The first inference window's 0-5 s **is** the clip's own 0-5 s. A phase effect at the head
could therefore be the clip edge in disguise. Dropping every clip's first and last inference
window leaves 654 interior windows across 247 clips, where no edge is a clip edge:

| bin | ref tokens | (S+D)/N | D/N | ×best |
|---|---|---|---|---|
| 0-5 s | 6,358 | 0.1132 | 0.0513 | 1.88 |
| 5-10 s | 6,582 | 0.1133 | 0.0390 | 1.43 |
| 10-15 s | 6,581 | 0.1149 | 0.0354 | 1.30 |
| 15-20 s | 6,657 | 0.1095 | 0.0392 | 1.44 |
| 20-25 s | 6,631 | 0.0952 | 0.0273 | 1.00 |
| 25-30 s | 6,649 | 0.1197 | 0.0459 | 1.68 |

The head effect **shrinks** from 0.0712 to 0.0513, so a large part of the bad start was the
clip edge and not the window boundary. It does not vanish, and the tail effect survives
intact.

## What it says

The usable region is **5 s to 25 s**. The best five seconds are 20-25 s. Discard the first
five and the last five.

The head has no left context, and the frozen config carries
`condition_on_previous_text: false`, so no context ever crosses a boundary by design. The
tail is cut mid-phrase and the model prefers silence to guessing half a word. Both surface
as **deletions**, which is the endpoint this project treats as primary.

## What it is worth, as a ceiling

The edges hold 21,663 of 64,555 reference tokens, 33.6%. If every second scored at the
5-25 s rate, pooled `(S+D)/N` would fall from 0.11802 to 0.11119, a gain of **0.68 points**.

That is a **ceiling for the overlap arm**, not a prediction. Stitching two passes together
introduces its own errors, and this arithmetic assumes a perfect merge.

## Why silence-aware splitting beats it

Arm P measured −1.14 WER points on the 39-window substrate, which is larger than this
ceiling. Different substrates, so the numbers do not subtract, but the direction is
informative.

Overlap **covers** a weak edge with a neighbour's strong middle. Cutting at silence does
something stronger: it puts the weak edge **on silence**, so there is no speech there to
lose. The edge penalty is still paid, and it is paid on audio with no words in it. It also
costs 1× compute instead of 1.5×.

Overlap earns its place exactly where P cannot cut: continuous speech with no qualifying
gap inside 30 seconds. The two are complementary.

## Caveats

- Deleted tokens are placed at a **neighbour's** timestamp, not their own, because a word
  that was never emitted has no time. Deletion mass is therefore attributed with a bias
  toward the following word.
- This is a descriptive profile with no gate, no bootstrap and no preregistered test. It
  motivates arms; it confirms none.
- Agreement-with-OpenCouncil, as everywhere on this substrate.

## The chain

- [The chunking preregistration](../specs/2026-08-17-chunking-aware-decoding-prereg.md)
- [The confidence-bearing substrate](2026-08-18-confidence-substrate.md), which produced the timestamps
- [What we serve, and the July adapter](2026-08-17-served-config-and-july-adapter.md), which found the same boundaries from the other side
