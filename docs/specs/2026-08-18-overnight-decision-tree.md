# Overnight decision tree, 2026-08-18

Written 00:50, before any chunking arm was scored, so that the morning's move is
decided by evidence and not by whatever the first number happens to look like.
Deadline for issue #3 is **2026-08-23**.

## Running when this was written

| job | state | ETA |
|---|---|---|
| `conf_substrate decode` (247 windows, word-level confidence) | 137/247 | ~04:10 |
| chunking arm **V** (`vad_filter=True`) | 4/39 | ~02:00 |
| chunking arms **P**, **Π**, **E** | queued behind V | +2 h each |
| Codex `6b95779f` — sol reviewing `chunking_decode.py` | running | — |
| Codex `5f2c5e6b` — implementing arms Π and E | running | — |
| `served_config decode --arm RW` | **PAUSED** (`kill -CONT 1760027`) | — |

## The gate, restated

From [the preregistration](2026-08-17-chunking-aware-decoding-prereg.md). An arm
passes only if the deletion rate falls with its 95% upper bound below zero, WER rises
by at most 0.002 at the upper bound, insertions likewise, and the deletion sign
survives leave-one-window-out. **Lower WER with higher deletions is a FAIL.**

## Branch 1 — an arm passes

Most likely on Π or E, since those attack the measured U directly.

1. Re-score on the **247-window** substrate, not just the 39, to check the effect is
   not a property of the small harness.
2. Apply the winning chunking to **W's adapter row** and re-measure W. W is the
   deliverable; a decode fix that does not survive fusion is not worth shipping.
3. Write it into the serving layer (`oc_asr_server.py`), because this is an
   **inference** change: no GPU, no retraining, and it can ship before the deadline.
4. Only then consider a training run whose segmentation matches the winning decode
   policy. Π winning is the strongest evidence for that; it directly implicates the
   2.79 s-of-speech-in-a-30 s-window mismatch.

## Branch 2 — every arm fails the gate but the deletion point estimate moves down

Underpowered, not refuted. 39 windows is small.

1. Re-run the best arm on the **247 windows**, where the token count is ~6x larger.
2. Do not touch the gate. Report both.

## Branch 3 — every arm fails and nothing moves

The U-shape is real but not caused by where the cut falls.

1. Close the experiment CLOSED-negative. It is a cheap, clean negative and it retires
   a hypothesis three separate sources found plausible.
2. Fall back to the **A-101 control run** (below), which does not depend on any of
   this.
3. Do not spend more CPU on segmentation.

## Branch 4 — arm P reports that no silence exists inside 30 s

Watch this number **before** any WER. If council speech has no qualifying gaps at
that scale, the production cut policy cannot apply here at all, and arms P and E are
uninterpretable while Π is unaffected (it needs no silence, it manufactures one).

Then: judge on Π alone, and record that the production policy does not transfer to
30-second windows. That is a finding worth writing down on its own.

## The training run, independent of all four branches

Codex reached this twice from different starting points: the single most valuable
training run is **not a new idea, it is a control**.

**Retrain `artifact-adapter-fixed` exactly, changing only the seed to 101.**

Same manifest, same r=32 q/v, same lr 1e-4, same 2 epochs, same 7,242 updates, same
base checkpoint, same GPU type as whatever is used next, no checkpoint selection from
the evaluation set.

Why it is worth a night: RUN2 differs from `adapter-fixed` in **seed, data, optimizer
steps and GPU** at once, so that comparison is uninterpretable forever as it stands.
One A-101 gives RUN2 a same-seed partner and turns a wasted experiment into a readable
one. It also gives us our **own** measurement of seed spread, instead of borrowing 2.1
points from another experiment.

Preregistered criterion for RUN2 vs A-101, fixed before decoding: deletion delta
<= -0.005 with a 95% upper bound below zero, WER delta upper bound <= +0.002,
insertion delta upper bound <= +0.002.

Cost: one A40 at $0.44/h. **A watchdog with a hard deadline is armed before anything
is uploaded, and the pod ID is recorded** — the pod bills from creation, and an
unattended pod is the one failure mode that bills all night.

## Explicitly not tonight

- **HParl / external hours.** Licence is CC BY-NC 4.0 at CLARIN 1602 and the HF
  mirror cannot erase the NC. Blocked on permission, not on engineering.
- **Full fine-tuning.** Data scale is closed on evidence: ~1300 h buys ~0.5 points.
  Our problem is label quality, not model capacity.
- **More LLM arbitration.** Measured worse twice, and its self-reported confidence
  saturates at 98-99 in 81% of answers, so there is no dial to turn.
- **Anything that needs a human at the MacBook.** The Grok browser needs its page
  reloaded before that CLI works again.
