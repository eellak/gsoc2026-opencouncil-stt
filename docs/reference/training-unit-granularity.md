# Training unit: single utterances or larger segments?

Status: **answer for the open question** (2026-06-23). The build currently slices
one clip per corrected utterance; the meeting asked whether that's right or whether
we should use larger chunks (context windows / whole speaker segments). Short
answer below, then the reasoning from the literature and our own data.

> **TL;DR (GR):** Όχι σκέτα utterances 1-3 δευτ. Το Whisper «βλέπει» πάντα παράθυρο
> **30 δευτ.** — εκπαίδευση μόνο σε πολύ μικρά κομμάτια χαλάει το long-form
> (segmentation, timestamps, hallucinations). Πρόταση: φτιάχνουμε clips ~**20-30
> δευτ.** ενώνοντας **γειτονικά utterances μέσα στο ίδιο speaker segment**, με όρια
> σε **σιωπή/VAD** (όχι μέσα σε λέξη), μικρό padding (~±0.2 δευτ.), χωρίς overlap
> για να μην έχουμε duplicates. Το διορθωμένο utterance είναι ο στόχος· τα γειτονικά
> μπαίνουν ως context (από το ήδη-reviewed κείμενο).

## Recommendation

Build training examples at **~20–30s**, by concatenating consecutive utterances
**within one speaker segment** up to that length — not one clip per bare utterance.
Keep the boundaries on silence (VAD), pad a little (~±0.2s), and don't overlap
windows so the same utterance never lands in two examples.

This isn't a free win — it's more pipeline work (concatenate the target texts, fix
the joined timestamps, dedup) — so if we were only ever going to feed pre-cut
single utterances at inference, short clips would be fine. We're not: the product
transcribes whole meetings. That's the deciding factor.

## Why bare utterances are the wrong default

**Whisper's input is fixed at 30 seconds.** Every clip is zero-padded or truncated
to exactly 30s before the encoder. So a 2-second utterance is 28 seconds of
silence with a sliver of speech — you're spending the model's window on padding and
teaching it almost nothing about how words flow into each other. The clip *content*
length is the real choice, and very short is wasteful.

**Short-only fine-tuning measurably breaks long-form.** This is the trap, and it's
well documented. Whisper transcribes a long recording by sliding its 30s window
using its own predicted timestamps and the previous text as a prompt. If you
fine-tune only on short, untimed clips, the model "forgets" how to produce
timestamps and consume previous-text hints, and long-form decoding degrades —
coarser segmentation, repetition loops, hallucinations — *even while per-clip WER
goes down*. Timmel et al. (2024) saw sentence-level fine-tuning push long-form SubER
to ~51 (worse than the 30.6 of the un-fine-tuned base); rebuilding the training data
as ~30s timestamped segments recovered it to ~41.6 and improved BLEU. The ivrit.ai
team independently hit the same wall and the same fix.

**Our errors are partly *caused* by tight boundaries.** Christos's point in the
meeting matches the WhisperX finding directly: many ASR errors happen because the
audio was cut just before/after a word, so the model never heard the full sound.
Cutting one clip exactly per utterance bakes that problem into training. Putting
boundaries on silence and giving a little context is the standard mitigation.

## What the canonical recipes actually do

- **HuggingFace "Fine-Tune Whisper for Multilingual ASR" (Sanchit Gandhi)** trains
  on Common Voice sentence clips as-is and ignores timestamps and long-form. It's a
  clean short-form WER recipe — fine as a *baseline*, but it optimizes the thing we
  don't only care about. Treat it as the floor, not the target.
- **ivrit.ai "Fine Tune Whisper the Right Way"** uses ~27s average timestamped
  segments plus a chunk of short untimed text, keeps timestamps on when available,
  and includes previous-text context **~50% of the time**. Their explicit warning is
  the short-only failure mode above.
- **WhisperX** places ~30s chunk boundaries on low-energy (silent) regions rather
  than fixed cuts, specifically to avoid mid-word truncation and the hallucinations
  it causes.

Sources: HF blog (huggingface.co/blog/fine-tune-whisper); Timmel et al. 2024,
*Fine-tuning Whisper on Low-Resource Languages* (arXiv 2412.15726); ivrit.ai
(ivrit.ai/en/2025/02/13/training-whisper); WhisperX (arXiv 2303.00747); Whisper
paper (arXiv 2212.04356).

## How this maps onto our data

Our corrections arrive **per utterance** from the review UI — that's the unit a
human edited, and it stays the unit of the *target correction*. The change is the
**audio window around it**: instead of cutting `[start, end]` of the one utterance,
cut from a few utterances before to a few after (within the same speaker segment),
up to ~30s, with the full (reviewed) text of that window as the target.

Practical rules so this doesn't backfire:

- **Stay inside one speaker segment.** The meeting JSON already groups utterances
  under `transcript[] → utterances[]` per speaker turn; a speaker segment is the
  natural ~tens-of-seconds unit and avoids splicing two voices into one clip.
- **No overlapping windows.** If utterance *N* is the corrected one, don't also
  emit a window centered on *N±1* that re-includes *N*. Walk segments in
  non-overlapping ~30s tiles, or one window per corrected utterance with a dedup
  pass keyed by utterance id. (The auto-research build already does composite-id
  dedup — extend it to spans.)
- **Only trust the neighbour text if the meeting is reviewed.** A context window
  pulls in neighbouring utterances as target text; those neighbours are only clean
  if the meeting passes the [trust cutoff](../specs/meeting-trust-cutoff-plan.md).
  For unreviewed meetings, keep the single corrected utterance only.
- **Keep timestamps where we can.** If we want long-form robustness we should train
  with timestamp tokens on the segment, not strip them.

## Open / to validate

- Exact target length and the short-vs-long mix ratio (literature uses ~30s
  segments + some short text; the 50% context-prompt probability is a common value,
  not a proven optimum). Sweep this on the mini PC once the val set is bigger.
- Whether to include timestamp tokens in the first large-v3 run or add them later —
  costs alignment work; decide alongside the build.
- This interacts with the dataset-size question (mentor): larger segments mean
  fewer, richer examples — recount hours after switching from utterance clips.
