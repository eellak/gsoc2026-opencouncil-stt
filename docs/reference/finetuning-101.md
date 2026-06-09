# Finetuning STT 101 — OpenCouncil edition

> Provenance: AI-drafted briefing, handed in by Angelos (2026-06). Treat the
> recommendations here as **starting points**, not decisions. Anything we
> actually commit to lives in [../decisions/](../decisions/_index.md); open
> questions are tracked there too. This file is the long-form reference the
> short decision entries point back to.

A briefing for anyone working on (or alongside) the speech-to-text finetuning
project: what finetuning is, how to think about our dataset, what to measure,
and what a training run actually costs.

> Scope: this is about **STT** (transcription), not TTS. Goal: adapt an existing
> model (Whisper) to Greek council audio so it makes fewer mistakes on the words
> that matter — names, places, jargon, acronyms, law numbers.

## Why we're doing this

Every council meeting on OpenCouncil goes through a generic STT provider, then
Eliza and Thanos spend hours per week fixing the transcript before publishing.
The errors cluster, not random:

- **Proper nouns**: council members, mayors, neighbourhoods, streets
- **Acronyms**: ΚΕΔΕ, ΟΠΕΚΑ, ΑΕΠΙ, ΦΕΚ, …
- **Numbers**: law numbers, decision numbers, budget figures
- **Formal/legal Greek**: phrasing rare in general training data
- **Speaker variability**: older speakers, regional accents, poor mics

A model finetuned on our own data should do much better on exactly these,
because we have what nobody else has: hundreds of hours of council audio paired
with human-corrected transcripts. The 2027 vision target is **10× better and
cheaper transcription than the current provider**, open-sourced, trained on 500+
hours.

## How STT finetuning works

Supervised learning. Each example is a pair: **audio** (16kHz mono, ~2–30 s) +
**text** (expected transcription, with punctuation and casing). The model
maximises the probability of the text given the audio — same loss as Whisper
pretraining, on our data.

Whisper always sees a fixed **30-second window**: shorter clips are
silence-padded, longer ones truncated. So our 2–10 s utterances are fine. The
text is tokenised with special prefix tokens:

```
<|startoftranscript|><|el|><|transcribe|><|notimestamps|> Καλησπέρα σας…<|endoftext|>
```

`<|el|>` = Greek. `<|notimestamps|>` because we already have utterance
boundaries from our pipeline.

**Utterances vs longer chunks.** Two caveats with short utterances:

1. Long-range context (a name mentioned 20 s ago) isn't taught — mostly OK
   because we run inference utterance-by-utterance anyway.
2. Inference-time prompting from prior text isn't taught — so set
   `condition_on_previous_text=False` at inference.

A reasonable mix: **~80% utterances as-is + ~20% longer chunks** (concatenated
consecutive same-speaker utterances up to ~25 s).

## Our dataset

**Two sources of labelled audio:**

1. **Corrected utterances** — a human changed what STT produced. The hard cases.
2. **Non-corrected utterances from fully-reviewed meetings** — a reviewer saw
   them and didn't touch them. Silent positives; treat as ground truth.

> **Critical caveat:** only use #2 from meetings that were *fully* reviewed. If
> reviewers can skip/skim, "non-corrected" might mean "not yet looked at."
> Verify in the pipeline before bulk-including. → open question in
> [../decisions/data.md](../decisions/data.md).

**The correction-bias trap.** Train only on corrections and you get a model good
at the hard cases and possibly worse at the easy ones — every example told it
"the obvious transcription is wrong." Suggested mix:

- **~30–50% high-confidence corrections** (include-flagged)
- **~50–70% non-corrected utterances** from fully-reviewed meetings

**How much data.** The unit is **hours of audio**, not utterance count
(~5 s/utterance → ~600–1000 utterances ≈ 1 h).

| Tier | Hours | Utterances | What we get |
| --- | --- | --- | --- |
| Minimum viable | 5–10 h | 4k–8k | Noticeable WER drop on jargon, names, acronyms |
| Solid | 30–50 h | 20k–35k | Strong domain adaptation; closes most of the gap |
| Diminishing returns | 100–200 h | 70k–140k | Marginal gains unless chasing SOTA |
| Vision target | 500 h | ~350k | SOTA on Greek council audio |

First useful checkpoint should come from 30–50 h; don't wait for 500 h.

**Train/val/test split — the most important rule: split by meeting and speaker,
not by utterance.** Random utterance shuffling puts the same speaker and meeting
in train and test, so you measure memorisation, not generalisation. Concrete:

- Split by meeting first: ~80% train, ~10% val, ~10% test.
- Hold out specific speakers entirely (test includes voices absent from train).
- Hold out one entire municipality if affordable — answers "does it work on a
  new municipality we onboard tomorrow?", the business question.
- Stratify val/test by error category to match train.

→ canonical decision: [Split by whole meeting](../decisions/data.md#2026-06---split-trainvaltest-by-whole-meeting-not-by-utterance).

## What to measure

**Plain WER is misleading** — it weights every word equally. Getting "και" vs
"κι" wrong: nobody cares. Getting a council member's name or a street wrong:
breaks search, speaker stats, notifications. A model can shave 2 WER points on
common words while regressing on named entities.

Better metrics, ordered by usefulness:

1. **NE-WER (Named-Entity WER)** — the one to actually optimise. WER over named
   entities only (person/party/neighbourhood/street/org names, law/decision
   numbers). We already have the entity list in Postgres.
2. **WER stratified by error category** — using the review-tool taxonomy.
   Answers "did finetuning fix what we built it to fix?"
3. **Diarization-aware WER (cpWER)** — WER conditional on correct speaker
   assignment; speaker pages and party stats depend on it.
4. **WER and CER (overall)** — universal baselines for comparability. CER is
   fairer for heavily-inflected Greek.
5. **Corrections per hour of audio** — the business metric. On fresh meetings,
   how many corrections per hour do Eliza/Thanos make? Compute as a shadow eval:
   run the model on recent meetings, diff against human-corrected ground truth.

Bolded as release-defensible: **NE-WER** and **corrections-per-hour**. The rest
is context.

## What we pick when we configure a run

**Base model:** whisper-large-v3 (obvious start) · large-v3-turbo (faster, worth
comparing) · medium (if GPU-constrained).

**Approach: start with LoRA / PEFT.** Trains ~1% of params, fits a single 24 GB
GPU, much less catastrophic forgetting, swappable adapters. Graduate to full
finetune once we know what works. LoRA start: rank 16–32, alpha 32–64, target
`q_proj,k_proj,v_proj,out_proj` (+ optionally MLP).

**Hyperparameters (sensible defaults):** LR 1e-5 full / 1e-4 LoRA · warmup ~10%
· effective batch 8–32 (grad-accum if VRAM-limited) · epochs 2–5 (watch val
loss) · weight decay 0–0.01 · grad clip 1.0 · **BF16** · dropout 0.0 · freezing
the encoder and training only the decoder is a fine middle ground for narrow
domains.

**Data-side:** keep SpecAugment on · use HF `WhisperFeatureExtractor`
consistently for train and inference · decide text normalisation **once** (for
us: keep punctuation, keep capitalisation, keep numerals as written) and apply
it identically to training labels and reference transcripts.

**Decoding:** beam 5 (greedy often nearly as good) · temperature 0.0 with
Whisper's fallback ladder · `condition_on_previous_text=False` · force
`language="el"`.

## Cost of a training run

GPU rental, mid-2026: A100 80 GB ~$0.89–1.40/hr community, ~$1.89 secure ·
H100 80 GB ~$1.38–2.40/hr · RTX 4090 24 GB ~$0.34/hr. Plan with **~$1.50/hr
A100**.

Throughput (large-v3, BF16): **LoRA ~5 h audio/GPU-hr**, **full finetune ~1.5 h
audio/GPU-hr**. Total GPU-hours ≈ `dataset_hours × epochs / throughput`.

**LoRA on one A100, 3 epochs:** 10 h → ~$10 · 50 h → ~$45 · 200 h → ~$180 ·
500 h → ~$450. **Full finetune, 3 epochs:** 10 h → ~$30 · 50 h → ~$150 ·
200 h → ~$600 · 500 h → ~$1,500.

**Hidden costs:** budget 2–3× for failed runs · sweeps multiply runs (sweep on a
10 h subset, then one full run with the winner) · storage negligible · use a
4090 for pipeline dev/debug, don't burn A100 hours on a tokenizer bug.

**Realistic first-cycle budget: $1,000–3,000.** Cheap vs engineering time. The
expensive resource is human hours on the dataset, eval harness, and integration
— **optimise for engineering throughput, not GPU cost.** Schema Labs should
apply for cloud credits (Google/AWS/Azure nonprofit programs, $5K–$25K).

## Practical workflow — zero to a real model

1. **Eval harness first.** Before training anything, build the pipeline that
   loads held-out meetings, runs the baseline, computes the metrics table. Make
   the numbers reproducible. Most underrated step.
2. **Smoke test.** Finetune 100 examples for 50 steps; confirm loss drops and
   output is still sensible Greek. Most bugs surface here (language token,
   feature extractor, label/audio mismatch).
3. **Scale gradually:** 1 h → 10 h → 50 h, checking val WER each step. No
   improvement at 10 h → something's wrong with data or config; don't brute-force
   500 h.
4. **Evaluate on the metrics that matter** — WER as sanity check, NE-WER and
   per-category WER as real signals, shadow-eval correction rate as the business
   signal.
5. **Iterate on the dataset, not just hyperparameters.** Most gains come from
   data quality. Ask "what's wrong with the data?" before "what's wrong with the
   LR?"
6. **Ship behind a flag.** Run in shadow mode alongside the current provider,
   compare on real incoming meetings, flip when eval numbers and the human-review
   experience both confirm it's better.

## Things to watch out for

- **Language/prefix tokens**: handle the prefix at training time; force
  `language="el"` at inference. HF `WhisperProcessor` does this if used
  consistently.
- **Catastrophic forgetting**: hold out non-council Greek (e.g. CommonVoice) and
  check it doesn't regress badly.
- **Hallucinations on silence/noise**: test on pauses, room noise,
  side-conversations.
- **Train/inference mismatch**: same feature extractor, sample rate, language
  token, text normalisation on both sides. Verify by transcribing a training
  example and matching the label.
- **Don't trust val loss alone**: val loss down + WER up usually means a text
  normalisation problem.

## What success looks like

- A finetuned Whisper checkpoint (LoRA adapter + base, or full weights).
- Lower NE-WER and per-category WER than baseline on a held-out test set
  including a held-out municipality.
- Measurable drop in correction-minutes-per-hour in the shadow eval.
- A shareable evaluation report (marketing + contribution to Greek ML).
- Model and recipe open-sourced.

## Where the mini PC fits (local compute)

Angelos has a mini PC: Ryzen 7 **7840HS** (Radeon **780M** iGPU, RDNA3) + 64 GB
RAM. It can allocate a large slice of RAM as iGPU VRAM, so **capacity is not the
constraint** — bandwidth, compute, and software support are.

- **iGPU training is not the path.** The 780M (gfx1103) is **not officially
  supported by ROCm**; it can sometimes be forced
  (`HSA_OVERRIDE_GFX_VERSION=11.0.0`) but it's fragile for training. Shared DDR5
  bandwidth (~80–100 GB/s) is ~20× below an A100's VRAM, and raw FP16 throughput
  is ~40× lower. A run that costs ~$45 / ~30 h on an A100 would take **weeks** on
  the iGPU, if ROCm cooperates at all.
- **What the mini PC IS good for:**
  - the **eval harness** and **baseline measurement** — running inference
    (Gladia comparison aside, zero-shot Whisper) to compute WER. With
    faster-whisper / CTranslate2 INT8 on the 8-core Zen4 CPU this is genuinely
    usable and free.
  - **pipeline smoke tests** and **LoRA dev/debug** on tiny subsets.
  - **data prep** (segmentation, normalisation, building train/val/test).
- **Do the real training on rented GPUs** (Runpod/Vast 4090 for dev, A100 for
  runs). First runs are ~$10–50 — trivial vs the engineering time the mini PC
  saves by handling all the local dev/eval work for free.

→ canonical: [GPU strategy](../decisions/data.md#2026-06---gpu-strategy-rent-for-training-mini-pc-for-evaldev).
