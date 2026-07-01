# First real GPU run — whisper-large-v3 + LoRA on the curated set

_Kaggle, single Tesla T4. Base model **whisper-large-v3** (1.55B params),
**LoRA** on attention projections — 7,864,320 trainable params (**0.51%**).
This is the first end-to-end fine-tune on the actual curated dataset (not the
tiny-Whisper CPU sweep in [`report.md`](report.md)), and the first one that
shows a **clear, real improvement** on held-out cities._

Source notebook: [`notebooks/whisper_finetune_kaggle.ipynb`](../../../notebooks/whisper_finetune_kaggle.ipynb).

## Setup

- **Data** built live from `/api/export` (the 2,179 human-**included**
  corrections) + a **no-edit backbone** of reviewed-but-unchanged utterances,
  excluding the 13 denylisted unreviewed meetings (`dropped 0 rows` — they carry
  no included rows anyway).
- **Split (city-disjoint):** the two held-out cities **orestiada + argos** are
  the validation; everything else is train.
  - `train = 1964` clips (232 meetings)
  - `val_corr = 191` clips — corrections in the held-out cities (the hard cases)
  - `val_reg  = 280` clips — reviewed **no-edit** speech in the same held-out
    cities (the regression guard: did we make ordinary speech worse?)
  - `audio_fail = 0` (every clip decoded).
- Greek forced at decode; LoRA on q/v projections; ~0.51% of weights trained.

## Result — improvement on both sets

Lower is better. Baseline = whisper-large-v3 **zero-shot** (no fine-tune);
After = the same model with the LoRA adapter.

| set | metric | baseline | after | Δ (abs) | Δ (rel) |
|---|---|---|---|---|---|
| **val_corr** (corrections) | WER | 33.41 | **26.73** | −6.68 | **−20.0%** |
| | WER (norm) | 25.14 | 18.91 | −6.23 | −24.8% |
| | CER | 15.74 | **10.40** | −5.34 | −33.9% |
| | eval_loss | 1.268 | 0.251 | | |
| **val_reg** (ordinary speech) | WER | 27.09 | **17.32** | −9.78 | **−36.1%** |
| | WER (norm) | 16.37 | 8.22 | −8.15 | −49.8% |
| | CER | 15.06 | 6.46 | −8.60 | −57.1% |
| | eval_loss | 1.679 | 0.146 | | |

Two things matter here, and they fix the open question from the CPU sweep:

1. **The corrected cases moved this time** (−20% relative WER, −34% CER), where on
   the 56-clip CPU val they were pure noise. The bigger model + a real 191-clip
   val is enough to see the signal.
2. **Ordinary speech improved more than the hard cases** (−36% WER, −50%
   normalized). The "train only on corrections → degrade easy speech" trap did
   **not** happen — same finding as the CPU run, now on large-v3. The model is
   learning the domain (council vocabulary, names, number/date formatting), which
   helps the easy majority most.

> Caveat on magnitude: the very large `val_reg` gains (esp. CER −57%) partly
> reflect **format/domain adaptation** (numerals, casing, punctuation house
> style) rather than purely acoustic gains. **Normalized WER is the fairer
> headline**, and it still drops hard. Absolute numbers need the fixes below
> before they're quotable.

## What went wrong at the end (fix before re-testing)

The run finished and saved (`/kaggle/working/whisper-lora-greek`), but the logs
carry three real problems:

1. **`clean_up_tokenization_spaces=True` on the WhisperTokenizer.** Transformers
   itself warns this is *destructive for BPE* — it strips spaces before
   punctuation. That corrupts the decoded text and **distorts WER/CER**,
   especially for the `punctuation_capitalization` and `accent_tonos` categories.
   → **Fix:** pass `clean_up_tokenization_spaces=False` in the eval decode (and
   when saving the processor).
2. **`pad_token == eos_token`, no `attention_mask` passed to `generate()`.**
   Transformers warns "you may observe unexpected behavior." With Whisper this can
   silently hurt generation near padding. → **Fix:** set a distinct pad token (or
   pass the `attention_mask` explicitly) for both eval and inference.
3. **The kernel restarted several times and rebuilt the dataset from scratch each
   time (~70 min/build, no cache).** Most of the wall-clock was wasted re-decoding
   audio. → **Fix:** build the clips once, persist them as a Kaggle dataset (or
   checkpoint + resume), so a restart skips data prep. Also set `HF_TOKEN` to
   avoid the unauthenticated-rate-limit warning.

None of these invalidate the *direction* (both sets improved by a wide margin),
but #1 means the **absolute** WER/CER are slightly off and shouldn't be quoted as
final until the re-run.

**Status (2026-06-24): #1 and #2 are fixed in the notebook**
(`notebooks/whisper_finetune_kaggle.ipynb`, cells 7/5/10; backup `.ipynb.bak`).
#3 (attention_mask warning) is left as-is on purpose — confirmed **benign for
Whisper**: the feature extractor pads every clip to the fixed 30 s input, so the
encoder sees uniform-length features and the missing mask changes nothing
(Codex high-effort review, 2026-06-24). A regression test pins the decode policy:
`eval/tests/test_decode_cleanup.py` (pure-Python parts pass locally; the
real-tokenizer check runs on Kaggle / mini PC).

## How to re-test

1. **Re-run the exact same frozen split** with fixes #1–#3 applied. Confirm the
   deltas hold once tokenization isn't corrupting the text.
2. **≥3 seeds + bootstrap CIs**, clustered by meeting (utterances in one meeting
   aren't independent). Report a confidence interval on each delta, not a point.
3. **Grow the validation set** toward the full held-out pool (~9,900 utterances in
   orestiada+argos) instead of 191/280, so the corrections delta is statistically
   solid.
4. **Report normalized WER as the headline** + per-category WER (does it fix what
   we built it to fix: names, acronyms, numbers?), keeping raw WER/CER as context.
5. Keep **large-v3 + LoRA + composition recipe** (corrections + no-edit backbone);
   re-tune LR/steps on GPU (they don't transfer from the CPU sweep).

## Getting it to the mentors (inference + hosting)

All three of these are feasible, safe, and free.

**Download + inference on the mini PC (Ryzen 7840HS, CPU).** Yes. The Kaggle
output `whisper-lora-greek` is the **LoRA adapter** (~tens of MB) + processor —
the 3 GB base (`openai/whisper-large-v3`) downloads from HF.
- Quick path: `transformers` + `peft` load base + adapter on CPU. Works; slow
  (RTF > 1) but fine for a bounded benchmark.
- Eval-grade path (the documented one): `merge_and_unload()` the adapter into the
  base, convert to **CTranslate2 int8**, run **faster-whisper** on the 8-core CPU.
  int8 large-v3 ≈ 1.5 GB; 64 GB RAM is plenty. This is what
  [GPU strategy](../../docs/decisions/data.md#2026-06---gpu-strategy-rent-for-training-mini-pc-for-evaldev)
  reserves the mini PC for. **Set the processor's
  `clean_up_tokenization_spaces=False` on load** so displayed transcripts aren't
  corrupted (same fix as eval).

**Host it safely + for free → private HuggingFace model repo.** Push the adapter
(and optionally the merged model) to a **private** HF repo, add the mentors as
collaborators. Free, access-controlled, standard. They pull it straight into the
`bench.opencouncil.gr` harness and run inference themselves — no always-on
endpoint needed (those aren't free). **Keep it private** until the publish-plan
license/PII review ([dataset-split-and-publish-plan](../../docs/specs/dataset-split-and-publish-plan.md#3-the-post-huggingface-dataset-publication));
public release of model **or** data is a separate decision. Set `HF_TOKEN` in the
notebook to push and to silence the unauthenticated-rate-limit warning.

**Running it in the benchmark.** Hand the mentors: the private HF repo + a short
"load adapter / merge / faster-whisper" snippet + the eval recipe (force Greek,
`condition_on_previous_text=False`, `clean_up_tokenization_spaces=False`). Then
the bench compares baseline → our model on the **frozen temporal/held-out** set
for release-defensible numbers — not the random provider scoreboard (its windows
leak across train/test).

## Provenance

- Notebook: `notebooks/whisper_finetune_kaggle.ipynb`; adapter saved to
  `whisper-lora-greek`.
- Curated source: 2,179 included corrections (live VPS review tool), per-city:
  chania 632, athens 576, sparta 161, samothraki 150, chalandri 135,
  vrilissia 115, xylokastro 110, zografou 107, orestiada 102, argos 91.
- Companion CPU sweep (data-recipe findings that motivated the composition):
  [`report.md`](report.md).
