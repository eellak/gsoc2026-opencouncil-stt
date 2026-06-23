# Fine-tuning dry-run plan (Whisper-large-v3, Greek council ASR)

Status: **plan for execution on the mini PC by an AI agent** (2026-06-23).
Reviewed with Codex (see end). The human runs this on the box; this file is the
brief. Verified facts: Grok (HF recipe currency), live probe of the mini PC.

## 0. Goal & scope (what this run IS and ISN'T)

**Goal:** run the *whole* fine-tuning pipeline end-to-end on the **~2000 real
reviewed corrections we already have**, to (a) prove the data format, (b) see
exactly what fine-tuning needs, (c) experiment with Whisper N-best/confidence
output, (d) de-risk before any real GPU training run.

**NOT:** the final published model or the full balanced dataset. This is a
realistic dry-run / pipeline validation (the "run it end-to-end" Notion idea).

## A. CONFIRMED architecture (2026-06-23) — supersedes older detail below

**Data source — already built, nothing to add.** The VPS review UI already serves
the curated training set:
- **`GET https://79-76-114-184.sslip.io/api/export`** (oracle-vm, Caddy + auto-
  HTTPS, public, no auth) → newline-delimited JSON of the **UI-`include`** groups.
- Live count: **1,905 included utterances ≈ 1.8 h audio**, all with
  `final_after_text`. Per city: chania 537, athens 511, sparta 140, samothraki
  126, chalandri 114, vrilissia 98, zografou 92, xylokastro 86, orestiada 82,
  argos 72, + a small tail.
- Each line has everything we need: `audio_url`, `start`, `end`,
  `final_after_text` (the corrected target), `city_id`, `meeting_id`,
  `error_categories`, `include_status`. So **Colab fetches one URL** — no separate
  export step to build.
- ⚠️ Optional hardening: the endpoint is public + unauthenticated. If we care, add
  a `?token=` check in `+server.ts`. Council data is semi-public, so low urgency.

**This is the curated set the user means by "~1,800 approved" — we train on these
first**, then add the no-edit backbone (~109 h from reviewed meetings) + other
tiers later.

**End-to-end flow (Colab):**
1. `requests.get(EXPORT_URL)` → parse JSONL → ~1,905 rows.
2. Filter out the **val cities** (`orestiada`, `argos`) → train; keep them → val.
3. Group by `meeting_id`; download each meeting `audio_url` (data.opencouncil.gr)
   **once**, cache; slice each utterance `[start,end]`, resample **16 kHz mono**.
4. Build HF `datasets` → LoRA fine-tune → WER on val → save adapter to Drive.

**Two compute tracks (run in parallel):**
- **Track 1 — Colab free (T4 16 GB):** the *real-ish* run — large-v3 LoRA on the
  1.8 h set. Free GPU, ~1–2 h, save adapter to Drive. The reproducible/trustworthy
  artifact.
- **Track 2 — mini PC (CPU) auto-research:** a **tiny** Whisper (`base`/`small`)
  in an automated experiment loop (mirrors the prompt `improve_loop.py`) that
  varies dataset/training choices — e.g. **error-type focus / error-type
  distribution / weighting** — and reports what moves val WER. CPU-cheap because
  the model is tiny; the goal is *what helps*, not a deployable model.
  *(Confirm interpretation: "Auto Research" = an automated fine-tuning
  experiment loop on a tiny model — like the prompt loop but for FT. Yes?)*

**Validation (you have no test set — here's the trustworthy answer):**
- **Quantitative val (automatic, reproducible):** WER on the held-out **orestiada
  + argos**, *before vs after* fine-tuning. That is the trustworthy number. It is
  NOT the "test" (= future data) but it's a valid disjoint measurement.
- **Qualitative / future test:** pull **new public meetings post-2026-06-01 from
  the OpenCouncil API** (same endpoint, no special access for public ones) as they
  appear, run the model, eyeball / WER. As they get reviewed they become the real
  temporal test. **No OpenCouncil grant needed** for public meetings.
- **Save the LoRA adapter to Drive** so nothing is lost; reload to run inference
  anytime.

---

## B. Keeping the run alive (Colab free) — compute ops

Key fact: **Colab compute runs on Google's servers**, not your machine. Your
machine only needs to keep the **browser tab connected** so Colab doesn't mark it
idle. So:
- **Your AnyDesk-into-minipc plan works.** The always-on mini PC has stable
  internet → host the Colab tab there, leave it open. The laptop's flaky internet
  is irrelevant. ✓
- **During active training there's no idle problem** — the 90-min idle timer only
  fires when *no cell is running*. One long training cell (our run is ~1–2 h ≪ the
  12 h free cap) keeps the session alive by itself.
- **Real safety net = checkpoint to Drive**, don't rely on keep-alive: mount Google
  Drive, `save_steps` to Drive, cache the built dataset to Drive. If the session
  dies, resume from the last checkpoint instead of restarting.
- **Even better free option — Kaggle Notebooks:** "Save & Run All (Commit)" runs
  the notebook **headless on Kaggle's servers** (no open tab at all, up to 12 h,
  T4×2 / P100, ~30 h/week). For an unattended ~1–2 h job this is cleaner than
  Colab — **no keep-alive trick needed**. Strong recommendation if the tab-alive
  dance is annoying.
- **Colab Pro ($10/mo):** background execution (close the tab, keeps running) +
  better GPUs. The clean paid fix if we do this often.

---

## 1. ⚠️ Hardware reality — the one real blocker

Live probe of the mini PC: **16 cores, 60 GB RAM, NO GPU** (no NVIDIA/CUDA, no
usable ROCm), ffmpeg present, 187 GB free, no `torch` yet.

- **Fine-tuning large-v3 on this box is NOT feasible** (CPU-only → days/weeks per
  epoch, swapping). Confirmed by Grok.
- **So split the compute:**
  - **mini PC does:** dataset build + format validation + a *small-model* pipeline
    dry-run (whisper-small, a few hundred steps) + CPU inference experiments
    (faster-whisper). Enough to prove every step works.
  - **Real large-v3 LoRA training:** rent a cloud GPU — **A100-40GB or RTX 4090 /
    A6000** is plenty for large-v3 LoRA on 2k samples (RunPod / Vast.ai / Lambda).
- **DECISION NEEDED:** (a) mini-PC dry-run with whisper-small first, *then* rent a
  GPU for the real large-v3 run? or (b) rent the GPU now and do it all there?
  Default recommendation: **(a)** — cheap, proves the pipeline before paying.

## 2. Target data format (what Whisper fine-tuning eats)

HuggingFace `datasets` row:
```
{
  "audio": {"array": <float32 mono @ 16 kHz>, "sampling_rate": 16000},
  "text":  "<corrected Greek transcript for that utterance>"
}
```
That is **all** the trainer needs: **audio clip + target text**. (Answer to your
question: yes — only audio + text. Timestamps are NOT a training target here; we
use them only to *cut* the clips. No "wrong text" needed — confirmed we publish
audio + correct text only.)

## 3. Building the dataset from what we have

Per included correction we have everything needed: `utterance_start`,
`utterance_end`, `audio_url`, and the corrected `after_text`.

**Build pipeline (`eval/build_asr_dataset.py`, to write):**
1. Take the **included** reviewed corrections (the UI include set) → ~2000 rows.
2. For each: download+cache the meeting audio (`audio_url`), slice `[start,end]`
   with ffmpeg, resample to **16 kHz mono**, pad ±0.2 s.
3. Target text = `after_text`, lightly normalized (consistent with how we score).
4. **Timestamp hygiene:** drop zero/negative or implausibly long spans; sanity-
   check clip-seconds vs text length; (Notion: ~10% of errors are timestamp
   issues → discard those). Whisper FT tolerates small boundary slop, not gross
   misalignment.
5. Save as a HF `datasets` dir (or jsonl + audio files) + a manifest.

**Open question:** does the app's **export** already emit
`audio_url + start + end + corrected_text`? If yes we read it directly; if not we
build from the corrections CSV + the cached meeting audio. (We already have all
the fields — this is plumbing, not missing data.)

## 4. Split for this experiment (revised per your call)

- **TEST = future data** (new meetings, post-cutoff, *not yet available*). Per your
  decision, the test is new data by design → **we do NOT carve out test cities.**
- **VALIDATION = 2 whole cities held out: `orestiada` + `argos`** (~27 h). Whole
  cities = **automatically speaker- and meeting-disjoint** (0 cross-city speakers),
  zero leakage, dead simple. We **drop** the Notion "+30% of speakers from the
  other 8 cities" — see §4a.
- **TRAIN = the other 8 main public cities** — corrections (this run) + later the
  no-edit backbone.
- For the tiny dry-run specifically, val can be just a small held-out slice; the
  2-city val is for the real run.

### 4a. Evaluation of the Notion split (you asked)
Notion: *2 whole cities → val, + 30% of speakers from the remaining 8 → val,
70% → train, ensure enough data per speaker.*
- **Feasible?** Yes, mechanically.
- **Best practice / needed?** The speaker+meeting-disjoint principle is right and
  we keep it. But (1) **test = future data**, so we don't need cities for *test* —
  the 2 cities serve **validation**. (2) The extra **"30% of speakers from the 8
  training cities"** is **not needed** here: holding out 2 whole cities already
  gives a clean disjoint val; adding a speaker-split on top mostly **wastes
  training data** and adds the "by count vs by minutes" complexity. Keep it simple:
  **2 cities = val, the rest = train.** (If later we want a *deployment-realistic*
  val — unseen speaker in a seen city — the speaker-split is the refinement, by
  speaker-**minutes** not count. Not for this run.)

## 5. Fine-tuning recipe (the real GPU run)

Libraries: `transformers`, `datasets`, `peft`, `accelerate`, `evaluate` (WER),
`faster-whisper` (inference). Model: `openai/whisper-large-v3`.

- `WhisperProcessor` + `WhisperForConditionalGeneration`.
- **LoRA** (PEFT): `r=32, alpha=64, target_modules=["q_proj","v_proj"]`,
  decoder-focused; **freeze the encoder** (domain-adapt the decoder without
  wrecking multilingual acoustics).
- `DataCollatorSpeechSeq2SeqWithPadding`, `Seq2SeqTrainer`.
- HParams (small-data regime): LR **1e-4 → 5e-5**, **3–5 epochs max**, early
  stopping on **val WER**, gradient checkpointing, bf16 (or 8-bit on the GPU).
- Optional augmentation: SpecAugment / mild noise / speed-perturb.
- **Gotchas (~2k samples):** overfitting is easy → early stop, few epochs; watch
  for large-v3 hallucinations/repetitions; pick the best checkpoint before it
  diverges. Quality of transcripts > quantity.

## 6. N-best / confidence output experiment (your "best answers, not just one")

Use **faster-whisper** for inference: `transcribe(..., beam_size=5,
word_timestamps=True)` → each segment has `avg_logprob`, `no_speech_prob`, and
per-word `probability`; beam search also yields **N-best** alternatives.

**Design — a condensed, thresholded signal for the downstream LLM:**
- Emit normal text for confident spans.
- For spans below a **threshold** (e.g. `avg_logprob < -1.0` **or** min word
  `probability < 0.7`), emit a compact JSON object: `{text, alternatives:[…N-best],
  conf}` so the fix-task LLM **focuses correction only on uncertain spans**.
- The threshold keeps it from firing on everything (your requirement).
- **Experiment:** A/B the fix-task with plain text vs this confidence-annotated
  input — does the richer signal lower HIR? (This connects ASR → fix-task.)

## 7. Phased execution (what the mini-PC agent runs, in order)

- **Phase 0 — env:** venv; install CPU torch + `transformers datasets peft
  accelerate evaluate faster-whisper`; verify ffmpeg.
- **Phase 1 — dataset:** run `build_asr_dataset.py` → ~2000 clips (16 kHz) + text +
  small val; load as HF dataset; eyeball 5 clips (audio length vs text) to confirm
  timestamp sync.
- **Phase 2 — baseline WER:** faster-whisper large-v3 inference on the val set,
  measure WER *before* fine-tuning (the number every later run is compared to).
- **Phase 3 — pipeline dry-run (CPU):** fine-tune **whisper-small** for a few
  hundred steps on train; confirm loss ↓, checkpointing, eval loop all work. Proves
  the loop end-to-end without a GPU.
- **Phase 4 — N-best/confidence (CPU):** run the §6 experiment; produce the
  condensed JSON; sanity-check the threshold.
- **Phase 5 — REAL run (GPU, separate box):** large-v3 LoRA per §5; WER on val;
  compare to Phase-2 baseline; keep best checkpoint.

## 8. Open questions / gaps (RESOLVED + remaining)

Resolved this session:
- ✅ **Data source:** UI `include` set via `/api/export` (public URL) — **ready**.
- ✅ **The "~2000":** it's the **1,905 included** utterances (1.8 h). Train on these
  first, + no-edit backbone later.
- ✅ **Export format:** already gives `audio_url+start+end+final_after_text`.
- ✅ **Compute:** Track 1 Colab free (T4) for the real-ish run; Track 2 mini-PC tiny
  model for auto-research. (No GPU rental needed for the dry-run.)
- ✅ **Split:** test = future data (no test cities); val = orestiada+argos held out.

Remaining to confirm:
1. **Auto-research scope (Track 2):** confirm "Auto Research" = automated FT
   experiment loop on a *tiny* Whisper that varies error-type focus/distribution
   and reports val-WER deltas. Anything specific you want it to sweep?
2. **Val size:** val = only the *included* orestiada+argos (82+72=154 utts), or the
   larger human-corrected pool in those cities (~9.9k) for a more stable WER?
3. **N-best schema:** confirm the condensed JSON shape for the LLM (§6).
4. **Audio normalization** (open since June): normalize volume per-meeting before
   slicing, or leave raw? (Affects training + inference.)
5. **Export auth:** add a `?token=` to `/api/export`, or leave public?

---

## Codex review (high effort, 2026-06-23) — fold these in

**Framing:** sound as **pipeline-validation + preliminary signal**, NOT evidence of
model improvement. 1.8 h will likely overfit; set expectations accordingly.

1. **Distribution bias is the key trap.** Corrections-only data is biased toward
   *errorful* utterances → fine-tuning may help similar errors while **ordinary
   speech regresses**. **Must include a "reviewed-unchanged" (no-edit) regression
   set in validation now** — otherwise correction-only val can't reveal general
   ASR degradation. (So pull *some* no-edit utterances for val even in the dry-run.)
2. **LoRA:** try **r=8/16 before r=32**; multiple **seeds (≥3)**, report mean/range
   not one best; select checkpoints by val metric, not fixed epochs.
3. **Val variance:** orestiada+argos *included* = only 154 utts → too few/high-
   variance. Use the **larger human-corrected pool** in those cities AND add
   meeting-level **k-fold (3–5)** across all cities; report per-city/per-meeting +
   bootstrap CIs. Verify each held-out city has enough duration + category coverage.
4. **Build correctness:** **decode each meeting mp3 to PCM once, then slice PCM**
   (no repeated compressed-file seeks); pad context but ensure the target text does
   NOT include neighbour speech; filter overlaps/crosstalk/silence/duration-text
   mismatch; cache source audio + record hashes; dedup utterances across splits;
   **manually audit a stratified sample of clips before training**.
5. **Metrics:** report **raw WER + normalized WER + Greek CER + S/D/I rates +
   per-category**; fix decoding params identically across baseline and FT; version
   the normalization rules.
6. **Tiny-model sweeps that transfer** (Track 2): **data composition**
   (corrections-only vs + no-edit backbone), **sampling/weighting** (uniform vs
   category-balanced vs capped oversampling), **LR / effective batch**, **segment-
   quality filters**. (Model-rank sweeps transfer poorly — prioritise data-mixture.)
7. **N-best caveat:** confirm the faster-whisper API actually exposes **beam
   alternatives** — word/token probabilities are NOT N-best hypotheses. Calibrate
   the confidence threshold against observed error rates.
8. **Reproducibility checklist:** frozen + versioned export **manifest** (row IDs,
   source hashes, split assignment, dataset stats); baseline eval with identical
   preprocessing/decoding; loss curves + checkpoint-selection rule; record code
   commit, package/model revisions, LoRA config, seed, dataset hash; log audio
   failures instead of silently dropping rows.
