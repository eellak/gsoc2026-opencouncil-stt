# Handoff: fine-tune evaluation postmortem and next directions (2026-07-25)

Audience: the next agent (or person) picking up the Greek council ASR work. This is a
self-contained brief. Read it before touching the model or the benchmark.

> **⚠ PARTIALLY CORRECTED (2026-07-29).** The claim below that the *training-time* "-32%"
> was measured against an unfairly-served baseline is **wrong**. The code shows the training
> baseline used the same trainer, clips, decode path and normalization
> (`notebooks/train_runpod.py:286-292`) — it was fair within its own harness. The
> unfair-baseline story holds only for the **benchmark** comparison. The real candidate
> causes (checkpoint selected on the val set; `val_reg` scored against another ASR's output;
> greedy-HF vs beam faster-whisper decoding) are analysed in
> [2026-07-29-where-we-are-stuck.md](2026-07-29-where-we-are-stuck.md). Read that first.

## TL;DR

The fine-tuned `whisper-large-v3` council LoRA **does not beat base whisper** under a
controlled, same-stack comparison. On the utterances it was specifically trained to
fix, it is slightly **worse** than base. The earlier positive results (benchmark
"wins", "+8.6pp on names", the training report's "-32%") were **evaluation
artifacts**: the model was always compared against a baseline running under worse
conditions than itself. `int8` quantization was suspected and **ruled out** (int8 ==
float32). Do not migrate this adapter to production. The path forward is (1) a
trustworthy controlled eval, then (2) inference-time contextual biasing with the
per-meeting roster and glossary, which targets names without retraining.

## The one real mistake, stated plainly

Every comparison that made the fine-tune look good measured it against a baseline that
was served differently and worse:

- The OpenCouncil benchmark's base-whisper provider (`hf-openai-whisper-large-v3`) runs
  through the HuggingFace serverless pipeline, which decodes poorly here (hallucinations
  like "Υπότιτλοι AUTHORWAVE", garbled names). Our model ran on a well-tuned
  faster-whisper stack. So "we beat base" mostly meant "our serving stack beats HF's."
- The training report's "-32% on val_corr" was measured against a baseline that was not
  the same fairly-served model / same normalization, so it overstated the effect.

When base whisper and the fine-tune run on the **identical** faster-whisper stack (same
decode params, same text normalization), the gain disappears. This is the classic
applied-ML trap (train/eval serving skew, unfair baseline). It is not a beginner error;
it is easy to miss and common. Catching it is the win here.

## The decisive evidence (controlled same-stack A/B, run 2026-07-24)

Base whisper and the fine-tune, both through the same faster-whisper/CTranslate2 stack
(beam=2, `condition_on_previous_text=False`), on held-out argos/orestiada audio, scored
with the same normalization. Scripts and raw results are in
`eval/controlled_eval/` (`ab_general_utterances.py`, `ab_corrected_utterances.py`,
`results_*.json`).

General entity-containing utterances (n=34):

| model (same stack) | WER | entity recall |
|---|---|---|
| base whisper float32 | 0.163 | 86.4% |
| ours float32 | 0.164 | 87.2% |
| ours int8 (shipped) | 0.164 | 87.2% |

Tie. And int8 == float32, so quantization is not eroding anything.

Actually-corrected held-out utterances (n=50, the fine-tune's target), scored vs the
human reference:

| model (same stack) | WER vs human ref |
|---|---|
| base whisper float32 | 0.158 |
| **ours float32** | **0.176 (worse)** |
| Scribe original "before" text | 0.155 |

Per-utterance head-to-head, ours vs base: better on 6, worse on 20, tied on 24. On the
cases it was built to improve, the fine-tune regresses. Observed failure modes: it drops
leading words (onset drift) and mangles name spelling (reference `Λιόλιος` becomes
`Λιόλειος`, `τεκμηριώσει` becomes `τεκμεριώσει`).

For contrast, the earlier (artifact) numbers from the 260-clip benchmark held-out slice:
ours 0.173 vs HF-served base 0.178, and on hard multi-word names ours 69.1% vs 60.5%.
Those gaps are the HF-pipeline handicap, not the adapter.

## What was ruled out

- **int8 quantization** — int8 matches float32 exactly on this data.
- **Broken export / merge** — the adapter is clearly active (outputs differ from base).
- **Catastrophic forgetting** — general Greek is fine; the model is not broken, just not
  better.

## Root causes (what to actually fix)

1. **Evaluation methodology (the big one).** No comparison was a controlled same-stack,
   same-normalization A/B until 2026-07-24. Fix this first, permanently.
2. **The fine-tune recipe is weak or slightly harmful.** LoRA r32 on q_proj/v_proj only,
   2 epochs, base encoder weights frozen (LoRA still trainable inside the encoder),
   trained on **isolated cut utterances**.
   **Update 2026-07-31:** a third cause was found that this list missed entirely — the
   [label-prefix bug](../reports/2026-07-31-label-prefix-bug.md). Every run trained on
   targets shifted one position. Weigh the recipe critique below against that. The isolated-utterance
   training is the likely source of the onset-drop (the model learned to start each
   segment "clean" and drops bridge words inside a continuous window). Name-spelling drift
   suggests the correction signal taught changes that do not generalize to unseen cities.

## Assets and where everything is

Model:
- Adapter (public): `opencouncil/whisper-large-v3-el-council-lora` on HuggingFace.
- Base: `openai/whisper-large-v3`. LoRA r32 / alpha64, targets q_proj+v_proj, encoder
  frozen, 2 epochs, lr 1e-4. Held-out cities `VAL_CITIES = {argos, orestiada}`
  (`notebooks/train_runpod.py`).
- Locally merged model: `/home/harold/oc-asr-serve/merged`; shipped ct2 int8:
  `/home/harold/oc-asr-serve/ct2` (mini-PC only, not in repo).

Serving:
- Mini-PC CPU endpoint: `https://asr.haroldpoi.dev` (faster-whisper, Cloudflare tunnel).
  Runbook: `docs/runbooks/self-hosted-asr-endpoint.md`.
- RunPod Serverless (scale-to-zero, pay-per-use): endpoint `o1jda6sxo85dnk`, image
  `ghcr.io/angelospk/oc-asr-serverless`, source `github.com/angelospk/oc-asr-serverless`.
  Runs float16 (int8_float16 throws CUBLAS_STATUS_NOT_SUPPORTED there). Note: it serves a
  model that is not better than base, so do not push it to OpenCouncil for production yet.

Evaluation data (all local, held-out = argos/orestiada):
- `data/asr/val_manifest.csv` — 9,875 held-out utterances (utterance_id, city, meeting,
  start, end, audio_url, text).
- `data/asr/export.jsonl` — corrections: `initial_before_text`, `final_after_text`,
  `edits`, `error_categories`, `include_status` per utterance. This is how you isolate the
  actually-corrected utterances.
- `data/asr/audio/*.mp3` — 368 full meeting recordings (naming `{city}__{meeting}.mp3`).
- `data/glossary/glossary.json` — entity vocabulary (`global` ~5,900 + `per_city`).
- Rosters: `eval/fetch_rosters.py` (per-meeting speaker lists, for contextual biasing).

Benchmark (`bench.opencouncil.gr`, needs `Authorization: Bearer tbk_...`):
- `2026-06-10-sample-benchmark` — 65 clips, ~78% are training meetings (contaminated).
- `2026-06-10-oc-benchmark` — 260 clips, 10 cities, richer provider list.
- `2026-07-23-june-benchmark-runpod-gpu-finetune-corre` — our clean 260/260 GPU re-run.
- Reminder: only argos + orestiada are held out; all other cities were in training.

Reports and memory:
- `docs/reports/2026-07-23-benchmark-results.md` — full benchmark write-up, ends with the
  controlled A/B section and the "do not migrate" conclusion.
- Controlled-eval scripts + raw results: `eval/controlled_eval/`.

## How to reproduce the controlled eval

Needs the `.venv-eval` environment (faster_whisper, ctranslate2, ffmpeg) and the ct2
models. `ab_general_utterances.py` builds base and ours as float32 ct2 (base from the HF
id, ours from the local merged model; for a clean machine, build ours from the public
adapter per the `oc-asr-serverless` Dockerfile recipe). Then run
`ab_corrected_utterances.py` for the decisive corrected-utterance test. Both write WER +
entity recall and a per-utterance head-to-head. Read the PORTABILITY NOTE at the top of
each script for the paths to change.

## Ideas to explore next (ranked, each with how to measure)

1. **Build a trustworthy controlled eval harness (foundation, do this first).**
   Generalize `eval/controlled_eval/` into one tool: same stack, same normalization, two
   held-out subsets (general + actually-corrected), reporting WER, CER, and entity recall
   (using the glossary), with a per-utterance head-to-head. Every future claim goes
   through this. Without it, you will chase artifacts again.

2. **Inference-time contextual biasing with roster + glossary (highest ROI, no retrain).**
   The one thing the fine-tune helped with is names, and that is exactly what biasing
   targets, without the onset/spelling regressions. faster-whisper supports `initial_prompt`
   and `hotwords`; feed the per-meeting roster (`eval/fetch_rosters.py`) and relevant
   glossary entries. Measure against base and against the fine-tune on the corrected-
   utterance subset. Hypothesis: biasing beats both on name accuracy while leaving general
   WER intact.

3. **Diagnose the training-vs-controlled discrepancy.** Reproduce the training-time
   val_corr eval exactly and find why it reported -32% while the controlled test shows a
   regression. Likely a normalization difference or an unfairly-served baseline. Worth an
   hour because it prevents repeating the mistake, and confirms whether any of the training
   gain was real.

4. **If you retrain the LoRA:** train on **context windows** (surrounding audio, not
   isolated cut utterances) to kill the onset drop; try higher LoRA rank and/or unfreezing
   the encoder; add explicit name-normalization to the correction targets. Gate every run
   through the harness from idea 1 before believing any number.

5. **Reframe the task as text post-editing, not acoustic fine-tuning.** The corrections
   are edits on Scribe's output (`initial_before_text` -> `final_after_text`), and Scribe's
   "before" text was already closer to the human reference than either whisper model
   (0.155 vs 0.158/0.176). An LLM post-editor over Scribe output, conditioned on the
   roster/glossary, may capture the corrections more directly than retraining the acoustic
   model. Cheap to prototype.

6. **Request a recent-only benchmark sample** (meetings after the 2026-05-19 training
   cutoff, all cities) so contamination stops distorting the aggregate. Secondary to the
   above; the controlled local eval is more trustworthy than the benchmark anyway.

## Cautions and gotchas

- Do not trust `bench.opencouncil.gr` base-provider numbers as a fair baseline; the HF
  serverless pipeline decodes poorly. Always compare on your own controlled stack.
- The dataset is under a GDPR/DPO legal hold and stays private; only the model adapter is
  public. Do not publish utterance text or audio.
- Benchmark runs cost real money and the 65-clip sample is ~78% contaminated.
- CTranslate2 `float16` is GPU-only; on CPU use `float32` as the full-precision baseline.
- RunPod: the GraphQL API is behind Cloudflare (Python urllib gets error 1010, use curl);
  serverless is GraphQL-only (`saveTemplate` then `saveEndpoint`), no `runpodctl` support.
- A temporary GPU pod for a benchmark run bills continuously ($0.22/hr); terminate it
  immediately after (a full 260-clip run is ~$0.35).
