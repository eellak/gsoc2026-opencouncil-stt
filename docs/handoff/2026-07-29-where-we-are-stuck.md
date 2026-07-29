# Handoff: where this project is stuck, and what the evidence actually says (2026-07-29)

Audience: the next AI agent or engineer. Your job is **to think, then propose a plan** —
not to start coding. Read this whole file before touching anything. It supersedes parts of
[the 2026-07-25 postmortem](2026-07-25-finetune-eval-postmortem.md), which contains a
**wrong root-cause claim** (flagged below).

## The project in one paragraph

OpenCouncil transcribes Greek municipal council meetings (ASR by Gladia/Scribe). Humans
review and correct the transcripts in a purpose-built UI, producing ~3.9k human-corrected
utterances. The GSoC goal: use those corrections to make ASR better for this domain —
originally by LoRA fine-tuning `whisper-large-v3`. The fine-tune was trained, reported
large wins, and those wins did not survive a controlled test. We are now unsure what is
real, and that uncertainty is the actual blocker.

## Where we are stuck (state this plainly to yourself)

We have **three evaluations that disagree**, and no way to adjudicate between them:

| evaluation | what it said |
|---|---|
| Training-time eval (2026-06-24) | fine-tune **much better**: val_corr WER 33.4→26.7, val_reg 27.1→17.3 |
| Controlled A/B on corrected utts (2026-07-24, n=50) | fine-tune **worse**: base 0.158 vs ours 0.176 |
| Controlled A/B on name-rich utts (2026-07-25, n=59) | fine-tune **better**: base 0.341 vs ours 0.334, and ours+hotwords best at 0.307 |

Until this is resolved, no claim about the model can be trusted, and no migration decision
can be made. **Resolving the contradiction is the critical path**, not training a better
model.

## What is actually established (high confidence)

1. **Contextual biasing with per-meeting roster hotwords improves name accuracy.**
   Base whisper name recall 27.2% → 36.0% (McNemar p=0.021, n=59, 114 gold names), at no
   WER cost. The roster comes from the OpenCouncil API, not from the reference — this is
   not an oracle, and it is deployable today. **This is the only measured, significant,
   reproducible gain in the project.** Report:
   [reports/2026-07-25-hotwords-biasing.md](../reports/2026-07-25-hotwords-biasing.md).
2. **int8 quantization is not a problem.** int8 == float32 on this data (n=34).
3. **The adapter is genuinely active** — outputs differ from base. Not a broken export.
4. **No catastrophic forgetting** — general Greek is intact.
5. **The corrections are mostly real acoustic errors, not formatting.** Category counts
   over all 3,854 included corrections: `substitution_phonetic` 44.7%,
   `punctuation_capitalization` 22.0%, `semantic_rewrite` 9.2%, `insertion` 8.8%,
   `homophone` 8.5%, name categories (person/place/org) ~11% combined. And **90.7% of
   corrections survive** lowercase + accent-strip + punctuation-drop normalization —
   so the task is not a punctuation game, and our normalization is not erasing the signal.
   (This was tested precisely because it was a plausible explanation. It was refuted.)

## CORRECTION to the previous postmortem — read this carefully

The 2026-07-25 postmortem claims the training-time "-32%" was measured against an
unfairly-served baseline. **That claim is wrong.** The code shows the training baseline was
computed by the *same* `Seq2SeqTrainer`, on the *same* clips, with the *same* decode path
and *same* normalization, immediately before training (`notebooks/train_runpod.py:286-292`).
Within its own harness it was a fair comparison.

The unfair-baseline story is true for the **benchmark** comparison (bench.opencouncil.gr
serves base whisper through a poorly-decoding HF pipeline), but *not* for the training
numbers. Anyone reading the postmortem alone will misdiagnose this. Fix that doc.

## The real candidate explanations (ranked, all still unproven)

### (a) Checkpoint selected on the val set itself — optimistic bias
The notebook that produced the numbers ran with
`load_best_model_at_end=True, metric_for_best_model="wer"` over 4 epochs, scored on
`val_corr` — the very set that was then reported. The baseline received no such selection.
With n=191 and 4 candidate checkpoints this can inflate the result by several points.
Later turned off (`train_runpod.py:272`), but the numbers were never re-run.

### (b) `val_reg` measured imitation, not accuracy
`val_reg`'s reference text is the **unedited Gladia/Scribe ASR output**
(`train_runpod.py:340-348`), not a human transcript. So "val_reg WER 27.1→17.3 (−36%)"
largely measures how well the LoRA learned another ASR system's house style (numerals,
casing, punctuation). **The fact that ordinary speech improved *more* than corrected speech
was a red flag that was read as good news at the time.** Treat any val_reg number as
uninterpretable.

### (c) Decoder stack — the biggest open question
Training eval used HuggingFace `Seq2SeqTrainer.predict_with_generate`, fp16, **greedy**
(`generation_num_beams` never set), max 225 tokens. The controlled eval used
faster-whisper/CTranslate2, **beam=2**, float32, `condition_on_previous_text=False`.

The suspicious fact: **base whisper scores WER 33.4 in the training harness but 0.158 in the
controlled harness.** Roughly twice as good, same model. Hypothesis worth testing: a large
part of what the LoRA "learned" was compensating for weak greedy HF decoding — an error
class that simply does not exist when the model is served properly. If true, the fine-tune
was solving a problem created by the eval harness.

Caveat: the two numbers are not directly comparable (different samples n=191 vs n=50,
different normalization: raw WER vs accent/punct-stripped). **That is exactly what the
running experiment below is designed to disentangle.**

### (d) Eval mirrored training too closely
Both training and eval used **isolated cut utterances** (±0.2 s pad, then zero-padded to
whisper's 30 s window). The model performed well in exactly the conditions it was trained
for, and degrades on continuous meeting audio — the observed onset-drop (dropping leading
words). Any eval built from isolated clips will overstate real-world performance.

### (e) Known unfixed bug
The run used `clean_up_tokenization_spaces=True` on the Whisper tokenizer, which corrupts
decoded text. Applied symmetrically to predictions and references, but the absolute numbers
were flagged as unreliable at the time and never re-run.

## The experiment now running (results will land in `eval/diagnose/`)

Designed to decompose the training-vs-controlled gap into its parts, on **one fixed sample**
of held-out corrected utterances, scored with **both** raw and normalized WER:

- Stage 1 (faster-whisper, larger n): base and ours × beam=2 and greedy(beam=1).
  Isolates decoder search width and normalization.
- Stage 2 (HF transformers `generate`, greedy, smaller n): base and ours through the
  *actual training eval code path*. Bridges the two worlds.

Read the results before planning. If base whisper collapses under greedy/HF and recovers
under beam=2, explanation (c) is confirmed and the original "-20%" is explained away.

Note: the exact 191-clip `val_corr` set **cannot** be reproduced — `export.jsonl` has grown
since June (3,854 rows now; the June run had 1,964 train clips). The experiment uses all
currently-eligible held-out corrected clips instead, which is a larger and better sample but
not a literal replication.

## Assets

Data (all local; **private, GDPR legal hold — never publish text or audio**):
- `data/asr/export.jsonl` — 3,854 corrections: `initial_before_text`, `final_after_text`,
  `edits`, `error_categories`, `include_status`.
- `data/asr/val_manifest.csv` — 9,875 held-out utterances (argos + orestiada).
- `data/asr/audio/*.mp3` — 368 full meeting recordings, `{city}__{meeting}.mp3`.
- `data/glossary/glossary.json` — entity vocabulary. `data/pii/rosters_full.json` — per-meeting
  speaker rosters for 311 meetings (use this one; `data/improve_loop/rosters.json` covers only 73).

Models:
- Adapter (public): `opencouncil/whisper-large-v3-el-council-lora`. LoRA r32/alpha64,
  q_proj+v_proj, encoder frozen, 2 epochs, lr 1e-4, `VAL_CITIES={argos, orestiada}`.
- Local merged: `/home/harold/oc-asr-serve/merged`; shipped ct2 int8: `/home/harold/oc-asr-serve/ct2`.
- Eval env: `.venv-eval` (faster_whisper, ctranslate2, transformers, ffmpeg). CPU only →
  use `float32`; `float16` is GPU-only in CTranslate2.

Code:
- `eval/controlled_eval/` — the four A/B scripts + raw results. Ad-hoc, share code by
  copy-paste, hardcoded scratch paths.
- `notebooks/train_runpod.py` — training + eval path (the port of the notebook that ran).
- `eval/fetch_rosters.py` — roster fetching for biasing.

## What to think about, and what a good plan looks like

Do not jump to "retrain with better hyperparameters". The evidence does not support that as
the highest-value move. Questions worth reasoning about:

1. **Is the acoustic fine-tuning framing even right?** Scribe's raw output (0.155) is
   already closer to the human reference than either whisper model (0.158 / 0.176) on the
   corrected subset. If the corrections are largely a *text* problem, an LLM post-editor
   conditioned on roster+glossary may capture them far more directly than any acoustic
   fine-tune. Cheap to prototype, never tried.
2. **Should biasing simply be the deliverable?** It is the one thing that works, it needs
   no training, and it is deployable in the OpenCouncil pipeline (the roster is available at
   inference time). What would it take to make that a defensible, well-measured contribution
   rather than a single n=59 run? (Widening the name subset to n in the low hundreds is the
   obvious first step.)
3. **What is the minimum trustworthy eval** that would let any future claim be believed?
   Same stack, same normalization, three held-out subsets reported *side by side* (general /
   name-rich / actually-corrected) so a config cannot look good on one and hide on another,
   with CIs and per-utterance head-to-head. The existing four scripts are 80% of this and
   need factoring, not rewriting.
4. **If retraining is justified after all**, the evidence points at: context windows instead
   of isolated clips (kills onset drop), no checkpoint selection on the reported set, a
   held-out set never used for selection, and human references only — never another ASR's
   output.
5. **Scope reality check.** This is a GSoC project with a deliverable and a deadline. A
   well-measured, honestly-reported negative result plus one working technique (biasing) is
   a legitimate and defensible outcome. Consider whether the plan should optimize for
   "a better model" or for "a trustworthy, useful contribution".

## Cautions

- Dataset is under GDPR/DPO legal hold. Stays private. Only the model adapter is public.
- Do not trust `bench.opencouncil.gr` base-provider numbers as a fair baseline (HF pipeline
  decodes poorly there). The 65-clip sample is ~78% training-contaminated; only argos and
  orestiada are genuinely held out.
- CTranslate2 `float16` is GPU-only; use `float32` on CPU.
- RunPod GraphQL API is behind Cloudflare — use `curl`, not Python urllib (error 1010).
  GPU pods bill continuously ($0.22/hr); terminate immediately after use.
- `rtk` is referenced in CLAUDE.md but is **not installed** on this machine; use git directly.
- Small-n results (n=34/50/59) are datapoints, not verdicts. Report CIs.

---

## RESULT (2026-07-29, same day): the contradiction is explained

The experiment ran on **n=300** held-out corrected utterances. Normalized WER (`gnorm`),
reference = human `final_after_text`:

| config | all (n=300) | long only, 4-30s & >=6 words (n=84) | short/short-text (n=216) |
|---|---|---|---|
| scribe_before (production ASR) | 22.39 | **15.59** | 30.40 |
| base, faster-whisper beam=2 | 27.08 | 16.99 | 38.96 |
| **ours, faster-whisper beam=2** | **22.66** | 16.43 | **29.98** |
| base, HF greedy | 28.25 | 17.20 | 41.27 |
| ours, HF greedy | 28.18 | 18.18 | 39.95 |

Bootstrap (2000 resamples) on the ours−base delta, faster-whisper beam=2:

| subset | delta | 95% CI |
|---|---|---|
| all (n=300) | **−4.35 pp** | [−6.86, −2.28] |
| long only (n=84) | −0.41 pp | [−3.44, +1.73] |
| short (n=216) | **−8.98 pp** | [−12.80, −5.46] |

### What this means

1. **The n=50 "regression" was a sampling artifact.** That run filtered to `4-30s AND
   >=6 words`. On exactly that filter here, ours ≈ base (−0.41 pp, CI straddles zero) —
   the old +0.018 was noise. The filter **systematically excluded the utterances where the
   fine-tune helps.**
2. **The fine-tune's gain is concentrated in SHORT utterances** (−8.98 pp, CI far from
   zero). Base whisper degrades badly on short isolated clips (38.96); the fine-tune, trained
   on exactly such clips, does not (29.98).
3. **Therefore "do not migrate" was based on a subset that hid the effect.** That conclusion,
   and the postmortem built on it, must be revisited.

### But do NOT declare victory — the load-bearing caveat

The gain appears on **isolated short clips**, which is exactly the condition the model was
trained on and exactly how this eval cuts audio. Production transcribes **continuous meeting
audio**, where "short utterance" is not a thing the decoder sees in isolation. **This gain may
be an artifact of the eval setup mirroring the training setup** (candidate cause (d) above).
Nothing here measures full-meeting performance. That is the next thing to test, and it is
the only test that can justify migration.

Two more sober facts:

- **Scribe (the current production ASR) is still the best overall** on long utterances
  (15.59 vs ours 16.43) and only slightly behind on short. Whisper+LoRA is not clearly
  better than what OpenCouncil already runs.
- **The gain only appears under faster-whisper**, not under HF greedy (28.18 vs 28.25 —
  a tie). Unexplained, and worth understanding before trusting the number.

### Ruled out

- `clean_up_tokenization_spaces` — True and False produced **identical** WER. Not a factor.
- Hypothesis (c) as originally stated is **wrong**: the fine-tune did not learn to compensate
  for weak greedy decoding. The opposite — its advantage shows up under beam search and
  disappears under greedy.

Raw data: `eval/diagnose/results.json`, script `eval/diagnose/decompose_training_gap.py`.
