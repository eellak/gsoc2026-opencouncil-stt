# GSoC 2026 Final Report — Fine-tuning ASR for Greek Municipal Councils (DRAFT)

**Status: DRAFT, 2026-08-23.** Placeholders of the form `<<TBD: …>>` mark numbers
that do not exist yet; they are collected in [Pending measurements](#pending-measurements)
so they can be filled in mechanically. This report supersedes and extends the
internal final report of 2026-08-12
([`2026-08-20-final-report.md`](2026-08-20-final-report.md), experiment
`exp-2026-08-20-final-report`); nothing in that report is contradicted here —
everything after it is extension, measured later. Canonical research state:
[`research/ledger.json`](../../research/ledger.json). Every number below names the
ledger experiment or report that produced it.

Project: GSoC 2026 with GFOSS / OpenCouncil — *Fine-tuning AI transcription models
for Greek Municipal Councils*. Student: Angelos Papamichail. Mentors: Christos
Porios, Andreas Kouloumos.

---

## 1. Summary — the honest headline

**The proposal's headline target was not met, and the project's most valuable
result is not the model.**

- The published fine-tuned adapter (`artifact-adapter-fixed`, LoRA on
  whisper-large-v3) **does beat base whisper-large-v3** on unseen cities and beats
  its own bug-affected predecessor — but it **does not beat the commercial systems**
  (ElevenLabs Scribe v2, Soniox): both confidence intervals contain zero
  (`exp-2026-08-10-benchmark-fixed-adapter`).
- On **domain terms — the thing the project set out to fix — it is clearly worse
  than the best commercial system**: DS-WER 0.4880 for our adapter against 0.3280
  for Soniox (`exp-2026-08-12-ds-wer`). The proposal's 15–20% relative DS-WER
  target is met only against the Gladia baseline, only on the point estimate, and
  its interval includes zero. Stated plainly: **the target was not met**.
- **The gap to the best commercial system is 4.18 WER points, and it is now
  itemised.** It is substitutions (+0.0533) and deletions (+0.0221); we are
  *better* than Scribe on insertions (−0.0337), writing fewer words that were
  never said. Spelling is not the cause: homophone misspellings, the part a
  decoder could fix, are worth 0.0066 of the 4.18. The largest recoverable bucket
  is words heard as entirely different words, which needs better listening rather
  than better spelling. 36% of what we delete disappears in runs of five or more
  consecutive words, against Scribe's 19%
  ([`2026-08-23-gap-to-scribe.md`](2026-08-23-gap-to-scribe.md)).
- What *was* achieved: a fine-tuned model that improves on the open-source
  baseline; a frozen, preregistered **evaluation framework** that caught several
  false wins before they shipped; a large body of **negative results** that closed
  expensive directions with evidence; and one genuinely large positive discovery —
  **composing multiple ASR systems word-by-word (no LLM) cuts WER from 0.1201 to
  0.1005 and lowers all three error components at once**
  (`exp-2026-08-16-composition-over-selection`). Composition is a bigger lever
  than fine-tuning ever was in this project, and it is the recommended production
  path.

Two different quantities appear throughout and are never merged:

- **agreement-with-OpenCouncil** — WER against OpenCouncil's own published
  transcript. It measures product compatibility. Every benchmark number below is
  this quantity unless marked otherwise.
- **fidelity-to-audio** — WER against a human who listened. This is the quantity
  that actually decides quality. It was measured once on a frozen human-verified
  gold set (`exp-2026-08-16-gold-set`), and the system ranking **flipped between
  scoring regions**, so no ranking is claimed from it.

---

## 2. Goals, as proposed

From the original proposal
([`docs/reference/gsoc-proposal.md`](../reference/gsoc-proposal.md)):

| Goal | Target |
|---|---|
| Milestone 1 | Reproducible dataset on HuggingFace + baseline report |
| Milestone 2 | LoRA adapter with **≥15% relative DS-WER improvement over Gladia** |
| Final milestone | Production-ready pipeline merged into OpenCouncil |
| Operational target | Measurable drop in Human Intervention Rate (HIR) |
| Metrics | Normalized WER/CER, DS-WER on domain terms, HIR |

---

## 3. Deliverables — what was actually delivered

### 3.1 Preprocessed dataset

**Built, not publishable.** The correction-pair extraction pipeline works
(`eval/build_dataset.py` and the `eval/hf_export/` tooling); the control training
set is a 28,967-row parquet (136 meetings, 9 cities, ~22.5 h) plus a newer
overlap-free contiguous-pack dataset (2,475–2,476 packs, ~18.6 h,
[`2026-08-19-overlap-clean-selection.md`](../specs/2026-08-19-overlap-clean-selection.md)).

The dataset itself is on **legal hold** (Data Protection Officer decision,
2026-07-17): text-level PII removal does not anonymise it, because every row
links to audio carrying the speaker's voice
([`docs/decisions/data.md`](../decisions/data.md)). The blocker is legal, not
technical. Milestone 1's "dataset on HuggingFace" is therefore **not delivered as
a public artifact**; the extraction and filtering pipeline is delivered as code.

### 3.2 Fine-tuned model

**Delivered and published.** `artifact-adapter-fixed` (LoRA r=32, α=64, q/v
projections, ~22.5 h, 2 epochs) is public at
[`opencouncil/whisper-large-v3-el-council-lora`](https://huggingface.co/opencouncil/whisper-large-v3-el-council-lora)
— corrected weights since 2026-08-16 (hub commit `e214de71`). A merged
CTranslate2 int8 build (`artifact-ct2-fixed`) is what the serving endpoint runs.

Its measured record is in [section 4](#4-results-and-their-limits). Short form:
better than base whisper, indistinguishable from Scribe v2 and Soniox on overall
agreement-WER on unseen cities, clearly worse than Soniox on domain terms.

It is plausibly the best openly available Greek ASR adapter for council-style
speech, but that claim is **untested outside our own domain**:
**Attempted, withdrawn, not repeated in time.** FLEURS `el_gr` was run on 2026-08-22 for
base whisper-large-v3, `artifact-adapter-fixed` and a clean-pack adapter on one stack with
the frozen decode config. The result was then found invalid: FLEURS' `id` field is the
*sentence* id, not the recording id, so the 650-row test split collapsed to 333 keys, wav
files overwrote each other, and roughly 317 rows were scored against the wrong audio. The
numbers were withdrawn rather than published
([report](2026-08-22-fleurs-out-of-domain.md), `exp-2026-08-22-fleurs-out-of-domain`).
The scripts now key by row index and refuse to emit a result unless every row survives;
the corrected rerun is open work. VoxPopuli would have been the better match — European
Parliament speech, the closest public analogue to council proceedings — but its
transcribed ASR subset covers 16 languages and Greek is not one of them..

### 3.3 Evaluation framework

**Delivered, and it is the part of the project that held everything else
honest.** Components:

- A frozen evaluation set
  ([`research/eval-freeze-2026-08/manifest.json`](../../research/eval-freeze-2026-08/manifest.json)):
  39 validation windows from the two training-disjoint cities (argos, orestiada),
  31 meetings, 11,911 reference tokens, frozen 2026-08-11 **before** any arm it
  judged was decoded, plus 7 temporal holdout windows that remain **sealed** —
  no experiment ever passed a gate that would have released them.
- A frozen Greek-aware normalizer and an external scorer with meeting-clustered
  bootstrap confidence intervals (`eval/controlled_eval/`).
- Preregistered gates for every training screen
  ([`docs/decisions/training-evidence.md`](../decisions/training-evidence.md)):
  paired seeds, deletion and insertion guards, leave-one-out stability, and a
  single-window domination check.
- Integration with OpenCouncil's own benchmark HTTP API (ledger capability
  `cap-bench-http-api`).
- A frozen human-verified **gold set** for fidelity-to-audio (27 cores, 6
  meetings, 6 cities; `exp-2026-08-16-gold-set`), kept as a one-shot challenge
  set.
- A frozen 300-row training-WER diagnostic sample
  (`exp-2026-08-18-training-wer`) reported alongside every training run,
  diagnostic only, never used for arm selection.

The publication plan for this framework is in
[section 10](#10-publishing-the-evaluation-harness).

### 3.4 Production integration

**Partial.** What exists: the merged CT2 model served via faster-whisper on a
scale-to-zero RunPod serverless endpoint; a served long-form decode policy
(`serve/decode_p.py`, `artifact-decode-policy-p`,
[`2026-08-21-decode-p-served.md`](2026-08-21-decode-p-served.md)); and the
benchmark-side plumbing to score any provider.

What does not exist: the adapter has **not** replaced the production transcriber
in OpenCouncil. During the project the product independently moved to ElevenLabs
Scribe v2, which our own measurements confirm is at least as good as our adapter
on overall agreement-WER and better on deletions. The honest production
recommendation is not "swap in our model" but **fusion**
([section 4.4](#44-the-composition-result)), specified for a two-system
production variant in
[`2026-08-21-fusion-production.md`](../specs/2026-08-21-fusion-production.md)
(`exp-2026-08-21-fusion-production`, OPEN, nothing measured yet).

**Now delivered.** Both adapters were registered as named providers on one
pod, one decoder stack, and scored on the same 391 held-out post-June windows
([section 4.7](#47-both-adapters-inside-the-products-own-benchmark)).

### 3.5 Documentation

**Delivered, arguably over-delivered.** The repository
([github.com/eellak/gsoc2026-opencouncil-stt](https://github.com/eellak/gsoc2026-opencouncil-stt))
carries a machine-checkable research ledger
([`research/ledger.json`](../../research/ledger.json), 78 experiment records with
status, conclusions and caveats), ~80 dated reports under `docs/reports/`,
preregistration specs under `docs/specs/`, runbooks, and decision records. Every
conclusion in this report traces to one of them.

### 3.6 Human Intervention Rate

**Never measured.** The proposal named HIR reduction as the operational target.
No before/after HIR measurement was ever run, because the model never entered
production ahead of the correction workflow. This is a plain miss, not a pending
number.

---

## 4. Results and their limits

### 4.1 The adapter against its baselines (agreement-with-OpenCouncil)

`exp-2026-08-10-benchmark-fixed-adapter`, 39 frozen validation windows, metric
`wer-nofillers`, 90% intervals:

| comparison | Δ WER (points) | interval | verdict |
|---|---|---|---|
| vs Scribe v2 | +1.12 | [−0.54, +2.73] | tie |
| vs Soniox | +1.19 | [−0.48, +2.70] | tie |
| vs base whisper-large-v3 | **−1.87** | [−3.35, −0.50] | we win |
| vs our own bug-affected predecessor | **−1.58** | [−2.41, −0.85] | we win |

Limits, from the ledger record itself: the base-whisper arm ran on a different
decoder stack (HuggingFace serverless), so the −1.87 includes an engine
difference and is **not a clean weights effect**; one seed; argos+orestiada is
validation, scored repeatedly, so potentially optimistic. A same-stack paired
contrast exists on the frozen training sample (`exp-2026-08-18-training-wer`):
base 0.2728 vs adapter 0.1313, delta +0.1415 [0.1138, 0.1725] — same machine,
same decoder — but that is training-domain material, not held-out validation.

### 4.2 DS-WER — the headline target, not met

`exp-2026-08-12-ds-wer`, 39 frozen windows, 250 domain-term occurrences (roster
surnames + city names; the list is a lower bound on domain vocabulary):

| system | DS-WER |
|---|---|
| Soniox | 0.3280 |
| Scribe v2 | 0.3720 |
| **ours (`artifact-ct2-fixed`)** | **0.4880** |
| base whisper-large-v3 | 0.5400 |
| Gladia (proposal baseline) | 0.5880 |

Against Gladia the relative improvement is +17.0% **on the point estimate only**;
the 95% meeting-block interval on the absolute delta is [−0.1869, +0.0040] and
crosses zero. Against the systems that matter today, we make roughly 50% more
domain-term errors than Soniox in relative terms. Excluding the two roll-call
windows, our DS-WER (0.4751) is indistinguishable from base whisper's (0.4696).
**The 15–20% relative DS-WER goal, read as originally intended — a meaningful
improvement on the words that matter — was not achieved.**

The useful finding inside the failure: our name errors are 90 substitutions to
28 deletions — the name is there, wrong by one or two characters. That priced
the roster-based repair work (below) and the decoder-focused retraining proposal
([section 9](#9-what-comes-next)).

### 4.3 Fidelity-to-audio — measured once, and it disciplined everything

`exp-2026-08-16-gold-set`, human-verified reference on untouched meetings (27
cores, 6 meetings, 6 cities): our adapter scores fidelity-WER **0.284**
[0.169, 0.455]; the published OpenCouncil pipeline output scores 0.198 on the
same audio — but the ranking **reverses** under a stricter scoring region, so
**no system ordering is claimed**. Three things it did settle:

- 4 of every 5 words that a second system emits and ours lacks were **really
  said** (53/66 human-supported).
- The production pipeline leaves **5.8% of spoken blocks with no published
  utterance at all** and loses 28.6% of certain words inside speaker overlap.
- Agreement-WER and fidelity-WER are different quantities, not a correctable
  offset (`exp-2026-08-04-ranking-flips` first showed the ranking flip).

A companion audit (`exp-2026-08-17-insertion-fidelity`) showed 23.7% of the
adapter's scored "insertions" (and 40.8% of Soniox's) sit on words the human
heard and the published text missed — the benchmark partly charges systems for
being right.

### 4.4 The composition result

`exp-2026-08-16-composition-over-selection`, 247 benchmark windows
(agreement-with-OpenCouncil): an exact three-way word alignment of Scribe v2 +
Soniox + our adapter with a per-column vote — **no LLM, no audio, no speaker
information** — takes WER **0.1201 → 0.1005** (−0.01966 [−0.02292, −0.01665])
and lowers deletions, insertions **and** substitutions simultaneously, every CI
excluding zero, no leave-one-out sign flip over windows, meetings or cities. The
composed output lands *below* the whole-window oracle of the three systems: the
composed text is one none of the three produced.

This is the largest measured effect in the project — roughly the size of the
entire fine-tuning gain over base whisper, on top of already-strong systems —
and it is where the fine-tuned model earns its keep: **as one voter among
several, not as a solo replacement**. The production-shaped two-system variant
is specified but unmeasured
([`2026-08-21-fusion-production.md`](../specs/2026-08-21-fusion-production.md)).
A caution recorded with it: whole-window *selection* with two systems was tried
and failed (0.1349, `exp-2026-08-02-asr-fusion`); composition with two systems
has never been measured.

A small related positive: roster-grounded phonetic name repair on top of the
composition survives on the full 247 windows (−0.00075 [−0.00109, −0.00044],
`exp-2026-08-11-name-repair`) — real but tiny, shadow-only.

### 4.5 The negative results, which redirect the product

Each of these is a CLOSED ledger record with evidence; none should be re-run:

- **Decode thresholds** (`exp-2026-08-12-decode-ablation`): the no-speech gate
  fires zero times in 39 windows; removing the temperature fallback makes every
  metric worse. Our deletions are in the weights, not the decoder settings.
- **Label purity** (`exp-2026-08-13-correction-only`): dropping the unverified
  half of the data moves WER +0.0015 — inside noise fourteen times larger.
- **Data scale** (`exp-2026-08-11-wer-levers-research`): the dominant residual
  error is homophone orthography the audio cannot decide; published analogues
  price ~1,300 h of data for ~0.5 WER points.
- **Deletion-targeted training** (`exp-2026-08-13-targeted-deletion-training`):
  raised the deletion rate it was built to lower.
- **External Greek corpora as stage-1 packs** (`exp-2026-08-14-external-packs`):
  no detectable change on top of the control.
- **Dense 30-s packing** (`exp-2026-08-19-dense-screen-300`): improved WER and
  deletions but failed the preregistered insertion and domination guards —
  STOP.
- **Serving-time repair of deletions** (`exp-2026-08-12-serving-stack`): the
  missing words are absent from all 8 beam hypotheses; no decode-time technique
  reaches them.

### 4.6 The two trainings compared — the incumbent recipe vs the clean-pack recipe

The scope asked for this comparison explicitly.

**Training 1 (the incumbent, best so far).** `artifact-adapter-fixed`: 28,967
single-utterance clips (mean 3.55 s), corrections + no-edit backbone, ~22.5 h, 2
epochs. This is the published model. On the frozen training sample its training
WER is 0.1313 (`exp-2026-08-18-training-wer`); correction rows fall from 0.4471
(base) to 0.2261, so it demonstrably learned its correction signal.

**Training 2 (the new perspective).** Instead of more data, *cleaner and more
realistic* data: contiguous single-speaker spans, overlap-free by diarization
filter, acoustically placed boundaries, ~22 s of speech per ≤29 s window —
shaped like what the model sees at inference
([`2026-08-19-overlap-clean-selection.md`](../specs/2026-08-19-overlap-clean-selection.md),
[`2026-08-20-clean-pack-screen-prereg.md`](../specs/2026-08-20-clean-pack-screen-prereg.md)).
A 12-pack human listening spot-check passed 12/12 on single-speaker and
text-matches-audio.

At **300 steps** (paired, 3 seeds, one GPU stack,
[`2026-08-21-clean-pack-screen.md`](2026-08-21-clean-pack-screen.md)) the
clean-pack arm beat the incumbent recipe in **all three seeds** (mean ΔWER
−0.00982, deletions and insertions both down) — a real early signal from very
few steps, exactly as the scope notes. It still failed the frozen
single-window-domination gate, so no promotion.

At **1,800 steps**
([`2026-08-22-clean-pack-medium.md`](2026-08-22-clean-pack-medium.md),
`exp-2026-08-22-clean-pack-medium`) the effect **dissolved**: mean ΔWER
−0.00252, one seed reversing sign, across-seed spread up ~4×. The diagnosis is a
design flaw, now recorded: with a fixed *step* budget the two arms trained to
wildly different points on their curves (control: 0.50 epochs; packs: 5.82
epochs). Budgets must be matched in **epochs**, not steps (repo issue
[#49](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/49)). A further
quantified confound: only 2.5% of pack utterances are corrections — the packs
diluted the correction signal ~1:8 relative to the control.

**Training 3 (the epoch-matched rerun).** The corrected comparison, 619 steps
(≈2 pack epochs on the contiguous pack), run 2026-08-22 on one RTX A4000. Every
adapter was scored by the same external scorer on the same 39 frozen validation
windows (11,911 reference tokens, 32 meetings), with the frozen `CONTROL` decode
config and a per-window ctranslate2 seed. **The incumbent was re-scored on this
same stack** so the comparison is same-machine, same-decoder, same normalizer —
the earlier 0.1542 figure for it came from a different stack and is not used here.

The decision rule was declared before any number was seen: ship a challenger only
if the **upper** bound of the paired, meeting-clustered bootstrap interval against
the incumbent falls below zero, Bonferroni-corrected across the three arms
(98.33%). Seed 13 was pre-declared as the shipping seed, so no seed selection
occurs. Step 619 is confirmatory; intermediate checkpoints are exploratory.

| adapter | WER | del | ins | sub | ΔWER vs incumbent | Bonferroni 98.33% CI | gate |
|---|---:|---:|---:|---:|---:|---|---|
| incumbent (`artifact-adapter-fixed`) | 0.16002 | 0.06196 | 0.02149 | 0.07657 | — | — | — |
| contiguous seed 13 *(ship seed)* | 0.14566 | 0.04450 | 0.02376 | 0.07741 | −0.01436 | [−0.02671, **−0.00171**] | **pass** |
| contiguous seed 29 | 0.13995 | 0.03980 | 0.02334 | 0.07682 | −0.02007 | [−0.03747, **−0.00595**] | **pass** |
| contiguous seed 47 | 0.13895 | 0.03971 | 0.02107 | 0.07816 | −0.02107 | [−0.03895, **−0.00736**] | **pass** |
| combined seed 13 | 0.15255 | 0.03896 | 0.03350 | 0.08009 | −0.00747 | [−0.02742, +0.01173] | fail |
| combined seed 29 | 0.15028 | 0.04047 | 0.02964 | 0.08018 | −0.00974 | [−0.03092, +0.00752] | fail |
| combined seed 47 | 0.14524 | 0.03753 | 0.02896 | 0.07875 | −0.01478 | [−0.03109, **−0.00060**] | marginal |

**The contiguous arm passes in all three seeds.** Relative improvement over the
incumbent is 9.0% (seed 13) to 13.2% (seed 47).

**It wins by deleting less, which is the direction that matters.** The incumbent
omits 6.20% of reference tokens; the contiguous adapters omit 3.97–4.45%, roughly
a third fewer. Substitutions are flat and insertions are flat to slightly up. This
project's standing warning is that a model which lowers WER by skipping hard
passages looks better and is worse; here the opposite happened — the gain *is* the
recovered speech.

Robustness checks, all pre-specified: 29 of 39 windows improved; the largest
single-window contribution is 24–28% of the total effect, well clear of the 67%
single-item domination that has misled this project before; the interval is
clustered by meeting, not by window.

**The combined arm — contiguous plus jitter-spliced packs — fails, and fails for a
legible reason.** Neither seed's interval clears zero against the incumbent, and
head-to-head against its own contiguous twin the combined arm is *worse*: +0.00688
for seed 13 (95% CI [−0.00212, +0.01587]) and +0.01033 for seed 29 (95% CI
[+0.00168, +0.01919], excluding zero). The mechanism is visible in the error
breakdown: insertions rise from 2.3% to 3.0–3.4% while deletions stay flat. The
synthetic joins — jittered 0.15–0.60 s separators with short fades, designed
specifically so no fixed separator could become a learnable reset cue — appear to
have taught the model to keep going across a boundary where it should stop.
**Splicing is not a deliverable. It is a recorded negative result with a
mechanism.**

- decoder-only LoRA: **not run.** It was designed as a fallback and the fallback was never
  needed: the contiguous arm passed in all three seeds, so the GPU budget went to the
  held-out benchmark instead. The hypothesis it would have tested — that the encoder has
  little headroom and the gain lives in the decoder — remains open, and the code to run it
  exists (`scripts/armb/run_decoder_only.sh`, `LORA_SCOPE=decoder`).

Supporting diagnostics for the new dataset:

- **not run.** Training-set WER is a diagnostic, not a claim about quality, and with a
  fixed deadline the GPU went to held-out measurement instead. Recorded as open.

**What this does and does not license.** These 39 windows are the project's
**validation** set, from the held-out cities argos and orestiada. They have been
scored repeatedly across this project, so a validation win is a reason to promote
a candidate to testing, not a reason to claim a product improvement. The claim
"the new adapter beats the published one" is supported *on validation, on one
stack, under a pre-declared rule*. It is not yet supported on the OpenCouncil
benchmark or on the 16 sealed evaluation windows, and the report does not make
the stronger claim until those run.

Two caveats inherited from the design, both unchanged by this result: only 2.5% of
pack utterances are corrections, so the packs dilute the correction signal ~1:8
relative to the incumbent's training set; and the incumbent's own recipe was never
re-run epoch-matched on its own data, so part of the gap could be the epoch budget
rather than the data. Neither is resolved here.

---

### 4.7 Both adapters inside the product's own benchmark

`exp-2026-08-23-post-june-held-out`, 391 windows across 117 city-meetings, none of
which appears in any training pack. Both adapters were served from the same pod
through the same faster-whisper CT2 float16 stack, with the served weights checked
by sha256 before each arm; the commercial providers ran on the same window list.
The metric is agreement-with-OpenCouncil (`wer-nofillers`), which records product
compatibility and decides nothing about fidelity.

| system | WER | del | ins | sub |
|---|---|---|---|---|
| ElevenLabs Scribe v2 | 0.1339 | 0.0094 | 0.0693 | 0.0552 |
| Soniox | 0.1455 | 0.0122 | 0.0703 | 0.0631 |
| **clean-pack contiguous s47** (new) | **0.1827** | **0.0313** | 0.0415 | 0.1099 |
| `artifact-adapter-fixed` (incumbent) | 0.1867 | 0.0525 | 0.0318 | 0.1024 |
| gpt-4o-transcribe | 0.1937 | 0.0506 | 0.0312 | 0.1120 |
| base whisper-large-v3 | 0.1988 | 0.0335 | 0.0428 | 0.1224 |
| Gladia (proposal baseline) | 0.2085 | 0.0405 | 0.0435 | 0.1246 |

Paired meeting-clustered bootstrap, 10,000 resamples, new adapter minus the other
system (negative favours the new adapter):

| contrast | delta | 95% CI | reading |
|---|---|---|---|
| vs incumbent `artifact-adapter-fixed` | −0.0040 | [−0.0078, **+0.0002**] | **crosses zero — not a confirmed win** |
| vs base whisper-large-v3 | −0.0161 | [−0.0218, −0.0114] | confirmed |
| vs gpt-4o-transcribe | −0.0111 | [−0.0229, −0.0014] | confirmed |
| vs Gladia | −0.0259 | [−0.0325, −0.0197] | confirmed |
| vs Soniox | +0.0371 | [+0.0291, +0.0461] | confirmed loss |
| vs Scribe v2 | +0.0488 | [+0.0396, +0.0588] | confirmed loss |

**On overall WER the new recipe does not beat the incumbent.** The interval touches
zero and the pre-declared rule requires the upper bound to fall below it. That rule
was fixed before any of these numbers existed and is not being relaxed now.

What did change is the composition of the errors:

| rate, new minus incumbent | delta | 95% CI |
|---|---|---|
| deletions | **−0.0212** | [−0.0251, −0.0174] |
| insertions | +0.0097 | [+0.0066, +0.0134] |
| substitutions | +0.0075 | [+0.0054, +0.0098] |

The new adapter omits roughly **40% less** of what was said, and pays for it in
words it gets wrong rather than words it drops. This is the direction section 4.3
argues matters: a model that lowers WER by omitting hard passages looks better and
is worse, and the incumbent's 0.0525 deletion rate was its worst property. For a
correction workflow, where a human must notice what is missing, a wrong word is
cheaper to fix than a silent gap.

**Caveats, both load-bearing:**

- **Single-window domination.** The net WER difference is −441 tokens over 110,610
  reference tokens, and the five largest windows carry **44.2%** of it. The overall
  WER delta is not broadly distributed. The deletion-rate result is a rate over the
  full 110,610 tokens and does not share this weakness.
- **No per-window decode seed.** The served endpoint does not seed ctranslate2 per
  window, unlike the offline screens. Windows that fall through to the temperature
  ladder are not bit-reproducible. This affects both arms identically.

The honest one-line summary: **the clean-pack recipe buys a large, well-measured
reduction in deletions and does not buy a lower overall WER**, and both facts belong
in the same sentence.

---

### 4.8 Composition, retested on held-out meetings

`exp-2026-08-23-fusion-postjune`. The composition result of section 4.4 was measured
on a 247-window set whose adapter row was the July model. This repeats it on the 391
held-out post-June windows, with the current adapter, on one scorer. No LLM, no audio,
no speaker information: three transcripts in, one text out.

| arm | WER | del | ins | sub |
|---|---|---|---|---|
| `oracle_msa` — best entry per column, a ceiling not a system | 0.0611 | 0.0099 | 0.0296 | 0.0216 |
| **W — exact 3-way MSA + per-column vote** | **0.1202** | 0.0120 | 0.0594 | 0.0487 |
| `oracle_win` — best whole hypothesis per window, a ceiling | 0.1206 | 0.0115 | 0.0576 | 0.0514 |
| V — whole-window consensus vote | 0.1325 | 0.0129 | 0.0643 | 0.0552 |
| Scribe v2 (prespecified comparator) | 0.1377 | 0.0096 | 0.0757 | 0.0524 |
| Soniox | 0.1475 | 0.0124 | 0.0751 | 0.0601 |
| clean-pack adapter | 0.1795 | 0.0317 | 0.0421 | 0.1058 |

Paired meeting-clustered bootstrap over 117 meetings, 10,000 resamples:

| contrast | delta | 95% CI |
|---|---|---|
| **W vs Scribe v2** | **−0.0175** | [−0.0215, −0.0134] |
| W vs V | −0.0123 | [−0.0141, −0.0105] |
| V vs Scribe v2 | −0.0052 | [−0.0095, −0.0008] |
| W vs `oracle_win` | +0.0004 | [−0.0029, +0.0030] |
| W vs `oracle_msa` | +0.0591 | [+0.0533, +0.0653] |

Three things follow, and the third is the one worth keeping.

**W beats the system the product actually runs**, by 1.75 WER points, on meetings no
adapter trained on. Scribe was named as the comparator before the run and also happens
to be the best single system here, so no comparator was selected after the fact.

**W has reached the ceiling of selection.** Its contrast against the whole-window
oracle — the best any method could do by picking whole hypotheses, using the reference
— is +0.0004 with an interval straddling zero. Any further gain has to come from
composing, not choosing. This reproduces on held-out meetings what
`exp-2026-08-16-composition-over-selection` found.

**And composition itself is nowhere near its own ceiling.** Choosing the best entry in
every column of the same alignment gives 0.0611, roughly half of W. The three
transcripts jointly contain a far better transcript than the vote extracts from them.
That gap, not the adapter, is where the remaining accuracy in this project lives.

Robustness, all of it computed before the section was written:

- **Not concentrated.** The five largest windows carry 15.5% of the net −1,939-token
  gain. For contrast, the adapter comparison in section 4.7 sits at 44.2%.
- **Broad across meetings.** W is better in 98 meetings, tied in 3, worse in 16 of 117.
  The largest single meeting supplies 5.4% of the effect.
- **Leave-one-meeting-out.** Dropping any one meeting leaves the delta in
  [−0.0186, −0.0167]. No meeting carries the result.
- **No reference leakage.** Shuffling every reference and recomputing leaves W's text
  and V's pick byte-identical. This is asserted in the script, not claimed in prose.

**Caveats:**

- `oracle_msa` is the best path through **one** alignment lattice, and that lattice
  minimises sum-of-pairs edit cost, not oracle WER. A different valid alignment could
  score lower. It bounds this substrate, not the information in three transcripts.
  Codex's review of the plan is the reason this is stated rather than sold as an
  absolute ceiling.
- Five alternative trios were declared before scoring and are exploratory. The trio
  carrying the **previous** adapter scores 0.1183, marginally better than the trio with
  the new one; trios without both commercial systems are far worse (0.1518, 0.1660).
  No winner among them is claimed.
- This is agreement-with-OpenCouncil, so it decides product compatibility and nothing
  about fidelity to audio.

The production consequence is unchanged and now better supported: the deliverable that
beats what OpenCouncil runs today is **not our adapter, it is composition over several
systems**, and our adapter is one affordable voter inside it.

---

## 5. Timeline of the work

| period | what happened |
|---|---|
| May–June | Community bonding; data extraction from OpenCouncil correction pairs; Greek-aware WER tooling; baselines ([`docs/reports/month-1-2026-06.md`](month-1-2026-06.md)) |
| 2026-07-22/23 | First full LoRA training; adapter published to HuggingFace |
| 2026-07-25 | Same-stack A/B shows the apparent gains were serving artifacts, not model gains (postmortem) |
| 2026-07-31 | **Label-prefix bug found**: every training target was shifted one token; everything trained before 2026-08-01 is invalid (`exp-2026-07-31-label-prefix-bug`) |
| 2026-08-02 | Retraining with the fix → `artifact-adapter-fixed` |
| 2026-08-04 | Audio-faithful reference built; **system ranking flips** under it (`exp-2026-08-04-ranking-flips`) |
| 2026-08-10 | Corrected adapter benchmarked: beats base and predecessor, ties commercial (`exp-2026-08-10-benchmark-fixed-adapter`) |
| 2026-08-11 | Evaluation freeze (39 windows + 7 sealed); error analysis |
| 2026-08-12 | DS-WER milestone measured (not met in substance); decode ablation closes; first internal final report |
| 2026-08-13/14 | Deletion-targeted training and external-pack arms: both negative |
| 2026-08-16 | Corrected weights published to the hub (`e214de71`); **composition discovery** (0.1201 → 0.1005); first fidelity-to-audio measurement (gold set) |
| 2026-08-17 | Roster name-repair survives on the full benchmark (tiny, shadow-only) |
| 2026-08-18/19 | Training-data audits (training WER, window density, listening audits); dense-pack screen STOP |
| 2026-08-21 | Clean-pack 300-step screen: all seeds positive, gate 6 STOP; long-form decode policy P served |
| 2026-08-22 | Clean-pack 1,800-step rung: effect dissolved, step-vs-epoch flaw diagnosed; two-system fusion production spec written |
| 2026-08-22/23 | Epoch-matched 619-step three-arm run — `completed 2026-08-22; contiguous arm passed in all three seeds, combined arm did not` |

---

## 6. What is unfinished, and why

1. **Fidelity-to-audio for the new clean-pack adapter.** Section 4.7 measures it
   only against our own published text. Whether its 40% deletion reduction is real
   speech recovered, or plausible text written over silence, needs the human-verified
   gold set — which is a one-shot challenge set and was not spent on it.
2. **Fusion in production** (`exp-2026-08-21-fusion-production`, OPEN): the spec
   is frozen, nothing is measured. Two-system composition has never been
   measured anywhere; the three-system result does not transfer by assumption.
3. **Promotion of the new adapter.** It is measured, not shipped: on overall WER
   it does not clear the pre-declared bar against the incumbent (section 4.7). The
   deletion result argues for it, the WER result does not, and that call needs a
   product owner rather than another run.
4. **Fidelity-to-audio for the fusion output (W)**: blocked on an ElevenLabs
   credential in this environment; the gold set's headline question is answered
   for the candidate pool only (`exp-2026-08-16-gold-set`).
5. **Public-benchmark generalisation** — the "best free Greek model" claim is
   untested outside council speech (placeholder in section 3.2).
6. **Dataset publication** — blocked legally (DPO hold), not technically.
7. **HIR measurement** — never run (section 3.6).
8. **HParl corpus** (`exp-2026-08-14-hparl-probe`, OPEN) — deprioritised behind
   an open legal question (CLARIN licence), kept for after GSoC.

Submission logistics: the official GSoC final evaluation requires a stable
public Work Product URL, approved by the mentors before submission.
`<<TBD: final Work Product Submission URL and the exact submission window dates>>`.

---

## 7. Code and artifact links

- **Repository (work product):**
  [github.com/eellak/gsoc2026-opencouncil-stt](https://github.com/eellak/gsoc2026-opencouncil-stt)
- **Model:**
  [huggingface.co/opencouncil/whisper-large-v3-el-council-lora](https://huggingface.co/opencouncil/whisper-large-v3-el-council-lora)
  (corrected weights, hub commit `e214de71`, 2026-08-16); ledger artifacts
  `artifact-adapter-fixed` / `artifact-ct2-fixed`
- **Research ledger:** [`research/ledger.json`](../../research/ledger.json)
- **Evaluation freeze:**
  [`research/eval-freeze-2026-08/manifest.json`](../../research/eval-freeze-2026-08/manifest.json)
- **Scoring and experiment code:** `eval/` and `eval/controlled_eval/`
- **Serving:** `serve/` (long-form decode policies P and X; runbook
  [`docs/runbooks/decode-p-policy.md`](../runbooks/decode-p-policy.md))
- **Key specs:** fusion production
  ([`2026-08-21-fusion-production.md`](../specs/2026-08-21-fusion-production.md)),
  post-ASR architecture
  ([`2026-08-21-postasr-architecture.md`](../specs/2026-08-21-postasr-architecture.md)),
  clean-pack preregistrations (`docs/specs/2026-08-20/21-clean-pack-*.md`)
- **Planning issues:**
  [#3 (endgame map)](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/3),
  [#50 (post-ASR architecture)](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/50),
  [#48/#49 (evaluation-gate and budget-design flaws, filed against ourselves)](https://github.com/eellak/gsoc2026-opencouncil-stt/issues/48)
- **None.** No pull request was opened against `schemalabz/opencouncil` during the
programme. The work is a research repository plus a published model and a serving
endpoint; the integration path is documented but was not upstreamed. This is stated
plainly rather than implied: the product-side integration is *described*, not *merged*.

---

## 8. Challenges and lessons learned

1. **The expensive mistakes were measurement mistakes, not code mistakes.** The
   first training's "−32% relative" improvement evaporated under a same-stack
   A/B: it was a serving artifact. Since then: never compare across stacks,
   freeze the decode config before seeing a number, check single-item
   domination, watch the deletion rate.
2. **One silent bug can invalidate a month.** The label-prefix bug shifted every
   training target one token and cost a measured 1.58–1.77 WER points; the fix
   is most of the model's win over its predecessor.
3. **Agreement is not accuracy.** Scoring against our own published text partly
   charges systems for hearing words the reference missed
   (`exp-2026-08-17-insertion-fidelity`). Keeping the two metrics separate
   changed decisions more than once.
4. **Preregistered gates earn their cost.** Three training screens that looked
   promising (dense packs, clean packs at 300 and 1,800 steps) were stopped by
   gates frozen before any number existed. One gate turned out to be degenerate
   near zero effect — and that flaw was filed as an issue (#48) rather than
   argued around.
5. **Budget training comparisons in epochs, not steps** (#49). The same
   `MAX_STEPS` on two datasets is two different experiments.
6. **The frozen evaluation set wears out.** The 39 windows have now driven two
   screens, a stopping rule, and a budget choice; ordinary confidence intervals
   no longer describe an untouched set. Future claims must say so or use fresh
   material — a core motivation for publishing the harness (section 10).
7. **Ecosystems move under you.** Mid-project, the product adopted Scribe v2,
   which beat our adapter's reason-to-exist as a solo transcriber. The honest
   response was to re-aim at composition, where our model still adds value.
8. **Negative results are deliverables.** Seven closed doors
   (section 4.5) are what prevents the next contributor from spending GPU money
   answering questions this project already answered.

---

## 9. What comes next

1. **Fusion as the production path.** Two-system per-column composition
   (`artifact-ct2-fixed` + Soniox), no LLM, gates frozen in
   [`2026-08-21-fusion-production.md`](../specs/2026-08-21-fusion-production.md).
   Ideally run experimentally as a third "virtual" provider inside OpenCouncil's
   benchmark tool; until measured it is a plan, not a claim.
2. **Decoder-only periodic retraining** (from the scope document, experimental —
   supported by direction, not yet by a result). The reasoning: the encoder has
   little headroom on this audio; the remaining errors (names off by a
   character or two, homophone orthography) live in the decoder. The proposal:
   continuously and *algorithmically* accumulate clean corrected speech from the
   OpenCouncil corrections database — low-overlap, cleanly splittable into ~25 s
   windows like our training packs — and when a threshold is reached (e.g.
   1,000 new items), run a small scheduled decoder-only LoRA retraining and cut
   a new model version. The 619-step decoder-only arm
   (**not run — see §4.6**) is the first evidence for or against
   this. It presupposes item 3:
3. **Publish the evaluation infrastructure with public metrics** (section 10),
   so every future retraining is measured against a fixed, public yardstick
   instead of a private, wearing-out one.
4. **Second-pass review of hard spots.** Transcription remains imperfect in
   known places (overlap, rosters, numbers); the most uncertain regions should
   be routed to a second system or a human — the disagreement islands that
   fusion computes are exactly that routing signal
   ([`2026-08-21-postasr-architecture.md`](../specs/2026-08-21-postasr-architecture.md)).
5. **Aftercare.** The dataset legal question (DPO hold) and the HParl licence
   question stay open with their ledger records.

---

## 10. Publishing the evaluation harness

*Draft text for the standalone publication of the evaluation framework, per the
scope: future retraining must be measurable against published metrics.*

### What is being published

**1. The validation windows.** The frozen 2026-08 evaluation set: 39 windows of
Greek municipal-council audio from two cities never seen in training (argos,
orestiada), spanning 31 meetings and 11,911 reference tokens, frozen on
2026-08-11 with a machine-readable manifest
([`research/eval-freeze-2026-08/manifest.json`](../../research/eval-freeze-2026-08/manifest.json))
recording the selection rule, the resampling block (meeting), the normalizer
identity and the code SHA. Seven additional temporally held-out windows remain
sealed and are excluded from publication until first use.
**Not pursued.** Publishing the evaluation windows would need a data-protection review of
voice PII, the same constraint that blocks the training set. That review was not sought
before the deadline, so the harness is not published. What *is* published is the code that
builds and scores it, which is the reproducible part..

**2. The scoring method.** The exact scorer, so a number computed elsewhere is
the same number: Greek-aware normalization (NFKC, accents kept, punctuation
stripped, filler words removed under the published `wer-nofillers` rule), WER
with its substitution/deletion/insertion decomposition always reported together,
and meeting-clustered bootstrap confidence intervals. Two rules travel with the
code as documentation, because they are what kept this project honest: (a)
never compare numbers produced on different decode stacks, and (b) the
reference is OpenCouncil's published text, so the metric is
**agreement-with-OpenCouncil** — it measures product compatibility, not
fidelity to the audio, and the one measured fidelity-to-audio pass showed the
two can rank systems differently.

**3. The test benchmark.** The OpenCouncil benchmark windows and API
(`cap-bench-http-api`), with each system — including any future adapter version
— registered as its own named provider. Published alongside:
the current baseline table (section 4.1 and 4.2 of this report), so any future
retraining has a fixed public number to beat, and the gate ladder of
[`docs/decisions/training-evidence.md`](../decisions/training-evidence.md)
(paired seeds, deletion/insertion guards, domination check with its known
near-zero degeneracy documented per issue #48).

### Why

The frozen private set has been consumed: it has driven stopping decisions and
a budget choice, so its confidence intervals no longer describe untouched
material. The decoder-only periodic-retraining loop of section 9 only works if
"the new version is better" is checkable by anyone against a published
yardstick. Publishing the windows, the scorer and the benchmark turns model
maintenance from a private judgement into a reproducible measurement.

**Not published**, for the reason above. The scoring code, the frozen decode config, and
the window-selection procedure all live in this repository and are the reusable artefacts.

---

## What is still open

Not placeholders — decisions, with reasons.

**Blocked on the human, and the only mechanically fatal one:** the Work Product
Submission URL and the exact submission window. A missing or unreachable URL fails the
evaluation regardless of the work behind it.

**Attempted and withdrawn:** the public Greek benchmark (FLEURS), for the key-collision
bug described in §3.2. Corrected scripts exist; the rerun is open.

**Deliberately not run, given the deadline:**

- the decoder-only LoRA arm — it was a fallback, and the fallback was not needed
- training-set WER on the new pack — a diagnostic, not a claim
- the combined arm at a matched *epoch* budget (~1,746 steps), which is what would be
  needed to say whether jitter-splicing helps or hurts rather than merely "did not help
  at equal optimizer updates"

**Ruled out of scope by decision, not by time:** the data-protection review needed to
publish the evaluation windows, publication of the harness itself, and upstream pull
requests.

**Running at the time of writing:** the post-june held-out benchmark, both adapters. Its
state is reported in §4.7 exactly as it stands, complete or not.
