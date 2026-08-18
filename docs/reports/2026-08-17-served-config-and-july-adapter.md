# We had never scored what we ship, and the July adapter question was already answered

2026-08-17. `exp-2026-08-17-served-config-and-july-adapter`. Zero GPU, zero paid API,
one machine, one decoder, one scorer. Harness:
[`notebooks/served_config_and_july.py`](../../notebooks/served_config_and_july.py),
tests [`eval/tests/test_served_config.py`](../../eval/tests/test_served_config.py),
results `$SC/served-config-2026-08/results.json`.

Two questions and one audit:

- **A.** Every decode conclusion in this project was produced at `beam_size=5,
  word_timestamps=False`. The deployment is not that. What does the served
  configuration score?
- **B.** The user believes the late-July adapter may be better than the corrected
  one. Provenance answers that before any compute; the number is run anyway.
- **C.** The conversion ladder that was supposed to explain the 2026-07-29 decoder
  asymmetry. Audited rather than run, and the audit is the finding.

<!-- NUMBERS -->

## What was frozen, and when

Everything below was fixed in the harness docstring and in the module constants
**before any hypothesis was decoded and before any WER was computed**, and none of it
was changed afterwards:

- the arm table and the rule that each arm equals `decode_ablation.CONTROL` with its
  declared keys changed and nothing else (checked against faster-whisper's *resolved*
  options, not against the request);
- the text representation (per-segment tokenization);
- the scorer (`eval_freeze.ftoks` + the frozen filler regex, `exp_same_stack.sdi`,
  micro-WER over the pooled 11,911 reference tokens);
- the uncertainty method (paired meeting-clustered bootstrap, 31 blocks, 4,000
  replicates, seed 7) and the secondary 32-block `(city, meeting_id)` split;
- the endpoint hierarchy: **WER primary, deletion rate as a safety endpoint,
  substitutions and insertions descriptive**;
- the interaction estimand `(S2 − S1) − (RW − R)`;
- the domination reporting (signed share, gross share, leave-one-out, separately);
- and the July guard sentence, which is emitted into the results JSON so it cannot be
  lost between the data and the prose.

A Codex review of the design ran before any decode started and changed it in three
load-bearing ways: it caught the segment-join bias described below, it rejected
reusing the cached 2026-08-12 control as a causal baseline, and it refused to let this
be called a production emulation at 16 threads. A second Codex review ran on the
findings.

## The substrate

The 39 frozen validation windows of
[`research/eval-freeze-2026-08/manifest.json`](../../research/eval-freeze-2026-08/manifest.json):
31 meetings, 11,911 reference tokens, argos + orestiada, meetings before 2026-06-01.
The 7 sealed temporal-holdout windows and the 16 locked evaluation windows were not
touched; the harness asserts the empty intersection before it decodes anything.

Reference is the cached benchmark report's `referenceText` — that is
*agreement-with-OpenCouncil*, not fidelity-to-audio. Nothing here says which
configuration is more faithful to what was said. It says which one agrees more with
our own published text.

## The measurement that had to be fixed first

`notebooks/decode_ablation.py` stores a window's hypothesis as
`"".join(segment.text)`. faster-whisper does not always put a leading space on a
segment — `exp-2026-08-16-adapter-confidence` counted **505 of 1,677 boundaries with
none** — so that join fuses the last word of one segment into the first word of the
next. And `word_timestamps` *moves segment boundaries*, which is one of the two
things under test here. Scoring the fused string would have charged the served
configuration for fusions the join invented, at exactly the boundaries being tested.

Production does not do that. `serve/oc-asr/oc_asr_server.py` emits `seg.text.strip()`
per utterance and `" ".join(...)` for the full transcript. So the primary hypothesis
here is built per segment, which is token-identical to the server's own join:

```python
hyp = [tok for seg in segments for tok in ftoks(seg)]
```

Two consequences. Segment texts are now stored, so any later question can be
re-scored without re-decoding. And the two arms whose decodes predate this experiment
— the cached 2026-08-12 control and the cached correction-only arm — stored only the
fused string, cannot be repaired retrospectively, and are therefore reported as
**exploratory** beside the primary table rather than inside it.

That is also why the control was re-decoded today rather than lifted from
`$SC/decode-ablation/eval-A.json`. There is a second reason and it is independent: the
paired bootstrap resamples meetings while holding the hypotheses fixed, so it contains
no run-to-run decoder variance at all, and `exp-2026-08-16-adapter-confidence`
**withdrew** the bit-exactness claim for this decode (16 of 18 windows reproduced, not
18 of 18). A five-day-old baseline cannot carry a causal attribution made today.

## What this is not

This is a **decode-option ablation on the frozen evaluation stack** — cpu / int8 / 16
threads, the stack every other number on these 39 windows came from. Three production
behaviours are absent, and none of them is papered over:

1. `OC_ASR_CPU_THREADS=8`. Held at 16 so these contrasts stay comparable with every
   existing number on this substrate. Bounded separately by the thread probe below.
2. `OC_ASR_MAX_INFER_SEC=150`, a streaming guard that stops consuming segments and
   appends a truncation marker. Not emulated. The wall-clock column reports how many
   windows ran past 150 s, which shows the guard is *reachable*; it is not a
   counterfactual truncation result, and the real guard is worse than that column
   suggests because the server starts its timer before it acquires the inference lock,
   so queueing counts against it.
3. Request queueing and the two routes' response assembly beyond the text join.

## B, first: the provenance answer, which comes before any compute

The July adapter is `artifact-adapter-july-broken`. Its ledger record says
`KNOWN_BROKEN`, and it says why: it was trained with the label-prefix bug, targets
shifted one position (`exp-2026-07-31-label-prefix-bug`).
[`CURRENT.md`](../../CURRENT.md) states the same thing as project policy —
everything trained before 2026-08-01 "cannot answer" the project's question.

Both artifacts were verified by content hash on disk today before anything was
decoded, against the ledger's recorded values:

| artifact | path | `model.bin` sha256[:16] | ledger |
|---|---|---|---|
| `artifact-ct2-fixed` | `~/oc-asr-serve/ct2-fixed` | `8a1a3b257d0c1bdb` | matches |
| `artifact-ct2-july-broken` | `~/oc-asr-serve/ct2` | `dfffde80906f2cd5` | matches |
| `artifact-adapter-july-broken` | `~/oc-train-checkpoints/adapter` | `ff3ba0b334551ed3` (`adapter_model.safetensors`) | matches |

So the July adapter **is** on disk, it **is** identifiable, and it already has a CT2
build that runs on the same stack — which is why the arm could be run at all. The
harness re-hashes the model before every decode and refuses to write a hypothesis
whose producing model it cannot name.

There is also a prior measurement, and the version of it that applies here is not the
headline one. `exp-2026-08-10-benchmark-fixed-adapter` reports −1.77 points for
corrected-minus-July on the 40-window benchmark slice; re-measured on **these** 39
clean windows the same report gives **−1.58 [−2.41, −0.85]** at 90%. That number came
off the benchmark's own server-side `wer-nofillers` on a **CUDA** pod, so it is not
comparable to anything below — different stack, different normalizer. It is a prior,
not a baseline.

### The guard that travels with the number

> `artifact-ct2-july-broken` remains `KNOWN_BROKEN` regardless of this contrast's
> sign. This is a forensic comparison of two fixed historical binaries on one CPU
> decode realization of a repeatedly-scored 39-window
> agreement-with-OpenCouncil slice. It is not evidence that the July training targets,
> adapter or deployment are usable. A July win would say only that this broken binary
> scored better here; a null would say this design did not resolve a difference, not
> that the two are equivalent. The contrast does not isolate the causal effect of
> fixing the label prefix, because training-seed variation is unreplicated and was
> measured at 2.1 WER points.

## C. The conversion ladder cannot answer its question as it stands

[`2026-08-09-longform-preflight.md`](2026-08-09-longform-preflight.md) names a cheap
test to separate "the training format did this" from "the deployment pipeline did
this": HF base → HF base + live LoRA → HF merged → CT2, over fixed 30 s chunks. It
records that the test was never run at scale ("Δεν έγινε"), and its own erratum notes
that the 4-chunk version it *did* run gave a result it then refused to generalise.

`eval/controlled_eval/conversion_ladder.py` implements it. **It was audited before it
was run, and it should not have been run as written.** Its three model defaults named
three different lineages:

| rung | default | built | lineage |
|---|---|---|---|
| 2, live LoRA | `~/oc-asr-serve/adapter-fixed-2026-08-01` | 2026-08-01 | **corrected** |
| 3, HF merged | `~/oc-asr-serve/merged` | 2026-07-23 | **July, label-prefix bug** |
| 4, CT2 int8 | `~/oc-asr-serve/ct2` | 2026-07-23 | **July, label-prefix bug** |

A gap appearing at rung 3 would have read as *a merge bug* when it was a different
adapter — precisely the confound the ladder exists to rule out, and the same class of
error as the 2026-08-10 provenance erratum on the preflight itself. The defaults are
now corrected to `merged-fixed` and `ct2-fixed`, and the mismatch is recorded in the
script.

Three further limits are documented in the script and **not** fixed, because fixing
them is a redesign, not a repair:

- there is no **CT2 fp16** rung, so rung 4 moves runtime, decoder and quantization in
  one step; a gap there cannot be attributed to int8 rather than to CTranslate2;
- `pairs` is not one quantity — rungs 1–3 count `<|t|>` marks ÷ 2 in the decoded
  string, rung 4 counts faster-whisper `Segment` objects;
- the HF rungs are plain greedy `generate` while the CT2 rung inherits
  faster-whisper's stochastic temperature-fallback ladder, so rung 4 is not a greedy
  control.

And the premise needs correcting too. The asymmetry that motivates all of this —
fine-tune 22.66 under faster-whisper against 28.18 under HF generate, base 27.08
against 28.25 — is from
[`2026-07-29-finetune-gain-decomposition.md`](2026-07-29-finetune-gain-decomposition.md),
where **"ours" is `artifact-adapter-july-broken`**, on n=300 isolated utterances from
two cities, scored with the training script's `gnorm` rather than the frozen
evaluation normalizer. It is an asymmetry measured on a model we know was trained on
shifted targets. Whether it exists for `artifact-adapter-fixed` at all has never been
checked.

**Not run, and costed.** A corrected ladder on this CPU is roughly 40 chunks × 4 rungs,
with the three HF rungs at fp32 dominating — on the order of 6–10 hours, plus a CT2
fp16 conversion (~10 min, ~3 GB, against 26 GB free) for the fifth rung that makes
rung 4 interpretable. It did not fit beside Parts A and B and it is not worth
funding until someone re-measures the asymmetry itself on the corrected adapter, which
is the cheap first step: the n=300 decomposition re-run with `artifact-adapter-fixed`
under both harnesses, ~2 h CPU. If the asymmetry is gone, the ladder is moot.

---

## The chain

- [Decode thresholds: no arm ships](2026-08-12-decode-ablation.md)
- [The corrected adapter against the July one, on the benchmark](2026-08-10-benchmark-corrected-adapter.md)
- [Word timestamps change the decode](2026-08-16-adapter-confidence.md)
- [What the 39-window harness can and cannot see](2026-08-16-harness-coverage-mde.md)
- [The untried inventory](2026-08-17-untried-inventory.md) (item A4)
