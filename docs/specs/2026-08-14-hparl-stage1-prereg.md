# Preregistration — HParl as a stage-1 adaptation corpus

2026-08-14. Status: **draft, not started.** No GPU may be created against this spec
until the open items in §7 are closed and the user approves the spend.

Ledger record: `exp-2026-08-14-hparl-probe` (the data work). The training experiment
gets its own OPEN record when this spec is approved.

## 1. Question

Does two-stage adaptation — first on filtered, punctuation-restored HParl
(out-of-domain parliamentary speech), then fine-tuning on our in-domain corrections —
beat fine-tuning on our corrections alone, on the frozen 2026-08 evaluation set?

Two-stage, **not** a mixture: `exp-2026-08-08-mixture-ratio` found the ratio is not a
lever, and the 2026-08-11 review found sequential adaptation wins when the domains
differ sharply. These domains differ sharply.

## 2. What the prior evidence says to expect

State this before running, so the result cannot be re-narrated afterwards:

- **The expected effect is small or zero.** `exp-2026-08-11-wer-levers-research`:
  ~1300h buys ~0.5 WER points, and the dominant residual error is homophone
  orthography the audio cannot decide. The Swiss German precedent added 582 hours for
  a 0.04-point change.
- **The plausible gain is robustness, not peak in-domain accuracy.**
- **There is a specific way this can make things worse.** 74% of HParl segments are
  sentence fragments (measured, §3). Training on fragments can teach truncation — and
  truncation is exactly the failure this project is fighting
  (`exp-2026-08-13-targeted-deletion-training`; the standing finding is that the
  deletions live in the weights). A deletion-rate regression is the primary risk, not
  a side concern.

**Therefore the gate below is two-sided: this experiment can lose.**

## 3. Data, and how it was built

Source: `Elormiden/Hellenic-greek-parliamentary-speech` (CLARIN HParl, ML-processed),
92,133 clips, ~120 h.

Pipeline, all measured on the pilot (see the report):

1. `eval/hparl2_filter.py` — MP3 at 32 kbps mono (4.0 kB/s), Soniox transcription,
   `<...>` tags stripped, alignment = 1 − WER on `greek_normalize` tokens.
   **Admission gate: align ≥ 0.95.** On the first 150-row sample this kept 49% of
   rows / 50% of audio.
2. `eval/hparl2_punctuate.py` — label = dataset words + Soniox punctuation, with
   `gpt-5.6-luna` deciding sentence completeness. **Hard word guard**: a row is only
   accepted if the model's `greek_normalize` token sequence is identical to the
   dataset transcript; otherwise it falls back to the deterministic transfer. On the
   74 pilot rows: 74/74 passed the guard, and **only 26% of segments are complete
   sentences**.

### Gate decision — preregistered 2026-08-14, before any WER exists

The `align ≥ 0.95` admission rule was my choice and was never validated. Two rounds of
human judgement (43 non-blind, then 62 blind with hidden controls at both extremes)
say it is wrong: it throws away 50 of the 56 rows a human accepted, and rows scoring
below 0.40 are mostly cases where **Soniox failed on a 1–2 s clip**, not cases where
the label is wrong. Full evidence in the report, §"Blind round".

**The gate for the training pack is therefore, fixed from now:**

```
align ≥ 0.50  AND  duration ≥ 1.5 s  AND  ≥ 3 tokens
edge-clipped rows are ADMITTED, carrying edge_clipped=true and weight=0.5
```

- `0.50` — admits 53 of the 62 blind-judged rows with **zero** human rejections, and
  misses only 5 rows the human accepted.
- `1.5 s` — both human rejections were at align 0.00 *and* under 1.5 s. The failure
  mode is short clips, not the score.
- **edge flags weight, not gate** — all 4 "don't know" answers ("it said the text but
  the start was cut") carry a missing first or last reference token, so the flag has
  the recall to find the defect; but it also fires on 28 rows the human approved, so
  it lacks the precision to reject on.

Both packs are built and kept: `hparl2-v1` at the old 0.95 (6.0 h) and `hparl2-v2` at
this gate (~13 h on the pilot). **v2 is the arm; v1 exists so the gate itself can be
tested if v2's result is ambiguous.** Neither is re-tuned after a number is seen —
that is the entire point of writing this down now.

Recorded caveats that must travel with any number this experiment produces:

- The 0.95 gate is effectively an exact-match test on a median 9-token row. It selects
  **easy audio** and yields a **Soniox-agreement sample**, not a random one.
- `<spoken_noise>` is not the discriminator (keep rate 48% tagged vs 52% untagged).
- The corpus is out of domain: parliamentary chamber, not municipal council.

## 4. Design

| | arm A (control) | arm B (treatment) |
|---|---|---|
| stage 1 | — | LoRA adaptation on filtered HParl |
| stage 2 | fine-tune on our corrections | same fine-tune, same data, same steps |

- **3 seeds per arm.** Non-negotiable: the measured per-seed spread is 2.1 WER points,
  so a single-seed result is a screen, never a conclusion.
- Matched updates, LR schedule and token exposure in stage 2 across arms. The only
  difference is the stage-1 initialisation.
- Base model and stage-2 data identical to the frozen recipe in
  `exp-2026-08-13-targeted-deletion-training`.

## 5. Frozen before any number is seen

- **Decode config is frozen now** and copied verbatim from the corrected-adapter
  benchmark runs. No beam-size selection after the fact.
- **Evaluation substrate**: the 2026-08 freeze,
  `research/eval-freeze-2026-08/manifest.json` — 39 validation windows, 31 meetings,
  11,911 reference tokens.
- **The 7 temporal holdout windows stay sealed.** Nothing in this spec releases them.
- Normalization: the existing benchmark normalizer, unchanged.

## 6. Metrics and the decision gate

Primary metric is **agreement-with-OpenCouncil WER** on the 39 validation windows —
and it is labelled as such. It records product compatibility. Any fidelity-to-audio
claim requires the separate measurement (queue item 3) and is out of scope here.

Gate, evaluated on the seed means:

| condition | threshold |
|---|---|
| WER (arm B − arm A) | **< 0** and the CI excludes zero |
| deletion rate (B − A) | **≤ 0** — a regression here fails the experiment outright |
| insertion rate (B − A) | < +0.0005 |
| substitution rate (B − A) | < +0.0005 |

Also reported, not gating: DS-WER on domain terms (`scripts/ds_wer.py`), name recall,
and a single-item domination check — no headline delta is quoted before confirming
that no single window supplies most of it.

**If the gate fails, the record closes CLOSED with a negative result.** Negative and
inconclusive-by-gate results are answers; they are not permission to rerun with a
different mix.

## 7. Open before this can start

- [ ] Pilot filter pass finished; keep rate confirmed on ≥ 10k rows across all shards,
      not 150 rows of one shard.
- [ ] Human spot-check of the punctuation output on the phase-2 review page, and of
      the 0.95 gate itself. Reported so far verbally: punctuation is good, with
      occasional comma-where-a-period-belongs. Prompt rule 3 was added for that; it
      has **not** been re-measured.
- [ ] **Licence resolved at source.** The HF mirror's own card conflicts with itself:
      YAML `cc-by-4.0`, README body CC BY-**NC** 4.0. CLARIN 1602 is the authority. If
      NC holds and OpenCouncil has any commercial dimension, this experiment does not
      run.
- [ ] Decide how many hours actually get used. 60 h of easy, out-of-domain, fragmentary
      audio is not obviously better than 20 h of it.
- [ ] Pod plan: hard-deadline watchdog armed **before** upload, pod ID recorded, per
      `docs/runbooks/runpod-training-pod.md`. A pod bills from creation.

## 8. Stop conditions

- Any arm failing twice the same way → stop, kill the pod, write down what broke.
- Stage-1 training loss diverging or the stage-1 checkpoint producing empty
  transcriptions on a smoke set → stop before stage 2.
