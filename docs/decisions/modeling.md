# Modeling and serving decisions

Decisions about what model/technique we actually run, and why. Keep entries short;
reasoning lives in the reports and handoffs linked from each entry.

## 2026-08-08 — Never compare two models across two decoding stacks

**Decision:** any claim of the form "model X does Y more than model Z" is invalid unless
both were decoded by the **same** engine with the same settings. Benchmark rows from
different providers do not qualify, even when the audio and the reference are identical.

**Evidence:** the fine-tune appeared to delete +1.54 points more than base. Re-decoding
both with one engine gives **+0.19, 90% [−0.50, +0.85]** — the original number sits
outside the interval. The same base weights produce deletion rates from **3.27 to 23.87**
depending on the decoder, so the between-stack spread is ten times the difference being
interpreted. [Report](../reports/2026-08-08-same-stack.md).

**What survives:** the fine-tune's real, measurable gain is **substitutions, −0.61 points,
CI excludes zero**. It puts the right word in more often. It does not delete more.

**Consequence:** the proposal to retrain on continuous 30-second windows loses its
motivating evidence. The mechanism stays plausible and would need its own criterion.

## 2026-08-08 — The corrections-to-clean-speech ratio is not the lever

**Decision:** stop spending GPU hours on the mixture ratio. 20/80 and 50/50 are
indistinguishable on the audio-faithful reference, so pick the ratio on **cost**, and 20%
corrections is cheaper in human review hours.

**Evidence:** seven preregistered runs, three seed pairs, $24.
C − A = **−0.24 points, 90% [−0.89, +0.36]**; the deletion guardrail passes at +0.08
against a 0.5 limit. Gate 1 was not met, so this is inconclusive by the frozen rule, not
by interpretation. Per-seed differences span **2.1 points** — an order of magnitude above
the mean effect, which is the number worth remembering.
[Report](../reports/2026-08-08-mixture-ratio.md) ·
[preregistration](../specs/mixture-ratio-preregistration.md).

**Caveat kept on the record:** all seven arms share the single-utterance clip shape. If
window shape dominates, this measured the ratio inside a regime that hides it.

## 2026-07-25 — Contextual biasing (roster hotwords) is the primary name-accuracy lever

**Decision:** name accuracy is addressed at **inference time** with per-meeting context
(speaker roster as faster-whisper `hotwords`), not by acoustic fine-tuning.

**Evidence:** on a name-focused held-out subset (n=59, 114 gold names, same-stack A/B),
base whisper + roster hotwords lifts name recall 27.2% → 36.0% (McNemar exact p = 0.021)
with no WER cost (bootstrap CI on the WER delta straddles zero). The LoRA fine-tune alone
is not significantly better than base (WER −0.008, CI straddles zero). See
[report](../reports/2026-07-25-hotwords-biasing.md) and the
[postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md).

**Why it holds up:** the roster is known *before* transcription — OpenCouncil already
fetches it from the meeting endpoint. Biasing uses information the generic model cannot
have, which is exactly the kind of gain that survives a fair baseline.

**Deployment shape (when it ships):**

- Caller passes `hotwords` per meeting: roster full names first, then single tokens,
  truncated to ~180 tokens (whisper's prompt window is 224; leave headroom).
- Roster source in this repo: `data/pii/rosters_full.json` (311 meetings). In production:
  `GET /api/cities/{city}/meetings/{meeting}` → `people[].name`, `parties[].name`.
- Nothing about the served model changes. This is a change to how the ASR is *called*.
- Reproduce: `eval/controlled_eval/ab_hotwords_names.py`.

**Not decided:** whether the fine-tune stays in the stack at all. `ours + hotwords` was the
best config on the name subset (WER 0.3072 vs base 0.3412, CI excludes zero) but the
adapter still regresses on the corrected-utterance subset. Resolve with the harness before
migrating anything.

## 2026-07-25 — The LoRA adapter is not deployed to OpenCouncil

**Decision:** `opencouncil/whisper-large-v3-el-council-lora` stays published but unused in
production. Under a same-stack A/B it ties base on general utterances and regresses on the
utterances it was trained to fix (0.176 vs 0.158).

**Why:** the earlier wins were serving-skew artifacts (baselines ran on the HF serverless
pipeline, our model on a tuned faster-whisper stack). Full reasoning in the
[postmortem](../handoff/2026-07-25-finetune-eval-postmortem.md).

**Standing rule:** no model is recommended for production without a same-stack,
same-normalization A/B against base on all three held-out slices (general, name-focused,
corrected).
