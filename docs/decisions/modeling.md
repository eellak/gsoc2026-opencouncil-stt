# Modeling and serving decisions

Decisions about what model/technique we actually run, and why. Keep entries short;
reasoning lives in the reports and handoffs linked from each entry.

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
