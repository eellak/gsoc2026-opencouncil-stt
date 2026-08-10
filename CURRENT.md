# Current Work

Last updated: 2026-08-10

Canonical research state: [`research/ledger.json`](research/ledger.json).
Agent protocol: [`CLAUDE.md`](CLAUDE.md).

The long narrative that used to live here is at
[`archive/current/2026-08-10-CURRENT.md`](archive/current/2026-08-10-CURRENT.md).
It is history, not state.

## Objective

Deliver a defensible answer to one question for GSoC: does domain fine-tuning of
whisper-large-v3 improve Greek council transcription enough to matter? The corrected
adapter (`artifact-adapter-fixed`) is the candidate. Everything before 2026-08-01 was
trained through the label-prefix bug and cannot answer it.

## Work queue

1. Publish `artifact-adapter-fixed` to HuggingFace. The benchmark now prices the
   delay: the published weights cost **1.77 WER points on unseen cities**. **Done
   when** the hub weights are the corrected ones.
2. `exp-2026-07-25-hotwords` — ship contextual biasing at serving time. Both
   independent reviewers ranked it the highest-value remaining direction: it targets
   names directly without retraining. **Done when** name recall is measured on a
   held-out set with the roster wired into the endpoint.
3. Decide whether fidelity-to-audio changes this ranking. The benchmark measures
   agreement-with-OpenCouncil; the one time both were measured, the ranking flipped.
   **Done when** the corrected adapter has a fidelity-to-audio number on unseen
   meetings.

## Blockers

- Item 1 is blocked on nothing technical — it is a decision. The corrected weights
  exist locally and are now measured.
- The dataset itself remains on **legal hold** (DPO, 2026-07-17): text-level PII
  removal does not anonymise it because each row links audio carrying the voice. See
  [decisions/data.md](docs/decisions/data.md).

## Recently changed

- `exp-2026-08-10-benchmark-fixed-adapter` closed: on unseen cities the corrected
  adapter is indistinguishable from Scribe and Soniox and beats its broken
  predecessor by 1.77 points, CI excludes zero, 2026-08-10.
- `exp-2026-08-10-packed-training` closed STOP; four overstated claims corrected
  after a Codex audit, 2026-08-10.
- `exp-2026-08-08-same-stack` gained a provenance erratum: its fine-tune arm was the
  broken adapter, 2026-08-10.
- `exp-2026-08-08-mixture-ratio` closed by user instruction, 2026-08-09.
