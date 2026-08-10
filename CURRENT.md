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

1. `exp-2026-08-10-benchmark-fixed-adapter` — measure the corrected adapter on the
   260-window OpenCouncil benchmark, served from a GPU pod, cloned against the July
   run so all other providers come free. **Done when** `report.json` exists and the
   argos+orestiada held-out slice is reported separately from the contaminated pool,
   with single-window domination checked.
2. `exp-2026-07-25-hotwords` — ship contextual biasing at serving time. Both
   independent reviewers ranked it the highest-value remaining direction: it targets
   names directly without retraining. **Done when** name recall is measured on a
   held-out set with the roster wired into the endpoint.
3. Publish `artifact-adapter-fixed` to HuggingFace. **Done when** the hub weights are
   the corrected ones. The public model card has promised this since 2026-08-01 and
   still serves the broken adapter.

## Blockers

- Item 3 is blocked on nothing technical — it is a decision, not a task. The
  corrected weights exist locally.
- The dataset itself remains on **legal hold** (DPO, 2026-07-17): text-level PII
  removal does not anonymise it because each row links audio carrying the voice. See
  [decisions/data.md](docs/decisions/data.md).

## Recently changed

- `exp-2026-08-10-benchmark-fixed-adapter` opened, 2026-08-10.
- `exp-2026-08-10-packed-training` closed STOP; four overstated claims corrected
  after a Codex audit, 2026-08-10.
- `exp-2026-08-08-same-stack` gained a provenance erratum: its fine-tune arm was the
  broken adapter, 2026-08-10.
- `exp-2026-08-08-mixture-ratio` closed by user instruction, 2026-08-09.
