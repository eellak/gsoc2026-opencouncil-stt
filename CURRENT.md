# Current Work

Last updated: 2026-08-12

**Active plan:** [`docs/specs/2026-08-11-endgame-handoff-plan.md`](docs/specs/2026-08-11-endgame-handoff-plan.md)
— Codex-reviewed handoff plan for the final 12 days: decode-threshold ablation,
DS-WER (Milestone 2), correction-only dataset ablation, final report. It reorders
the queue below; queue item 2 (hotwords/name repair) keeps its own plan and gate.

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

## Product decision: answered 2026-08-12 — transcripts are **clean**

Filled pauses («εεε») and false starts are **stripped**. Asked and answered once, per
the handoff plan. This is the standard future listening hours should be produced
against, so that they are compatible with each other.

Nothing frozen was changed on the strength of it: the benchmark normalizer, the
existing labels and the 2026-08 evaluation freeze all stay exactly as they were. The
decision governs new data collection, not any number already measured.

## Blockers

- Item 1 is blocked on nothing technical — it is a decision. The corrected weights
  exist locally and are now measured.
- The dataset itself remains on **legal hold** (DPO, 2026-07-17): text-level PII
  removal does not anonymise it because each row links audio carrying the voice. See
  [decisions/data.md](docs/decisions/data.md).

## Endgame plan — where the four workstreams stand (2026-08-12)

- **Task 0 — evaluation freeze: done.**
  [`research/eval-freeze-2026-08/manifest.json`](research/eval-freeze-2026-08/manifest.json)
  fixes 39 validation windows (31 meetings, 11,911 reference tokens) and 7 temporal
  holdout windows. The plan's "minus the 7" was wrong; the rule catches one window in
  argos/orestiada, so it is 40 − 1.
- **Workstream 2 — DS-WER: closed.** `exp-2026-08-12-ds-wer`. Milestone 2 met on the
  point estimate (+17.0% vs Gladia), interval includes zero.
- **Workstream 1 — decode ablation: running.** `exp-2026-08-12-decode-ablation`,
  39 windows × 6 arms on CPU. First pass discarded and re-run after the seed was
  moved from `(arm, window)` to `window` — see the prereg.
- **Workstream 3 — correction-only: training.** `exp-2026-08-13-correction-only`,
  pod `ul4z0drp5owiac` (A40), hard deadline `2026-08-12T11:53:25+03:00`.
- **Workstream 4 — final report: not started**, waits on 1 and 3.

## Recently changed

- `exp-2026-08-12-ds-wer` closed: on domain terms we sit at 0.488 against Soniox
  0.328 and Scribe 0.372, far worse than our tie with them on overall WER, and our
  name errors are substitutions not deletions, 2026-08-12.
- `exp-2026-08-10-benchmark-fixed-adapter` closed: on unseen cities the corrected
  adapter is indistinguishable from Scribe and Soniox and beats its broken
  predecessor by 1.77 points, CI excludes zero, 2026-08-10.
- `exp-2026-08-10-packed-training` closed STOP; four overstated claims corrected
  after a Codex audit, 2026-08-10.
- `exp-2026-08-08-same-stack` gained a provenance erratum: its fine-tune arm was the
  broken adapter, 2026-08-10.
- `exp-2026-08-08-mixture-ratio` closed by user instruction, 2026-08-09.
