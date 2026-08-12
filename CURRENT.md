# Current Work

Last updated: 2026-08-12

**The endgame plan is finished.** All four workstreams of
[`docs/specs/2026-08-11-endgame-handoff-plan.md`](docs/specs/2026-08-11-endgame-handoff-plan.md)
are closed, 2026-08-12. Read
[`docs/reports/2026-08-20-final-report.md`](docs/reports/2026-08-20-final-report.md)
first — it is the answer to the project's question, with its limits.

What is left is the queue below: one decision (publish), one experiment (name
lexicon), one gap (fidelity-to-audio).

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
   independent reviewers ranked it the highest-value remaining direction, and this
   cycle **priced it**: our domain-term errors are 90 substitutions to 28 deletions
   (`exp-2026-08-12-ds-wer`), i.e. the name is written, just wrong by a character or
   two. That is what a lexicon repairs. It is also the only cheap lever left — the
   other three are closed below. **Done when** name recall is measured on a held-out
   set with the roster wired into the endpoint.
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

## Three closed doors — do not reopen without a reason

Each cost real time this cycle and each is answered in the ledger:

- **Decode thresholds** (`exp-2026-08-12-decode-ablation`). The no-speech gate fires
  **zero** times on the 39 windows, so it cannot be causing our deletions; removing
  the temperature fallback makes every metric worse.
- **Label purity** (`exp-2026-08-13-correction-only`). Dropping all `no_edit` rows
  moves WER by +0.0015, against a known 2.1-point per-seed spread.
- **Data scale** (`exp-2026-08-11-wer-levers-research`). The dominant residual error
  is homophone orthography the audio cannot decide; ~1300h buys ~0.5 points.

The 2026-08 evaluation freeze
([`manifest.json`](research/eval-freeze-2026-08/manifest.json)) is the substrate for
all three: 39 validation windows, 31 meetings, 11,911 reference tokens. Its 7
temporal holdout windows are **still sealed** — no arm ever passed a gate that would
have released them.

## Recently changed

- `exp-2026-08-20-final-report` closed: the answer is yes-but-modestly, and what
  remains points at names rather than at audio, 2026-08-12.
- `exp-2026-08-13-correction-only` closed: dropping the unverified half buys nothing,
  labelled suggestive; `artifact-adapter-correction-only` registered, 2026-08-12.
- `exp-2026-08-12-decode-ablation` closed: no threshold ships; six arms collapse into
  two behaviours, 2026-08-12.
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
