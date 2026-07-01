# Logs

Dated implementation/vault history. Kept intentionally sparse — see cadence rules below.

## Cadence

- **Weekly digest** on Fridays (`YYYY-MM-DD-weekly.md`). One file per week.
- Skip the week entirely if none of these happened:
  - a canonical doc was updated (`CURRENT.md`, `docs/decisions/**`, `docs/roadmap.md`, `docs/specs/**`);
  - a decision was added, changed, or superseded;
  - an implementation milestone landed;
  - a real ambiguity surfaced that needs a future decision.
- **Ad-hoc log** allowed only for a significant incident or one-off decision that does not fit a weekly digest. Name it descriptively (`YYYY-MM-DD-<slug>.md`).

Routine "I reviewed the diff and nothing structural changed" does not deserve a log entry.

## Index

- [2026-05-12 - Vault setup](2026-05-12-vault-setup.md): consolidated record of vault organization, agent instructions, meeting-notes skill, Mermaid diagrams, and PRD todo notation.
- [2026-05-18 - Weekly digest](2026-05-18-weekly.md): full CSV ingest restored, audio CORS/decoding workaround in place, decisions split into themed files.
- [2026-07-01 - Auto-selected batch-2](2026-07-01-next-batch-selection.md): end-to-end automated selection of 7,364 fine-tune edits (candidate pool → Soniox faithfulness calibration → interestingness ranking → Sonnet triage); faithfulness metric validated; Soniox gold pass pending human go.

Older daily-normalization entries (May 13–18) are archived under [`archive/logs/2026-W20-daily-normalizations/`](../../archive/logs/2026-W20-daily-normalizations/) as historical noise.
