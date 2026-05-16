# 2026-05-15 - Daily Normalization

## Summary

Reviewed files changed in the last 24 hours after the May 14 normalization pass. The touched files already preserve the current exploration-first direction and correctly mark full CSV ingest as partial/restoration-needed.

## Changes

- Added this daily maintenance log.
- Linked this maintenance log from `docs/project-map.md`.

## Left Unchanged

- `CURRENT.md` still points to the next practical work: restore or replace full CSV ingest, cache meeting JSON, and build correction-to-utterance matching.
- `docs/roadmap.md` already keeps Phase 1 and Phase 2 as partially in progress with the right todo states.
- `docs/specs/local-data-model.md` already separates implemented local label state from planned meeting cache and matched correction records.
- `ui/README.md` already describes the active local-first SvelteKit prototype and its dummy fixture constraint.
- `docs/decisions.md`, meeting notes, and reference notes did not need edits.

## Ambiguous / Human Review

- `archive/ui.archive-2026-05-13/` still contains generated dependency/build material. It remains archived rather than deleted, but may be too heavy to keep long-term.
- The active `ui/scripts/` folder still lacks a full CSV ingest script. The archived `archive/ui.archive-2026-05-13/scripts/ingest.ts` should be reviewed before restoring or replacing it.
