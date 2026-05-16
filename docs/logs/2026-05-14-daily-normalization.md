# 2026-05-14 - Daily Normalization

## Summary

Reviewed files changed in the last 24 hours after additional UI prototype work and a root-level superseded UI archive appeared.

## Changes

- Restored `ui/README.md` from default Svelte scaffold text to project-specific prototype instructions.
- Moved the superseded `ui.archive-2026-05-13/` folder under `archive/` to preserve the normalized vault structure.
- Updated `CURRENT.md`, `docs/roadmap.md`, and `docs/specs/local-data-model.md` so they no longer claim an active full CSV ingest script exists under `ui/scripts/`.
- Linked this maintenance log from `docs/project-map.md`.

## Left Unchanged

- `docs/decisions.md` already matches the current project direction: local UI exists, meeting JSON matching remains next.
- Meeting notes and meeting index remain canonical and did not need restructuring.
- Specs and reference notes still point to the right homes for matching, local state, and UI behavior.

## Ambiguous / Human Review

- `archive/ui.archive-2026-05-13/` contains generated dependency/build material. It was moved rather than deleted, but it may be too heavy to keep long-term.
- The current UI has a dummy seeding script but no full CSV ingest script in `ui/scripts/`; canonical notes now mark full ingest as partial/restoration-needed.
