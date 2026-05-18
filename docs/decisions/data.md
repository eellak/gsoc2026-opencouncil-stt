# Data decisions

CSV ingest, content categorisation, stable IDs, task version.

## Accepted

### 2026-05-12 - Do not block on the CSV export query

The exact external query that produced `utterance-edits-may12-26.csv` is not required before we can proceed.

Reason: the CSV already has edit text, timestamps, audio URL, meeting name, and meeting date. We can use these to join against cached meeting JSON for the prototype.

### 2026-05-12 - Task version is not required for first exploration

`TaskStatus.version` is useful for rigorous historical baseline analysis, but not required for the first dataset exploration UI.

Reason: the current job is to inspect corrections and classify error types, not to prove model-version regressions.

### 2026-05-12 - Waiting on a new corrections export with stable IDs

`meeting_name` + `meeting_date` is **not** sufficient to identify a meeting uniquely. Mentors have been asked for a new export that includes stable identifiers (likely `utterance_id`, `meeting_id`, `city_id`, `speakerSegmentId`).

Implication: until that export arrives, any matcher we write is a workaround. Once it arrives, matching becomes a direct ID join.

### 2026-05-16 - Categorise all CSV rows instead of dropping

All 379 194 rows are loaded into the database regardless of quality. Each row receives an `ingest_category` column (`clean`, `noop_edit`, `whitespace_only`, `empty_before`, `empty_after`, `embedded_reasoning`, `reversed_timestamps`, `multiline_text`) and a `cleaning_applied` tag list.

Reason: dropping rows at ingest time is irreversible and makes it impossible to inspect what was excluded. With categories in the DB, the stats UI can show counts per category and the user can browse even the rejected bins. Future re-ingestion (when the new CSV arrives with stable IDs) will re-run the same categorisation.

CSV is structurally clean (RFC-4180 quoted, parseable by `csv-parse`). Issues are content-level only: 5 219 no-ops, 1 384 multiline, 1 219 whitespace-only, 1 950 empty-before, 3 812 empty-after, 27 embedded-reasoning. 96.4% categorise as `clean`.

Scripts: `ui/scripts/analyze-csv.ts` (read-only report) and `ui/scripts/ingest-csv.ts` (full ingest with categorisation). Library of pure transforms in `ui/scripts/lib/csv-clean.ts` with unit tests.
