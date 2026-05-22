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

### 2026-05-19 - Stable IDs export arrived

A second export landed at the repo root (`data-1779206108158.csv`, ~246 MB, 397 556 rows) carrying the IDs that were pending in [2026-05-12 - Waiting on a new corrections export with stable IDs](#2026-05-12---waiting-on-a-new-corrections-export-with-stable-ids):

- `utterance_id` (the stable utterance identifier)
- `meeting_id` (FK into the new normalised `meetings` table)
- `city_id`

`ui/scripts/ingest-csv-v2.ts` reads this CSV and upserts directly into the Postgres schema. The first ingest landed 393 970 rows after CSV-level filtering.

### 2026-05-19 - Normalise meetings out of corrections

`meeting_name`, `meeting_date`, `city_id`, `audio_url`, `youtube_url`, and `audio_cdn_url` are no longer stored per-correction. They live on a new `meetings` table (PK `meeting_id`, 242 rows) and `corrections` joins via `meeting_id`.

Reason: the unnormalised layout duplicated those columns across hundreds of thousands of rows for only 242 distinct meetings — roughly 110 MB of redundant text plus the index overhead. With normalisation the corrections table dropped from 463 MB to ~106 MB and stays well under the Supabase free-tier ceiling.

Implementation note: `corrections.audio_cdn_url` was moved to `meetings.audio_cdn_url`; `scripts/apply-audio-cdn-map.ts` still writes the same key, just on a different table.

### 2026-05-19 - Keep only the latest edit per utterance

`corrections` stores **one row per `utterance_id`** — the most recent edit, ordered by `COALESCE(edit_updated_at, edit_timestamp) DESC, edit_id DESC`. The 106 365 superseded chain edits are not kept in the live DB.

Reason: for the training/evaluation dataset the only useful signal is the final corrected text. Intermediate edits in a chain are noise: they capture transient states (e.g. mid-typo, accidental space, partial paste) that the reviewer themselves discarded in the next edit. Loading the chain into the review UI also wastes reviewer time on rows that are already known to be superseded.

Numbers from the CSV (see [data/reports/latest-per-utterance.md](../../data/reports/latest-per-utterance.md) for distribution and worked examples):

- 287 605 unique utterances total
- 70.2 % had a single edit (no chain)
- 27 % had a 2- or 3-edit chain
- ~0.4 % had 5+ edits; the longest chain is 27 edits on one utterance

Implication: if at some point we want the full chain for audit or to study reviewer behaviour, we re-ingest `data-1779206108158.csv` into a separate `corrections_history` table — the CSV is the source of truth. The decision is reversible without data loss.

Implementation: `latest_per_utterance` flag computed via window function in `ui/scripts/ingest-csv-v2.ts` follow-up, non-latest rows deleted in batches, table compacted via `VACUUM FULL` to bring DB size from 568 MB to 215 MB.
