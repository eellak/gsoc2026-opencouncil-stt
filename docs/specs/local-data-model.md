# Local Data Model

Purpose: define the local state needed by the correction exploration UI.

> Note: an experimental branch (`codex/file-backed-review-ui`) supplements
> the DB-backed model below with a file-backed group cache; see
> [File-backed prototype](#file-backed-prototype-codexfile-backed-review-ui-2026-05-20)
> at the bottom.

## Status

- [ ] Draft matching rules.
- [x] Confirm storage choice.
- [x] Implement first local database/schema.
- [x] Replace full CSV ingest with v2 stable-ID ingest.
- [ ] Extend schema for cached meeting JSON and matched utterances.

## Tables / Records

### `corrections`

Source: `data-1779206108158.csv` for current review data. `utterance-edits-may12-26.csv` is retained as the v1 reference export.

Fields:

- `edit_id`
- `utterance_id`
- `meeting_id`
- `edit_timestamp`
- `edit_updated_at`
- `before_text`
- `after_text`
- `edited_by`
- `utterance_start`
- `utterance_end`
- `ingest_category`
- `cleaning_applied`
- `latest_per_utterance`

Current live DB keeps only the latest edit per `utterance_id`. Superseded chain edits remain in the CSV; see [decisions/data.md](../decisions/data.md#2026-05-19---keep-only-the-latest-edit-per-utterance).

### `meetings`

Source: `data-1779206108158.csv`, normalised out of correction rows.

Fields:

- `meeting_id`
- `meeting_name`
- `meeting_date`
- `city_id`
- `audio_url`
- `audio_cdn_url`
- `youtube_url`

### `meeting_cache`

Source: large OpenCouncil meeting JSON endpoint.

Status: planned, not implemented in the current Postgres schema yet.

Fields:

- `meeting_id`
- `city_id`
- `source_url`
- `fetched_at`
- `json_path`
- `content_hash`

### `matched_corrections`

Derived from CSV + meeting JSON.

Status: planned, not implemented in the current Postgres schema yet.

Fields:

- `edit_id`
- `utterance_id`
- `speaker_segment_id`
- `speaker_tag_id`
- `person_id`
- `city_id`
- `meeting_id`
- `match_confidence`
- `match_reason`
- `context_before`
- `context_after`

### `review_labels`

Human/LLM review state.

Status: implemented in `ui/drizzle/schema.ts`. Current hosted state uses Supabase Postgres via `DATABASE_URL`; v2 ingest is handled by `ui/scripts/ingest-csv-v2.ts`.

Fields:

- `edit_id`
- `error_category`
- `include_status`: `unreviewed`, `include`, `exclude`, `uncertain`
- `adjusted_start`
- `adjusted_end`
- `reviewer_notes`
- `human_updated_at`

### `review_events`

Append-only history of review-label changes.

Status: implemented as the Postgres `events` table. Earlier JSONL history was used by the local SQLite prototype.

Example:

```json
{"ts":"2026-05-12T12:00:00Z","edit_id":"...","field":"include_status","old":"unreviewed","new":"include","actor":"human"}
```

## Acceptance Criteria

- [x] The UI can filter and update labels without rewriting the source CSV.
- [ ] Every matched correction can be traced to both CSV row and utterance JSON record.
- [ ] Ambiguous/unmatched rows are preserved.
- [~] Human labels and LLM labels are stored separately.
- [x] Label changes have history.

## File-backed prototype (`codex/file-backed-review-ui`, 2026-05-20)

The experimental branch keeps the CSV as the canonical source and removes the runtime DB dependency. State lives in three files under `ui/`:

### `ui/.cache/groups.v1.json` + `meta.json`

Built by `bun ui/scripts/build-cache.ts`. The cache is regenerated when the CSV's `(size, sha256-prefix-of-head+tail+size)` fingerprint changes — pure mtime is not trusted. `meta.json` records `source_hash`, `source_size`, `cache_version`, `generated_at`, `group_count`, `edit_count`, `missing_utterance_id_count`. Writes are atomic (tmp + rename) so a crash mid-write leaves the previous cache intact.

A `Group` (see `ui/src/lib/domain/groups.ts`) contains:

- `utterance_id`, `meeting_id`, `city_id`, `meeting_name`, `meeting_date`
- `audio_url`, `audio_cdn_url` (currently `null` on this branch; the mirror map is a fallback input, not the primary source), `youtube_url`
- `start`, `end` — taken from the latest edit's timestamps
- `initial_before_text` — earliest `before_text` in the chain
- `final_after_text` — latest `after_text` in the chain
- `edits[]` — sorted by `(edit_timestamp asc, csv_row asc)`; csv_row tiebreaks equal timestamps deterministically
- `chain_consistent: boolean` — false if any `edits[i].before_text !== edits[i-1].after_text`
- `label: GroupLabel` — re-hydrated from the sidecar on read

### `ui/.state/review-events.jsonl` (append-only)

One JSON object per PATCH: `{ id, ts, utterance_id, source: "local", patch }`. Patch semantics:

- omitted fields = no change
- explicit `null` = clear the field

Validated before append: `include_status ∈ {unreviewed, include, exclude, uncertain}`, `adjusted_*` numeric/finite/non-negative, `adjusted_start < adjusted_end`. Writes are serialised through a single in-process queue (single-node assumption).

### `ui/.state/review-labels.snapshot.json`

`{ last_event_id, labels: { [utterance_id]: GroupLabel } }`. Rewritten atomically every 100 events. On startup the sidecar loads the snapshot, then replays JSONL events with `id > last_event_id`; a truncated final JSONL line is tolerated (likely crash-mid-write), corruption earlier in the file throws loudly.

### Review unit

On this branch the review unit is the **utterance group**, not the individual edit. Labels are group-level; intermediate edits are visible via the UI's chain-toggle and exported in `/api/export`. The legacy DB-backed flow on `main` is per-edit and is unaffected.
