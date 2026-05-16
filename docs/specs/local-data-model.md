# Local Data Model

Purpose: define the local state needed by the correction exploration UI.

## Status

- [ ] Draft matching rules.
- [x] Confirm storage choice.
- [x] Implement first local database/schema.
- [~] Restore or replace full CSV ingest script. Current active script seeds dummy fixtures.
- [ ] Extend schema for cached meeting JSON and matched utterances.

## Tables / Records

### `corrections`

Source: `utterance-edits-may12-26.csv`.

Fields:

- `edit_id`
- `edit_timestamp`
- `edit_updated_at`
- `before_text`
- `after_text`
- `edited_by`
- `utterance_start`
- `utterance_end`
- `audio_url`
- `youtube_url`
- `meeting_name`
- `meeting_date`

### `meeting_cache`

Source: large OpenCouncil meeting JSON endpoint.

Status: planned, not implemented in `ui/data/corrections.sqlite` yet.

Fields:

- `meeting_id`
- `city_id`
- `source_url`
- `fetched_at`
- `json_path`
- `content_hash`

### `matched_corrections`

Derived from CSV + meeting JSON.

Status: planned, not implemented in `ui/data/corrections.sqlite` yet.

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

Status: implemented in `ui/src/lib/server/db.ts` and seeded for development by `ui/scripts/seed-dummy.ts`. A previous full CSV ingest script is archived under `archive/ui.archive-2026-05-13/scripts/ingest.ts` and should be restored or replaced before full-dataset review.

Fields:

- `edit_id`
- `error_category`
- `include_status`: `unreviewed`, `include`, `exclude`, `uncertain`
- `adjusted_start`
- `adjusted_end`
- `reviewer_notes`
- `llm_category`
- `llm_include_suggestion`
- `llm_confidence`
- `updated_at`

### `review_events`

Append-only history, preferably JSONL even if current state is stored in SQLite.

Status: implemented as `ui/data/events.jsonl` by `ui/src/lib/server/events.ts` when review labels change.

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
