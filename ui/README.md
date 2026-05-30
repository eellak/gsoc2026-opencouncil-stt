# OpenCouncil Correction Review UI

Local SvelteKit prototype for reviewing OpenCouncil ASR correction pairs before any training work.

The app is intentionally local-first. It reads correction rows from SQLite, lets a reviewer inspect `before_text` / `after_text` diffs with audio timing, stores human labels locally, and exports included rows as JSONL.

## Status

Implemented:

- Review screen with red/green diff, audio controls, editable timestamps, error category, include/exclude/uncertain status, notes, and previous/next navigation.
- Stats screen for category/status distributions and basic correction metadata.
- Local SQLite tables for raw corrections and review labels.
- Append-only JSONL event history when labels change.
- Dummy fixture seeding for development.

Still planned:

- Cached meeting JSON per meeting.
- Correction-to-utterance matching with confidence levels.
- Meeting ID, city ID, utterance ID, speaker/person metadata, and surrounding utterance context.
- Matched/ambiguous/unmatched reporting.

## Commands

```sh
bun install
bun run seed
bun run dev
```

The dev server defaults to:

```text
http://127.0.0.1:5174
```

Useful checks:

```sh
bun run check
bun run test
```

## Local State

Default database:

```text
ui/data/corrections.sqlite
```

Override with:

```sh
CORRECTIONS_DB=/path/to/corrections.sqlite bun run dev
```

Label-change events are written to:

```text
ui/data/events.jsonl
```

Local data files are ignored by `ui/.gitignore`.

## Routes

- `/`: correction list and filters.
- `/review/[edit_id]`: review one correction.
- `/stats`: aggregate stats.
- `/api/export`: JSONL export of included corrections.

## Current Constraint

The first implementation uses dummy fixture data. The next project step is to add cached meeting JSON and direct or confidence-based matching from CSV corrections to transcript utterances.
