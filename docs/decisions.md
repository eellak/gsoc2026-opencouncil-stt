# Decisions

This file tracks project decisions and open questions. Keep it short and current. Move long reasoning into dated logs or reference notes.

## Accepted

### 2026-05-12 - Exploration before training

We will not start fine-tuning yet. First we need to understand the correction dataset and build tooling to inspect corrections with audio and context.

Reason: training on noisy or poorly understood correction pairs can optimize for the wrong target.

### 2026-05-12 - Use the large meeting JSON for the first prototype

The first exploration UI should use the existing meeting transcript JSON rather than requiring a new API endpoint.

Reason: the endpoint already contains meeting metadata, city, transcript segments, utterances, people, parties, subjects, and speaker tags. That is enough for local exploration if we can match CSV corrections to utterances.

### 2026-05-12 - Do not block on the CSV export query

The exact external query that produced `utterance-edits-may12-26.csv` is not required before we can proceed.

Reason: the CSV already has edit text, timestamps, audio URL, meeting name, and meeting date. We can use these to join against cached meeting JSON for the prototype.

### 2026-05-12 - Range requests are not a first milestone

Audio range-request support is useful later, but not a blocker for the first UI.

Reason: the first prototype can use normal browser audio playback from `audio_url` and seek to timestamps.

### 2026-05-12 - Task version is not required for first exploration

`TaskStatus.version` is useful for rigorous historical baseline analysis, but not required for the first dataset exploration UI.

Reason: the current job is to inspect corrections and classify error types, not to prove model-version regressions.

### 2026-05-12 - `utterance.text` is the corrected text, not the original

In the OpenCouncil meeting JSON, `utterance.text` reflects the **after-edit** state. The original pre-edit text exists only in the corrections export (`before_text` column), not in the live transcript JSON.

Implication: we cannot recover `before_text` by reading the meeting JSON; it must come from the corrections export. Matching CSV rows to utterances should compare `after_text` to `utterance.text`, not `before_text`.

### 2026-05-12 - Waiting on a new corrections export with stable IDs

`meeting_name` + `meeting_date` is **not** sufficient to identify a meeting uniquely. Mentors have been asked for a new export that includes stable identifiers (likely `utterance_id`, `meeting_id`, `city_id`, `speakerSegmentId`).

Implication: until that export arrives, any matcher we write is a workaround. Once it arrives, matching becomes a direct ID join.

### 2026-05-12 - Local storage: SQLite + JSONL event log

Current state in SQLite (corrections, matched utterances, cached meeting metadata, review labels). Append-only JSONL for label-change history.

Reason: SQLite gives fast filtering and stats for the UI; JSONL gives an auditable trail without complicating the schema.

### 2026-05-15 - Turso + Vercel for hosted review state

The exploration UI will be deployed on Vercel with Turso (libSQL) as the hosted database.

Reason: SQLite-compatible API (minimal code change from node:sqlite), no git commits for data changes, generous free tier, works with Bun and Vercel serverless. Local dev uses `file:./data/corrections.sqlite` via the same libsql client, so development workflow is unchanged.

Configuration: `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` env vars set in Vercel project settings. Never committed to git.

Migration script: `ui/scripts/migrate-to-turso.ts` (run once to copy local SQLite to Turso).

### 2026-05-15 - Waveform bars + peaks-cache prefetch

Switched wavesurfer.js rendering to bar mode (`barWidth:2, barGap:1, barRadius:2, normalize:true`) to eliminate the "double shape / square wave" visual artifact caused by the filled polygon rendering two mirrored shapes.

Added a module-level peaks cache (`ui/src/lib/domain/peaks-cache.ts`) that pre-decodes neighbor audio via OfflineAudioContext. The next item's waveform renders without decode lag.

Added `data-sveltekit-preload-data="eager"` on the "next item" link so SvelteKit fetches the next page's server data as soon as the current page mounts (not on hover).

### 2026-05-16 - Categorise all CSV rows instead of dropping

All 379 194 rows are loaded into the database regardless of quality. Each row receives an `ingest_category` column (`clean`, `noop_edit`, `whitespace_only`, `empty_before`, `empty_after`, `embedded_reasoning`, `reversed_timestamps`, `multiline_text`) and a `cleaning_applied` tag list.

Reason: dropping rows at ingest time is irreversible and makes it impossible to inspect what was excluded. With categories in the DB, the stats UI can show counts per category and the user can browse even the rejected bins. Future re-ingestion (when the new CSV arrives with stable IDs) will re-run the same categorisation.

CSV is structurally clean (RFC-4180 quoted, parseable by `csv-parse`). Issues are content-level only: 5 219 no-ops, 1 384 multiline, 1 219 whitespace-only, 1 950 empty-before, 3 812 empty-after, 27 embedded-reasoning. 96.4% categorise as `clean`.

Scripts: `ui/scripts/analyze-csv.ts` (read-only report) and `ui/scripts/ingest-csv.ts` (full ingest with categorisation). Library of pure transforms in `ui/scripts/lib/csv-clean.ts` with unit tests.

### 2026-05-16 - CORS workaround via Vercel proxy (pending proper fix)

Audio files at `data.opencouncil.gr` don't return `Access-Control-Allow-Origin`, so direct browser `fetch()` fails. Current workaround: `/api/audio?u=…` Vercel serverless function proxies requests server-side, forwards Range headers, and sets `Cache-Control: public, max-age=86400`.

This is a known limitation. Preferred fix: ask OpenCouncil to add `Access-Control-Allow-Origin: *` to audio file responses — zero infrastructure cost, removes the proxy entirely. Alternative: mirror the ~400 audio files to Cloudflare R2 or Vercel Blob with CORS enabled. Decision deferred to mentor sync.

See `docs/issues/ui-project.md` for the full written request.

## Open

### Diagram maintenance

The vault now contains Mermaid diagrams in:

- `CURRENT.md`
- `docs/roadmap.md`
- `docs/reference/opencouncil-meeting-json.md`
- `docs/specs/exploration-ui.md`

When the project flow, roadmap phases, matching strategy, or review state model changes, update the relevant diagram in the same edit as the text.

### Correction-to-utterance matching strategy

How should a CSV row be matched to a specific utterance in the large meeting JSON?

Likely matching fields:

- `audio_url`
- `youtube_url`
- `meeting_name`
- `meeting_date`
- `utterance_start`
- `utterance_end`
- `after_text` compared with `utterance.text`

Need to define confidence levels: exact match, timestamp-near match, text-near match, ambiguous, unmatched.

### Error taxonomy

We need a practical taxonomy that supports both LLM pre-classification and human correction in the UI.

The taxonomy should separate:

- ASR/domain errors useful for training;
- punctuation/formatting-only edits;
- semantic/contextual edits;
- timestamp/alignment problems;
- unclear or useless rows.

### Include/exclude semantics

Define exactly what `include` means.

Likely meaning: include this correction in the candidate dataset for future training/evaluation, not necessarily final training approval.

### Training/evaluation pair

For the first UI, do not force a final decision. Show each CSV correction as an edit pair.

Later, for training/evaluation, decide whether the unit is:

- individual edit pair;
- first `beforeText` to last `afterText` per utterance;
- only human edits;
- only selected included rows.
