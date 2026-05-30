# Storage decisions

Local review state, history log, hosted database.

## Accepted

### 2026-05-12 - Local storage: SQLite + JSONL event log

Current state in SQLite (corrections, matched utterances, cached meeting metadata, review labels). Append-only JSONL for label-change history.

Reason: SQLite gives fast filtering and stats for the UI; JSONL gives an auditable trail without complicating the schema.

### 2026-05-15 - Turso + Vercel for hosted review state

The exploration UI will be deployed on Vercel with Turso (libSQL) as the hosted database.

Reason: SQLite-compatible API (minimal code change from node:sqlite), no git commits for data changes, generous free tier, works with Bun and Vercel serverless. Local dev uses `file:./data/corrections.sqlite` via the same libsql client, so development workflow is unchanged.

Configuration: `TURSO_DATABASE_URL` + `TURSO_AUTH_TOKEN` env vars set in Vercel project settings. Never committed to git.

Migration script: `ui/scripts/migrate-to-turso.ts` (run once to copy local SQLite to Turso).

Superseded by [Supabase Postgres for live review state](#2026-05-19---supabase-postgres-for-live-review-state) after the v2 CSV landed and the table needed normalised meetings plus a larger hosted Postgres workflow.

### 2026-05-19 - Supabase Postgres for live review state

The current live review database is Supabase Postgres, project `opencouncil-edits-v2`, with schema managed by Drizzle in `ui/drizzle/schema.ts`.

Reason: the v2 CSV introduced stable `utterance_id`, `meeting_id`, and `city_id`, and the implementation now normalises meeting metadata into a separate `meetings` table. Postgres also supports the reduction and maintenance operations used for latest-per-utterance cleanup.

Implementation notes:

- Runtime code reads `DATABASE_URL` as a Supabase Postgres pooler URL.
- `ui/scripts/ingest-csv-v2.ts` upserts v2 CSV rows and normalised meetings.
- `review_labels` stores current review state.
- `events` stores review audit events.

### 2026-05-20 - File-backed prototype on `codex/file-backed-review-ui` (experimental, local-only)

An experimental branch (`codex/file-backed-review-ui`) replaces the runtime DB dependency with three files on disk. The intent is to validate the utterance-group review unit before deciding whether to commit to file-backed storage on `main`. This branch does NOT supersede the Supabase decision above; merge to main is conditional on the prototype proving out.

Layout:

- `data-1779206108158.csv` (immutable) — canonical source.
- `ui/.cache/groups.v1.json` + `meta.json` — grouped cache built by `bun ui/scripts/build-cache.ts`. Regenerated when the CSV's content hash, size, or cache version changes; force with `REBUILD=1`. Atomic writes (tmp + rename).
- `ui/.state/review-events.jsonl` — append-only event log, one JSON per PATCH.
- `ui/.state/review-labels.snapshot.json` — latest label per `utterance_id`, rewritten every 100 events; on boot the snapshot loads, then the JSONL tail is replayed (truncated final line is tolerated as a likely crash-mid-write).

Review unit on this branch is the **utterance group** (`utterance_id`), not the individual edit. Labels are group-level; the full chain of edits remains visible in the UI and in `/api/export`. See [local-data-model.md](../specs/local-data-model.md) for the on-disk shapes.

Deployment scope: **local-only**. The in-memory repo plus append-only sidecar files cannot run on serverless (Vercel) — a horizontal deploy needs the DB-backed flow on `main`.

### 2026-05-22 — Keep the `/api/oc-meeting` CORS bridge instead of direct browser fetch

The review page slices OpenCouncil meeting JSONs client-side (see [matching.md](matching.md)). The natural next step would be to drop our `/api/oc-meeting/{cityId}/{meetingId}` thin proxy and fetch directly from `https://opencouncil.gr/api/cities/{cityId}/meetings/{meetingId}`. We verified by curl that opencouncil's API does **not** return an `Access-Control-Allow-Origin` header on either `GET` or `OPTIONS`, so a direct browser fetch is blocked.

Options considered:

- (a, accepted) **Keep the thin proxy as-is.** ~50 lines, no parsing, no caching of its own beyond `Cache-Control: public, max-age=3600`. Caddy serves the response over HTTP/2 from the same origin so the browser doesn't pay a CORS preflight either.
- (b, rejected for now) **Move the bridge into Caddy as a `reverse_proxy + header` directive.** Saves a Node request handler but only relevant if Node CPU becomes the bottleneck — we expect ~one ~600 KB meeting JSON per ~minute of review, easily inside the Oracle Free shape.
- (c, future) **Ask OpenCouncil to add `Access-Control-Allow-Origin: *`** (or a specific allowlist). Cleanest long-term; out of our hands. If/when this lands, the client switches its base URL and the proxy goes away with a one-line change.

The client-side LRU keeps each fetched meeting in memory for the rest of the session, so the proxy fires at most once per (city, meeting) per session — server load is bounded regardless of how many reviewers are paging through utterances of the same meeting.
