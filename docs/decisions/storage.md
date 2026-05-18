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
