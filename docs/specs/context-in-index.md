# Spec/Plan — bake surrounding context into the local index

> Status: **reviewed by Codex (high effort) 2026-06-19; revisions folded in
> below.** Goal: remove the live per-utterance context fetch to opencouncil.gr at
> review time; serve surrounding utterances from the local sqlite index. Keep the
> client contract identical; fall back to live upstream **per meeting** when a
> meeting's transcript isn't stored.

## Verified upstream contract (curl, 2026-06-19)

`GET /api/utterance/{id}/context?before=N&after=N` returns:
`{ meeting: {id, cityId, name, dateTime}, before: [...], after: [...] }`; each
item `{id, text, start, end, speakerTagId}`. **`before` and `after` are both
chronological ascending** (oldest→newest by `start`). The local path must emit
the same: `ORDER BY seq` ascending, `before` = rows with `seq < anchor`, `after`
= rows with `seq > anchor`.

## Why

Today the review queue/index ([sqlite-repo.ts](../../ui/src/lib/server/repo/sqlite-repo.ts))
is built **only from the corrections CSV** → it holds just the edited utterances.
Surrounding context (±N neighbours) is fetched **live per utterance** at review
time via `/api/oc-context/{id}` → `opencouncil.gr/api/utterance/{id}/context`.
That live dependency is the entire source of the 502/timeout/«Μη διαθέσιμο
πλαίσιο»/retry/auto-skip surface (and the local IPv6 flakiness). For an offline,
reproducible review+dataset tool, a frozen local snapshot is more robust, faster,
and RAM-neutral (the repo queries sqlite on-disk; nothing extra resident).

## Data source

`GET https://opencouncil.gr/api/cities/{cityId}/meetings/{meetingId}` → full
meeting JSON (same endpoint the legacy [oc-meeting bridge](../../ui/src/routes/api/oc-meeting/[city_id]/[meeting_id]/+server.ts) uses).
Shape (see [opencouncil-meeting-json.md](../reference/opencouncil-meeting-json.md)):
`transcript[]` speaker segments → each has `speakerTagId` + `utterances[]`; each
utterance has `id`, `text`, `startTimestamp`, `endTimestamp`.

## Schema — new tables (in [build-sqlite.ts](../../ui/src/lib/server/cache/build-sqlite.ts))

```sql
CREATE TABLE transcript (
  utterance_id TEXT NOT NULL,
  city_id      TEXT NOT NULL,
  meeting_id   TEXT NOT NULL,
  seq          INTEGER NOT NULL,  -- ARRAY-ORDER position within the meeting
  text         TEXT NOT NULL,
  start        REAL,              -- nullable: upstream may omit
  "end"        REAL,
  speaker_tag  TEXT,              -- speakerTagId (grouping key only; never a name)
  PRIMARY KEY (city_id, meeting_id, utterance_id)
);
CREATE UNIQUE INDEX idx_transcript_meeting_seq ON transcript(meeting_id, seq);

-- Per-meeting presence manifest: lets the bridge decide local-vs-fallback PER
-- MEETING (not just "table exists"), so a partial build is correct.
CREATE TABLE transcript_meeting (
  city_id    TEXT NOT NULL,
  meeting_id TEXT NOT NULL,
  utt_count  INTEGER NOT NULL,
  PRIMARY KEY (city_id, meeting_id)
);
```

- **`seq` = array order, NOT timestamps** (Codex: timestamps aren't guaranteed
  unique/monotonic). Walk `transcript[]` in array order, then each `utterances[]`
  in array order; assign contiguous 0-based `seq`. Frozen.
- **PK `(city_id, meeting_id, utterance_id)`** — don't trust global utterance-id
  uniqueness across meetings. `UNIQUE(meeting_id, seq)` enforces contiguity.
- Store **only eligible meetings** (same exclusion set as `groups`: public +
  non-degenerate-meeting). Private transcripts never fetched/stored.
- Holds **all** utterances of each eligible meeting (unedited + already-classified
  are valid neighbours) → more rows than `groups` (~600k+ vs 234k). On-disk → RAM
  unchanged.
- **Malformed-field policy:** missing `text` → skip that utterance **and log +
  count** (no silent drop, no seq hole within the kept set — seq is assigned over
  kept rows); null `start`/`end` stored as null. Never reject a whole meeting for
  one bad utterance. Covered by tests.

## Build pipeline ([build-cache.ts](../../ui/src/scripts/build-cache.ts) / [build.ts](../../ui/src/lib/server/cache/build.ts))

1. After eligibility is known, collect distinct eligible `(city_id, meeting_id)`.
2. **Separate, resumable-by-meeting phase** (the expensive part, ~313 fetches).
   Idempotent staging flow per meeting (Codex): fetch JSON → cache raw blob to
   `.cache/meetings/{city}__{meeting}.json` → parse/extract → insert rows →
   insert its `transcript_meeting` manifest row → next. Resume skips meetings
   already present in `transcript_meeting`. Concurrency-limited (4–6); per-request
   timeout; bounded retries w/ exponential backoff + jitter; **4xx = permanent**
   (record failure, move on), **5xx/timeout = transient** (retry). Don't buffer
   many bodies at once (1GB VM).
3. **Normal flow builds the transcript index if possible**; on partial/total
   failure it records status + failed count and the bridge falls back live for
   the missing meetings. `SKIP_TRANSCRIPT_INDEX=1` stays as an *emergency* opt-out
   only — never the default correctness path.
4. **Versioning — separate from the CSV/label snapshot** (Codex must-fix). Leave
   the existing CSV `source_hash`/`exclusions_hash` and label snapshots **exactly
   as-is**. Add independent transcript meta rows:
   - `transcript_schema_version = 1`
   - `transcript_manifest_hash = sha256(sorted eligible meeting ids + extractor
     version + schema version)`
   - `transcript_build_status = complete | partial | absent`
   - `transcript_failed_count = N`
   The transcript lookup disables **itself** on version/hash mismatch; it must
   never invalidate label state.
5. Build into the table within the fresh-index build (atomic swap of the whole
   `.sqlite` as today); mind WAL growth during bulk insert.

## Read path — [/api/oc-context/[utterance_id]](../../ui/src/routes/api/oc-context/[utterance_id]/+server.ts)

1. `const repo = await getRepo()` (same pattern as the review routes).
2. `const local = repo.getContext(id, before, after)`.
3. If non-null: return the **verified shape** — `{ meeting: {id, cityId, name,
   dateTime}, before:[{id,text,start,end,speakerTagId}], after:[...] }`, both
   arrays chronological ascending — with the existing cache headers. No network.
4. If null: **fall back to the existing live upstream proxy unchanged**. `null`
   funnels **all** of these into the one explicit fallback decision: table absent,
   table empty, `transcript_schema_version`/`transcript_manifest_hash`
   missing-or-mismatched, **this meeting not in `transcript_meeting`** (per-meeting
   miss), or `utteranceId` unknown.

## Repo — new method on the `ReviewRepo` interface ([repo/index.ts](../../ui/src/lib/server/repo/index.ts))

`getContext(utteranceId, before, after): OcContextShape | null`

- **sqlite-repo**:
  1. Resolve the **anchor** row `(city_id, meeting_id, seq)` for `utteranceId`;
     absent → null.
  2. Confirm the meeting is in `transcript_meeting`; absent → null (per-meeting
     fallback).
  3. Clamp `before`/`after` to `>= 0` **in code** (no negative arithmetic in SQL).
  4. `SELECT utterance_id,text,start,"end",speaker_tag FROM transcript
     WHERE meeting_id=? AND seq >= ?(anchor-before) AND seq <= ?(anchor+after)
     ORDER BY seq`, then **exclude the anchor by `seq == anchor_seq`** (not by
     id) → split into before (`seq < anchor`) / after (`seq > anchor`).
  - Prepared statements built once in `load()`; on any of the null-conditions
    above (incl. missing `transcript` table / failed meta check) return null.
- **file-repo** (prototype): return `null` (always live fallback) — production
  path is sqlite. Test that the route then behaves identically to today's proxy.

## Client

No contract change to [meeting-context.svelte.ts](../../ui/src/lib/client/meeting-context.svelte.ts).
The retry + distinct private/transient messages (just added) stay as the
fallback path; they fire rarely once context is local.

## Verification (TDD — these are the acceptance evals)

- **Extraction**: fixture meeting JSON → rows in exact array-`seq` order; empty
  segments produce no rows and no seq holes; `speakerTagId`/`start`/`end`/`text`
  preserved; duplicate/missing timestamps don't affect order (seq = array pos);
  missing `text` → utterance skipped + logged + counted.
- **Versioning**: CSV/label snapshot hash **unchanged** when only transcript data
  changes; `transcript_manifest_hash` changes when eligible set or
  schema/extractor version changes; missing transcript meta → fallback, **not**
  label-cache invalidation.
- **sqlite lookup**: returns local context for an indexed meeting; `null` for
  unknown id / table absent / table empty / meta missing-or-mismatched / meeting
  not in `transcript_meeting`; excludes anchor by seq; preserves ascending
  before/after order; clamps at meeting boundaries; `before=0` / `after=0`;
  anchor is first / last utterance; window exceeds bounds.
- **Route parity**: indexed fixture → `/api/oc-context/{id}` shape & ordering ==
  the verified upstream contract; unindexed/missing → live fallback, identical
  contract; logs show **no** `[oc-context]` upstream fetch on local hits.
- **Build pipeline**: each eligible meeting fetched once + blob cached; 2nd run
  reuses blobs & skips meetings already in `transcript_meeting`; interrupted build
  resumes without duplicate rows; transient failures retry+backoff; permanent
  (4xx) recorded without poisoning successes; partial completion detectable in
  meta; memory bounded across many meetings.
- **Contract-vs-live** (run before freezing): sample a known indexed meeting,
  diff local vs live for before/after cardinality, item order, field names,
  `meeting.id`/`meeting.cityId` — catches drift so fallback can't mask it.
- **Manual**: dev run, navigate, confirm no `[oc-context]` upstream calls on
  indexed hits; rebuild real index, confirm disk delta (~50–100 MB) + unchanged RSS.

## Acceptance criteria (user, 2026-06-19)

### A. No existing classification may be lost

Structurally safe by design — but must be **verified**, not assumed:

- Classifications (labels) live in `.state/` (`review-labels.snapshot.json` +
  `review-events.jsonl`), keyed by `utterance_id`. [`SidecarStore.load`](../../ui/src/lib/server/state/sidecar.ts)
  loads them independently of `cache_hash`; an index rebuild never touches them.
- The transcript table is versioned **separately**, so the rebuild does **not**
  change `source_hash`/`exclusions_hash` → `cache_hash` unchanged → stats/category
  snapshots stay valid and labels are wholly unaffected.
- **The deploy must NEVER overwrite the VM's `.state/`** — the VM holds the real
  reviewer work; the local `.state/` is only a snapshot copy.
- **Verify**: before deploy, record VM `sidecar_labels=N` (from the boot log) and
  the labelled-utterance count; after deploy + restart, assert the same N and that
  a sample of pre-existing labelled utterance_ids still resolve to their labels.
  Back up the VM `.state/` first.

### B. Deploy to the Oracle VM

- Code: via the normal main→VM path (deploy tracks `main`).
- The rebuilt `.sqlite` index is **gitignored** → transferred out-of-band
  (rsync/scp to the VM), not committed. **Resolve the exact transfer + restart
  steps at deploy time** (systemd `opencouncil-ui.service`, port 3000, no bun).
- Order: back up VM `.state/` → ship index alongside (not over) `.state/` →
  restart service → confirm boot log (`sidecar_labels` unchanged; transcript meta
  present) → smoke-test that context loads locally with no `[oc-context]` upstream
  calls.

## Rollout

- Backup `.state` (existing `.state-backups` posture) before rebuild.
- New index has the table; old indexes → live fallback. Optional `cache_version`
  bump if the meta/hash scheme requires it.

## Resolved (Codex review + curl, 2026-06-19)

1. **Versioning** → separate transcript meta; CSV/label hashing untouched (above).
2. **Build failure** → partial success, per-meeting manifest + status/failed-count;
   resumable by meeting via `transcript_meeting` + blob cache (above).
3. **Route wiring** → `getRepo()` cached singleton, same as review routes. Fine.
4. **`seq`** → array-order only, frozen; empty segments / dup-or-missing timestamps
   are non-issues because order never uses timestamps (above).
5. **Ordering** → verified: before/after chronological ascending (above).
6. **`/api/oc-meeting`** → grep callers first; retire only if unused (defer; not in
   this change's scope).
