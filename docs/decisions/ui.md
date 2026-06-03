# UI decisions

Exploration-vs-training stance and prototype UI choices.

## Accepted

### 2026-06-03 - Auto-skip unavailable utterances in review (branch `codex/file-backed-review-ui`)

An utterance whose OpenCouncil context can't be fetched can't be reviewed, so the UI auto-skips it. We learn this when the per-utterance context bridge (`/api/oc-context`) returns a non-OK status. The bridge passes through upstream `401/403/404` (classified client-side as `error_kind: 'private'`); `5xx`/timeout/network stay `502` (`'transient'`). On a `'private'` error the review page auto-advances **one utterance** in the last navigation direction. A consecutive-skip cap (25) pauses the loop with a banner so a run of failures can't sweep the queue; an available load resets the streak. No label is written — skip is navigation only.

**Empirically verified status (2026-06-03):** the live `…/api/utterance/{id}/context` endpoint returns **`404 {"error":"Utterance not found"}`** for unavailable utterances and `200` for available ones — there is no distinct `403`. We skip **per utterance** on its own fetch rather than marking a whole meeting unavailable, for two reasons: (1) `meeting_id` slugs collide across cities (see data.md — `feb26_2025` is both a private Athens meeting that 404s and a public Chania one that 200s under the same slug), so a meeting-level memo keyed loosely would mis-skip; and (2) per-utterance skip is robust regardless of whether a meeting is uniformly available. In the default seeded (shuffled) order a meeting's utterances aren't adjacent anyway, so a per-meeting memo would buy almost nothing. Transient `502` never auto-skips — conservative by design.

Files: `ui/src/lib/client/auto-skip.svelte.ts`, `meeting-context.svelte.ts`, `routes/api/oc-context/[utterance_id]/+server.ts`, `routes/review/[utterance_id]/+page.svelte`.

### 2026-05-20 - Multi-category labels, seed UX, direct-CDN audio (branch `codex/file-backed-review-ui`)

Five small UX changes shipped together so the prototype is share-link friendly and exercises more of the taxonomy.

1. **Multi-category per utterance.** `GroupLabel.error_category: string | null` becomes `error_categories: string[]`. Legacy single-value sidecar records normalize to a one-element array on read; writes emit only the new shape. Lossy in spirit — historical records can carry at most one category. Canonical normalizer lives at `ui/src/lib/domain/labels.ts`. Stats UI re-labelled "category assignments" because the sum exceeds reviewed-utterance count.
2. **Seed UX.** `DEFAULT_SEED = 1` removed. `/` with no `?seed=` renders a landing input (empty = random uint32). `/?seed=N` redirects straight to the first item under that seed. Seed propagates through every `reviewHref`/`editHref`/`errorCategoryHref` (`ui/src/lib/shared/urls.ts`). Strict `^\d+$` parse, range `[0, 2^32-1]`. Two reviewers with the same seed see the same first 100 items in the same order. Share button on the review page copies the canonical URL.
3. **Stats clickable error_category column.** Rows now link to `/error-category/{taxonomy_id}`, a new file-backed list page that mirrors the existing `/category/[ingest_category]` structure but pivots on the human label.
4. **`/edit/[edit_id]` deep link.** New server route resolves `edit_id` → `utterance_id` via a lazy index on `FileRepo`, then 302s to `/review/{utterance_id}?seed=N&highlight={edit_id}`. The review page flashes a banner pointing at the highlighted edit. Fixes the broken `/category/[ingest_category]/+page.svelte:46` link to `/review/{edit_id}` which had been 404'ing.
5. **Direct-CDN audio + ±5 player pool.** `<audio>` media tags do no-CORS requests, so direct playback from `data.opencouncil.gr` works even without `Access-Control-Allow-Origin`. Removed `data.opencouncil.gr` from `KNOWN_CORS_BLOCKED` in `audio-source.ts`; the visible player now points at the direct URL with `onerror` fallback to `/api/audio?u=…`. New `ui/src/lib/client/audio-pool.svelte.ts` keeps a ±5 LRU pool of hidden `<audio preload="auto">` elements, capped at 3 concurrent warms with a generation counter so j/k spam aborts stale warms. Bytes per visit stay under ~5 MB — no return to the 4 GB regression that killed the previous prefetch.

Deferred: flag-pairs-of-utterances (no implementation, captured for later); per-edit label model (stays per-utterance/group).

### 2026-05-12 - Exploration before training

We will not start fine-tuning yet. First we need to understand the correction dataset and build tooling to inspect corrections with audio and context.

Reason: training on noisy or poorly understood correction pairs can optimize for the wrong target.

### 2026-05-15 - Waveform bars + peaks-cache prefetch

Switched wavesurfer.js rendering to bar mode (`barWidth:2, barGap:1, barRadius:2, normalize:true`) to eliminate the "double shape / square wave" visual artifact caused by the filled polygon rendering two mirrored shapes.

Added a module-level peaks cache (`ui/src/lib/domain/peaks-cache.ts`) that pre-decodes neighbor audio via OfflineAudioContext. The next item's waveform renders without decode lag.

Added `data-sveltekit-preload-data="eager"` on the "next item" link so SvelteKit fetches the next page's server data as soon as the current page mounts (not on hover).
