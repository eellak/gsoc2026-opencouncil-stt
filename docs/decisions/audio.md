# Audio decisions

CORS workaround, Vercel proxy, fixed-file URL map, range requests.

## Accepted

### 2026-05-12 - Range requests are not a first milestone

Audio range-request support is useful later, but not a blocker for the first UI.

Reason: the first prototype can use normal browser audio playback from `audio_url` and seek to timestamps.

### 2026-05-16 - Audio workaround via Vercel proxy and fixed-file map (pending proper fix)

Audio files at `data.opencouncil.gr` don't return `Access-Control-Allow-Origin`, so direct browser `fetch()` fails. Current workaround: `/api/audio?u=…` Vercel serverless function proxies requests server-side, forwards Range headers, and sets `Cache-Control: public, max-age=86400`.

Some source MP3s also fail browser/WebAudio decoding even when server-side ffmpeg can read them. For those, the current operational workaround is to re-encode clean MP3 copies, store original-to-fixed mappings in `data/audio-fix/url-map.json`, apply them into `corrections.audio_cdn_url`, and let the UI prefer `audio_cdn_url` while preserving the original `audio_url`.

This is a known limitation. Preferred fix: ask OpenCouncil to add `Access-Control-Allow-Origin: *` to audio file responses and fix or regenerate malformed MP3s at the source. Alternative: mirror fixed audio files to a durable bucket such as Cloudflare R2 or Vercel Blob with CORS enabled. GitHub Release hosting is useful for the current prototype, but is not yet a durable product decision.

See `docs/issues/ui-project.md` for the full written request.

### 2026-05-20 - Audio source order: original first, proxy fallback, mirror last (branch `codex/file-backed-review-ui`)

Before this change the file-backed branch was pre-applying the GitHub mirror as `audio_cdn_url` for every URL in `data/audio-fix/url-map.json`, so the audio path treated GitHub as the primary source. The mirror is only a workaround for known-broken originals.

Current policy for the native audio player:

1. Try the original OpenCouncil URL directly in `<audio>`.
2. If native playback errors, retry via `/api/audio?u=<original>`.
3. Keep the GitHub mirror map available for future fallback work, but do not make it the default source.

`audio_cdn_url` on the `Group` stays `null` on this branch. The mirror map can be fetched by the client (`/api/audio/mirror-map`) and should only be used as a last-resort fallback. Real failures still write to `data/decode-failures.jsonl` for triage.

GitHub mirror serves without `Access-Control-Allow-Origin`, so any future JS-fetch or WebAudio path must route the mirror through the proxy. If even the proxied mirror fails, use a different mirror host.

### 2026-05-20 - Waveform component removed pending a segment-render library (branch `codex/file-backed-review-ui`)

Even with the original-first policy above, a single page load was downloading ~4 GB of audio. Three overlapping causes:

1. `schedulePrefetch()` in the client queue called `fetch(url).arrayBuffer()` for every cached group. With seeded-random order, the 5 next groups almost always belonged to different meetings → 5 unique full-meeting MP3s pulled per top-up.
2. `wavesurfer.js` independently downloaded the full meeting MP3 to compute peaks (needed because we never pre-computed them).
3. A duplicate `<audio preload="metadata">` element in the same component opened a third parallel fetch.

Net: up to 6 unique URLs × 3 downloaders ≈ 18 full-file fetches of 150–250 MB recordings, observed live at ~4 GB transferred for a single page load.

For this iteration we removed the visual waveform entirely:

- Deleted `ui/src/lib/components/Waveform.svelte`, `ui/src/lib/domain/peaks-cache.ts`, `ui/src/lib/client/review-queue.svelte.ts`, `ui/src/lib/mock/` and `ui/src/routes/mock/[edit_id]/`.
- Removed `wavesurfer.js` from `ui/package.json`.
- Stripped `schedulePrefetch()` and `prefetchAudioBytes()` from `ui/src/lib/client/group-queue.svelte.ts`. The legacy per-edit queue was also deleted, since it depended on the now-gone peaks cache.
- `ui/src/routes/review/[utterance_id]/+page.svelte` now renders a single native `<audio controls preload="none">` pointing at `/api/audio?u=…`. Playback uses byte-range requests served by `ui/src/routes/api/audio/+server.ts` (which already streams + forwards `Range`). First playback starts after one ~64 KB chunk lands; no full-file fetch.

The reviewer loses the visual waveform until we ship a follow-up: a `/api/audio/segment?u=&start=&end=` endpoint (ffmpeg with `-ss/-t`, streaming ~50–200 KB extracted MP3 per utterance) plus a JS waveform library that renders from that small blob.

Cache key for the future segment endpoint will be `sha1(originalUrl + '|' + canonical_start + '|' + canonical_end)` — reviewer timestamp edits will not be cached, to keep the on-disk segment cache bounded.
