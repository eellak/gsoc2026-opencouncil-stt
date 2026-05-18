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
