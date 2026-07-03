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

### 2026-07-03 - KNOWN ISSUE: training-clip boundary sync (must fix before training)

The next-batch audio-verification (`?queue=nb2audio`, 4,329 edits) uses a
faithfulness gate of `cer_local <= 0.20` under **local alignment**
(`fuzz.partial_ratio(gold, soniox)`). Local alignment is deliberately tolerant of
a missing/extra syllable or neighbouring context at the clip edges — that is why
it works despite loose CSV utterance spans. So the **selection** is not broken by
boundary sloppiness.

**But the raw CSV `utterance_start/end` are loose/tight** (a clip can start or end
mid-syllable, or bracket neighbouring speech). Evidence: rejected clips skew short
(median 1.6s vs 2.1s kept); several 0.20-0.30 "rejects" are actually fine labels
whose CER was inflated by an edge syllable / number formatting / a bled-in extra
word (e.g. gold "…ή 400 ή 440 εκεί οι κλίνες" vs soniox "…400-440 οι κλίνες").
~28 kept-pool clips returned empty Soniox (transcription failures, not bad labels).

**TODO before building the actual training clips:** do NOT cut on raw CSV
boundaries. Snap to **silence / VAD** and add padding (~±0.2s) so no syllable is
clipped — the training audio must fully contain the label, or the model learns
from misaligned (audio ≠ text) examples. With a real `SONIOX_API_KEY`, the async
API returns per-token `start_ms/end_ms`, which gives exact word-level boundaries;
the realtime path does not, so use VAD there. Owner flagged this explicitly
(2026-07-03): "leave the threshold, just note the audio must be checked/fixed".

**Addressed for the published dataset (2026-07-03)** by `eval/hf_export` (spec
[hf-dataset-export](../specs/hf-dataset-export.md)): a boundary pass force-aligns
the label onto each clip (CTC forced aligner) + VAD and emits a `boundary_status`
(`ok`/`adjusted`/`suspect_cut_start`/`suspect_cut_end`/`suspect_bleed_in`/
`align_failed`) plus VAD-snapped `start_adj`/`end_adj` with ±0.2s padding.
Suspects/failures go to `data/hf-dataset/boundary-audit.csv` for sampled human
review and get `null` adjusted spans in the published files. Training-clip
builds should consume `start_adj`/`end_adj` for the `ok`/`adjusted` rows. Note:
the aligner is the PyPI `ctc-forced-aligner` (deskpai, ONNX MMS CTC, uroman
romanization for Greek) — chosen over MahmoudAshraf's git package because the
spec leaves the library open and it needs no untrusted external install.
