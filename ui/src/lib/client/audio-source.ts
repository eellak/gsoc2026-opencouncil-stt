// Resolve an audio URL into a fallback chain.
//
// Policy (set on branch codex/file-backed-review-ui):
//   1. Try the ORIGINAL opencouncil URL directly.
//   2. On real decode/CORS failure, retry via the proxy at /api/audio?u=…
//   3. Keep any known GitHub mirror as a last-resort fallback input.
//
// We deliberately do NOT use the mirror as the primary source. GitHub serves
// the audio files without CORS, so a browser fetch against the mirror always
// fails in fetch/WebAudio paths; using the mirror preemptively only adds an
// extra round-trip on every clip.

export interface AudioSources {
	/** First-attempt URL — the unmodified opencouncil URL. */
	direct: string;
	/** Proxied form of the direct URL. */
	proxied: string;
	/** Mirror (GitHub) URL if known; null otherwise. Always served via the proxy. */
	mirrorProxied: string | null;
}

let mirrorMap: Record<string, string> | null = null;
let mirrorMapPromise: Promise<Record<string, string>> | null = null;

async function loadMirrorMap(): Promise<Record<string, string>> {
	if (mirrorMap) return mirrorMap;
	if (!mirrorMapPromise) {
		mirrorMapPromise = (async () => {
			try {
				const resp = await fetch('/api/audio/mirror-map');
				if (!resp.ok) return {};
				return (await resp.json()) as Record<string, string>;
			} catch {
				return {};
			}
		})().then((map) => {
			mirrorMap = map;
			return map;
		});
	}
	return mirrorMapPromise;
}

// Direct playback works even from origins without CORS headers because
// `<audio>` / `<video>` perform no-CORS media requests. Only JS `fetch()` and
// WebAudio decode paths need CORS. We keep `direct` set to the unmodified URL
// for the `<audio>` element; any future fetch-based code path must use
// `proxied`.

export function resolveAudioUrls(originalUrl: string): AudioSources {
	const mirror = mirrorMap?.[originalUrl] ?? null;
	const proxied = `/api/audio?u=${encodeURIComponent(originalUrl)}`;
	return {
		direct: originalUrl,
		proxied,
		mirrorProxied: mirror ? `/api/audio?u=${encodeURIComponent(mirror)}` : null
	};
}

export function ensureMirrorMapLoaded(): Promise<void> {
	return loadMirrorMap().then(() => undefined);
}
