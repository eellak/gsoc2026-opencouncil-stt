import type { Page } from '@playwright/test';

/**
 * One observation of an audio-relevant network event.
 *
 * `phase` distinguishes the three things we care about:
 *   - 'request'        — request was issued (network or cache, we don't know yet)
 *   - 'response'       — a response came back from the network
 *   - 'servedFromCache' — Chromium emitted Network.requestServedFromCache (HTTP/disk cache hit, no network)
 */
export interface AudioNetEvent {
	phase: 'request' | 'response' | 'servedFromCache';
	requestId: string;
	url: string;
	method?: string;
	range?: string | null;
	status?: number;
	mimeType?: string;
	fromDiskCache?: boolean;
	fromServiceWorker?: boolean;
	t: number;
}

const AUDIO_URL_RE = /(data\.opencouncil\.gr\/.+\.mp3)|(\/api\/audio(?:\?|$))/;

export function isAudioUrl(url: string): boolean {
	return AUDIO_URL_RE.test(url);
}

/**
 * Attach a CDP Network listener to `page` and start recording every audio
 * request/response/cache-hit into the returned array.
 *
 * The whole point of going through CDP rather than `page.on('request')` is
 * `Network.requestServedFromCache`, which is the only reliable "served from
 * HTTP cache, no network" signal Chromium exposes. Without it we can't tell
 * apart a fast cached response from a fast fresh response.
 */
export async function recordAudioEvents(page: Page): Promise<AudioNetEvent[]> {
	const events: AudioNetEvent[] = [];
	const session = await page.context().newCDPSession(page);
	await session.send('Network.enable');

	const start = Date.now();
	// requestServedFromCache carries only a requestId. To resolve it back to a
	// URL we need a complete requestId→URL map — INCLUDING requests we
	// otherwise filter out of `events`. So we always populate this map on
	// requestWillBeSent, then filter on what we push into `events`.
	const idToUrl = new Map<string, string>();

	session.on('Network.requestWillBeSent', (e) => {
		idToUrl.set(e.requestId, e.request.url);
		if (!isAudioUrl(e.request.url)) return;
		events.push({
			phase: 'request',
			requestId: e.requestId,
			url: e.request.url,
			method: e.request.method,
			range: (e.request.headers['range'] ?? e.request.headers['Range']) ?? null,
			t: Date.now() - start
		});
	});

	session.on('Network.responseReceived', (e) => {
		if (!isAudioUrl(e.response.url)) return;
		events.push({
			phase: 'response',
			requestId: e.requestId,
			url: e.response.url,
			status: e.response.status,
			mimeType: e.response.mimeType,
			fromDiskCache: e.response.fromDiskCache,
			fromServiceWorker: e.response.fromServiceWorker,
			t: Date.now() - start
		});
	});

	session.on('Network.requestServedFromCache', (e) => {
		const url = idToUrl.get(e.requestId) ?? '<unknown>';
		// Only record audio cache hits (consistent with the rest of the log).
		if (!isAudioUrl(url)) return;
		events.push({
			phase: 'servedFromCache',
			requestId: e.requestId,
			url,
			t: Date.now() - start
		});
	});

	return events;
}

/**
 * Convenience: collapse a flat event log into a per-URL summary table for
 * console.table output.
 */
export function summariseByUrl(events: AudioNetEvent[]) {
	const byUrl = new Map<string, { requests: number; responses: number; cacheHits: number; statuses: Set<number> }>();
	for (const ev of events) {
		const slot = byUrl.get(ev.url) ?? { requests: 0, responses: 0, cacheHits: 0, statuses: new Set<number>() };
		if (ev.phase === 'request') slot.requests++;
		if (ev.phase === 'response') {
			slot.responses++;
			if (ev.status) slot.statuses.add(ev.status);
		}
		if (ev.phase === 'servedFromCache') slot.cacheHits++;
		byUrl.set(ev.url, slot);
	}
	return Array.from(byUrl.entries()).map(([url, s]) => ({
		url: url.length > 90 ? url.slice(0, 87) + '...' : url,
		req: s.requests,
		res: s.responses,
		cache: s.cacheHits,
		statuses: Array.from(s.statuses).join(',')
	}));
}
