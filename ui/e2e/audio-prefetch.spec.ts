import { test, expect, type Page } from '@playwright/test';
import { recordAudioEvents, summariseByUrl, isAudioUrl } from './lib/audio-events';

const SEED = 42;
const J_PRESSES = 5;

/** Strip query string so direct CDN URL and /api/audio?u=<same> aren't compared as different. */
function stripQuery(url: string): string {
	const q = url.indexOf('?');
	return q >= 0 ? url.slice(0, q) : url;
}

interface PoolElementSnapshot {
	src: string;
	readyState: number;
	networkState: number;
	buffered: string;
}

async function snapshotPoolElements(page: Page): Promise<PoolElementSnapshot[]> {
	return page.evaluate(() => {
		// Visible player has class `native-player`; pool elements have none.
		const els = Array.from(document.querySelectorAll<HTMLAudioElement>('audio:not(.native-player)'));
		return els.map((el) => {
			const ranges: string[] = [];
			for (let i = 0; i < el.buffered.length; i++) {
				ranges.push(`${el.buffered.start(i).toFixed(1)}-${el.buffered.end(i).toFixed(1)}`);
			}
			return {
				src: el.currentSrc || el.src || '',
				readyState: el.readyState,
				networkState: el.networkState,
				buffered: ranges.join(',') || '∅'
			};
		});
	});
}

/**
 * Diagnostic test: navigate to a seeded utterance, press j several times,
 * report what the browser actually does. Assertions are intentionally soft
 * — the *table* is the deliverable. Tune the bounds once we have a few
 * baseline runs.
 */
test('j-press prefetch: measure latency, requests, cache hits', async ({ page, baseURL }) => {
	// Init-script: capture-phase listeners that count loadstart/canplay/error
	// events on the visible player (the `native-player` element). With the
	// element-swap pool, the same DOM element can be promoted from hidden
	// pool → visible without firing a new `loadstart` (its src didn't
	// change — that's the whole point of the optimization), so these
	// counters won't always advance on j press. The test pairs them with a
	// generation counter exposed by the pool (`window.__audioPoolGeneration`)
	// which bumps on every setActive, plus a direct readyState check.
	await page.addInitScript(() => {
		const w = window as unknown as {
			__visibleAudio?: { loadStarts: number; canPlays: number; errors: number; lastError: string | null };
		};
		w.__visibleAudio = { loadStarts: 0, canPlays: 0, errors: 0, lastError: null };
		// Media events do not bubble, but they ARE delivered during the
		// capture phase to ancestor listeners. So a single capture-phase
		// listener on document catches every <audio> event without needing
		// to track element lifecycle. We filter to the visible player by
		// matching the `native-player` class (the warm-pool elements have
		// no class) so the counters reflect what the user actually hears.
		const isVisible = (t: EventTarget | null): t is HTMLAudioElement =>
			t instanceof HTMLAudioElement && t.classList.contains('native-player');
		document.addEventListener(
			'loadstart',
			(e) => { if (isVisible(e.target)) w.__visibleAudio!.loadStarts++; },
			true
		);
		document.addEventListener(
			'canplay',
			(e) => { if (isVisible(e.target)) w.__visibleAudio!.canPlays++; },
			true
		);
		document.addEventListener(
			'error',
			(e) => {
				if (!isVisible(e.target)) return;
				w.__visibleAudio!.errors++;
				const el = e.target as HTMLAudioElement;
				w.__visibleAudio!.lastError = el.error ? `code=${el.error.code} src=${el.currentSrc}` : el.currentSrc;
			},
			true
		);
	});

	// 1. Resolve a deterministic starting utterance via the seeded queue.
	const queueResp = await page.request.get(`${baseURL}/api/review/queue?seed=${SEED}&from=0&n=1`);
	expect(queueResp.ok(), 'queue endpoint must be reachable').toBeTruthy();
	const queue = (await queueResp.json()) as { groups: Array<{ utterance_id: string }> };
	const firstId = queue.groups?.[0]?.utterance_id;
	expect(firstId, 'seeded queue returned at least one group').toBeTruthy();

	// 2. Start recording network events BEFORE the page loads, so the very
	//    first audio request is captured.
	const events = await recordAudioEvents(page);

	// 3. Deep-link to the first utterance.
	const navStart = Date.now();
	await page.goto(`/review/${firstId}?seed=${SEED}`);

	// Wait for the visible <audio> element to be promoted by the pool and
	// reach HAVE_CURRENT_DATA.
	await page.waitForSelector('audio.native-player', { timeout: 15_000 });
	await page.waitForFunction(
		() => {
			const w = window as unknown as { __visibleAudio: { canPlays: number } };
			const a = document.querySelector('audio.native-player') as HTMLAudioElement | null;
			return !!a && (a.readyState >= 2 || w.__visibleAudio.canPlays > 0);
		},
		{ timeout: 15_000 }
	);
	const firstCanplay = Date.now() - navStart;

	// 4. Press j several times and time each navigation → canplay.
	const jTimings: number[] = [];
	const jUrls: string[] = [];
	let currentId = firstId;
	for (let i = 0; i < J_PRESSES; i++) {
		// Snapshot pre-press state.
		const before = await page.evaluate(() => {
			const w = window as unknown as {
				__visibleAudio: { loadStarts: number; canPlays: number; errors: number };
				__audioPoolGeneration?: number;
			};
			const a = document.querySelector('audio.native-player') as HTMLAudioElement | null;
			return {
				loadStarts: w.__visibleAudio.loadStarts,
				canPlays: w.__visibleAudio.canPlays,
				errors: w.__visibleAudio.errors,
				gen: w.__audioPoolGeneration ?? 0,
				src: a?.currentSrc ?? ''
			};
		});

		const eventCountBefore = events.length;
		const t0 = Date.now();
		await page.keyboard.press('j');

		// SvelteKit nav completes → URL pathname has changed.
		await page.waitForFunction(
			(prev) => !window.location.pathname.endsWith(`/${prev}`),
			currentId,
			{ timeout: 10_000 }
		);

		// Pool generation bumped → setActive ran for the new id. This is the
		// authoritative "the visible audio element is now the target element"
		// signal — works whether or not the element fires a fresh loadstart.
		await page.waitForFunction(
			(prevGen) => {
				const w = window as unknown as { __audioPoolGeneration?: number };
				return (w.__audioPoolGeneration ?? 0) > prevGen;
			},
			before.gen,
			{ timeout: 5_000 }
		);

		// Now wait for the visible element to be playable. Two paths:
		//   (a) Element was already loaded in the pool → readyState>=2 instantly.
		//   (b) Element was still loading → wait for canplay counter or readyState.
		// Either way we poll the live element rather than relying on the
		// loadstart→canplay sequence.
		const reachedReady = await page
			.waitForFunction(
				(prevCanplay) => {
					const w = window as unknown as { __visibleAudio: { canPlays: number; errors: number } };
					const a = document.querySelector('audio.native-player') as HTMLAudioElement | null;
					if (!a) return false;
					return (
						a.readyState >= 2 ||
						w.__visibleAudio.canPlays > prevCanplay ||
						w.__visibleAudio.errors > 0
					);
				},
				before.canPlays,
				{ timeout: 10_000 }
			)
			.then(() => true)
			.catch(() => false);
		const elapsed = Date.now() - t0;

		const after = await page.evaluate(() => {
			const w = window as unknown as { __visibleAudio: { loadStarts: number; canPlays: number; errors: number; lastError: string | null } };
			const a = document.querySelector('audio.native-player') as HTMLAudioElement | null;
			return { ...w.__visibleAudio, src: a?.currentSrc ?? '' };
		});
		const newIdMatch = page.url().match(/\/review\/([^?#/]+)/);
		const newId = newIdMatch ? decodeURIComponent(newIdMatch[1]) : '<unknown>';

		jTimings.push(elapsed);
		jUrls.push(after.src);

		// Per-press summary of events emitted DURING this navigation.
		const slice = events.slice(eventCountBefore);
		const reqs = slice.filter((e) => e.phase === 'request').length;
		const ress = slice.filter((e) => e.phase === 'response').length;
		const cached = slice.filter((e) => e.phase === 'servedFromCache').length;
		const errFlag = after.errors > before.errors ? ` ERROR(${after.lastError ?? '?'})` : '';
		const readyFlag = reachedReady ? '' : ' TIMEOUT(no canplay)';

		// Warm-status check: was the URL the visible player just landed on
		// already in flight (or finished) BEFORE the press happened? If yes,
		// the pool did its job for this navigation. If no, the press is a
		// cold load — the user is waiting on a fresh range request.
		const targetUrl = stripQuery(after.src);
		const priorRequests = events
			.slice(0, eventCountBefore)
			.filter((e) => e.phase === 'request' && stripQuery(e.url) === targetUrl);
		const priorResponses = events
			.slice(0, eventCountBefore)
			.filter((e) => e.phase === 'response' && stripQuery(e.url) === targetUrl);
		const warmStatus =
			priorResponses.length > 0
				? `WARM(res=${priorResponses.length})`
				: priorRequests.length > 0
					? `IN-FLIGHT(req=${priorRequests.length},res=0)`
					: 'COLD';

		console.log(
			`[j #${i + 1}] ${elapsed}ms  id ${newId.slice(0, 8)}…  reqs=${reqs}  res=${ress}  cacheHits=${cached}  ` +
				`urlChanged=${before.src !== after.src}  ${warmStatus}${errFlag}${readyFlag}`
		);

		// On a slow press, dump the pool's element state — exposes whether
		// the pool actually has elements for upcoming neighbours, and how
		// loaded each one is. Helps distinguish "warm aborted before it
		// started" from "warm started but bytes never arrived".
		if (elapsed >= 200) {
			const poolSnapshot = await snapshotPoolElements(page);
			console.log(`  pool state at press end (${poolSnapshot.length} hidden elements):`);
			for (const p of poolSnapshot) {
				console.log(
					`    ${p.src.slice(-40)}  rs=${p.readyState} ns=${p.networkState} buffered=${p.buffered}` +
						`${stripQuery(p.src) === targetUrl ? '  ← target' : ''}`
				);
			}
		}

		currentId = newId;
	}

	// 5. Diagnostic dump.
	console.log('\n=== j-press timings (ms) ===');
	console.table(jTimings.map((ms, i) => ({ press: i + 1, ms })));

	console.log('\n=== audio requests by URL ===');
	console.table(summariseByUrl(events));

	const totalRequests = events.filter((e) => e.phase === 'request').length;
	const totalResponses = events.filter((e) => e.phase === 'response').length;
	const totalCacheHits = events.filter((e) => e.phase === 'servedFromCache').length;
	console.log(
		`\n=== totals === firstCanplay=${firstCanplay}ms requests=${totalRequests} responses=${totalResponses} cacheHits=${totalCacheHits}`
	);

	// 6. Assertions.
	//
	// Baseline learned from this measurement loop on localhost dev:
	//   - Chrome does NOT populate the HTTP cache for the warmed range
	//     requests issued by the hidden audio-pool elements. Cache-hit
	//     counts are reliably 0 even when playback IS instant — Chrome
	//     reuses bytes via the media element's internal decoder buffer,
	//     which is invisible to Network.requestServedFromCache. So a
	//     cache-hit assertion would be a false signal. The honest signal
	//     is end-to-end j-press canplay latency.
	//   - On a warm pool, j-press latency is typically 30–80ms. The first
	//     press after pool start and presses near the warm-window edge
	//     can be slower (a few hundred ms).

	expect(totalRequests, 'audio-pool should issue prefetch requests').toBeGreaterThan(0);

	const fastPresses = jTimings.filter((ms) => ms < 200).length;
	const slowMax = Math.max(...jTimings);

	console.log(
		`\n=== verdict === ${fastPresses}/${jTimings.length} presses under 200ms; slowest=${slowMax}ms; ` +
			`htttpCacheHits=${totalCacheHits} (expected 0 — Chrome's audio decoder reuse is invisible to HTTP cache)`
	);

	// At least one j-press must hit the warm path. If zero presses are
	// under 200ms, the audio-pool is doing nothing useful and that's a hard
	// regression. (The flakiness *between* this lower bound and the ideal
	// "all presses fast" is the bug we're using this loop to diagnose —
	// don't fail the test on it; let the verdict line carry that story.)
	expect(fastPresses, 'at least one j-press should reach canplay in <200ms (warm path working)')
		.toBeGreaterThan(0);

	// Catastrophic regression bound. Element-swap pool removes the
	// visible-vs-pool fetch competition; the remaining slow-press cause is
	// "warm hadn't finished loading yet." On localhost dev hitting a CDN
	// that can have cold-cache misses, occasional ~2-3s outliers happen.
	// 15s = "something is genuinely broken" — tune down once a server-side
	// segment-slicing endpoint (200 OK, real Cache-Control) lands.
	expect(slowMax, 'worst-case j-press canplay latency').toBeLessThan(15_000);

	// Sanity: every recorded URL is an audio URL (catches regex regressions).
	for (const ev of events) {
		expect(isAudioUrl(ev.url)).toBe(true);
	}
});
