/**
 * Audio prefetch pool: keeps a small set of hidden <audio> elements with
 * `preload="auto"` for the ±radius utterances around the current one, so the
 * browser fetches the first chunks of each upcoming MP3 *before* the user
 * presses j/k.
 *
 * Critically, we also `audioEl.currentTime = neighbour.start` after the
 * `loadedmetadata` event fires. The full meeting MP3 is ~135 MB; without the
 * seek the browser only caches the first few hundred KB (header + initial
 * frames). With the seek, it issues a second range request for the byte
 * offset corresponding to the utterance start time, populating the browser's
 * HTTP cache for that exact range. The visible player on navigation then
 * gets a cache hit when it does its own seek.
 *
 * Constraints learned from codex review:
 *   - <audio preload="auto"> is a hint; don't assume bytes have actually landed.
 *   - 11 simultaneous range requests can saturate the connection. We rate-limit
 *     to MAX_CONCURRENT warms; the rest queue and start as slots free up.
 *   - On rapid j/k the user may jump past pre-warmed items. Each warm carries a
 *     `generation` counter tied to the current utterance; when the generation
 *     bumps, in-flight warms are aborted (clearing `src` cancels Range requests
 *     in Chromium and WebKit).
 *
 * Lives in the browser only. SSR-safe via the `typeof document` guard inside
 * `ensureContainer`.
 */

export interface Neighbour {
	utterance_id: string;
	url: string;
	/** Region start in seconds. Used to force a byte-range fetch via seek. */
	start: number;
}

const MAX_CONCURRENT = 4;
const MAX_POOL = 18;

class AudioPool {
	private elements = new Map<string, HTMLAudioElement>();
	private generation = 0;
	private active = 0;
	private waiting: Array<() => void> = [];
	private container: HTMLDivElement | null = null;

	private ensureContainer(): HTMLDivElement | null {
		if (typeof document === 'undefined') return null;
		if (this.container && this.container.isConnected) return this.container;
		const el = document.createElement('div');
		el.setAttribute('aria-hidden', 'true');
		el.style.position = 'absolute';
		el.style.left = '-9999px';
		el.style.top = '0';
		el.style.width = '0';
		el.style.height = '0';
		el.style.overflow = 'hidden';
		document.body.appendChild(el);
		this.container = el;
		return el;
	}

	/**
	 * Mark `currentId` as the focus; ensure ±radius neighbours are being warmed.
	 * Aborts any warms started under a previous generation that haven't started yet.
	 */
	warm(currentId: string, neighbours: Neighbour[]): void {
		const container = this.ensureContainer();
		if (!container) return;

		this.generation += 1;
		const gen = this.generation;
		this.waiting.length = 0; // queued-but-not-started warms from the previous generation are gone

		// Make sure the current id has an element (no need to be warmed via the
		// throttle queue — the visible <audio> already drives its own loading).
		// We still want it in the LRU map so future generations don't evict it.
		if (!this.elements.has(currentId)) {
			this.elements.set(currentId, this.createEl(container, '', 0));
		}

		for (const n of neighbours) {
			if (n.utterance_id === currentId) continue;
			if (this.elements.has(n.utterance_id)) continue;
			this.schedule(gen, container, n);
		}
		this.evict(currentId, neighbours);
	}

	private schedule(gen: number, container: HTMLDivElement, n: Neighbour): void {
		const start = () => {
			if (gen !== this.generation) return; // generation moved on while we were queued
			this.active += 1;
			const el = this.createEl(container, n.url, n.start);
			this.elements.set(n.utterance_id, el);
			const release = () => {
				el.removeEventListener('loadeddata', release);
				el.removeEventListener('canplay', release);
				el.removeEventListener('error', release);
				this.active -= 1;
				const next = this.waiting.shift();
				if (next) next();
			};
			// `canplay` is the right signal here — it fires once the browser has
			// buffered enough from the *current playback position* to start
			// playing, i.e. after the seek-driven byte range has actually
			// landed. Falling back to `loadeddata` is just safety in case a
			// codec quirk prevents canplay from firing.
			el.addEventListener('canplay', release, { once: true });
			el.addEventListener('loadeddata', release, { once: true });
			el.addEventListener('error', release, { once: true });
		};
		if (this.active < MAX_CONCURRENT) {
			start();
		} else {
			this.waiting.push(start);
		}
	}

	private createEl(container: HTMLDivElement, src: string, start: number): HTMLAudioElement {
		const el = document.createElement('audio');
		el.preload = 'auto';
		// Intentionally NOT setting `crossOrigin`. Per the HTML spec, any value
		// (including the empty string) puts the element into CORS mode, and the
		// opencouncil origin doesn't return Access-Control-Allow-Origin. Hidden
		// warms only need to populate the HTTP cache, not run through WebAudio,
		// so a no-cors load is what we want.
		if (src) el.src = src;
		// Seek to the utterance's start once metadata arrives. The browser
		// then issues a second range request for the byte offset matching
		// `start`, which is the actual segment the user will hear. Without
		// this, the warm only buffers the MP3 header.
		if (src && start > 0) {
			const onMeta = () => {
				try {
					el.currentTime = start;
				} catch {
					/* element may have been evicted between meta load and now */
				}
			};
			el.addEventListener('loadedmetadata', onMeta, { once: true });
		}
		container.appendChild(el);
		return el;
	}

	/**
	 * Evict elements not in the new neighbour set + currentId, capped at MAX_POOL.
	 * Removes from the DOM and clears `src` (which cancels in-flight Range requests).
	 */
	private evict(currentId: string, neighbours: Neighbour[]): void {
		const keep = new Set<string>([currentId, ...neighbours.map((n) => n.utterance_id)]);
		for (const [id, el] of [...this.elements.entries()]) {
			if (keep.has(id)) continue;
			if (this.elements.size <= MAX_POOL) break;
			el.src = '';
			el.removeAttribute('src');
			el.load();
			el.remove();
			this.elements.delete(id);
		}
	}

	/** Test/debug only. */
	_stats() {
		return { size: this.elements.size, active: this.active, queued: this.waiting.length, generation: this.generation };
	}
}

export const audioPool = new AudioPool();
