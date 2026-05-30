/**
 * Audio prefetch pool, element-swap edition.
 *
 * Earlier design (parallel hidden warmers): pool owned a set of hidden
 * <audio> elements that preloaded the next ±N utterances, and the page
 * mounted its OWN <audio> for playback. Diagnostic e2e proved this fails
 * the central goal — Chrome stores the warmed bytes in the hidden
 * element's media buffer and DOES NOT share them with the visible
 * player. The page's <audio> issues a fresh range request even when the
 * URL is already fully buffered next door. Plus the pool's hidden
 * elements keep HTTP/1.1 sockets open, queueing the visible player's
 * fetch behind them. Bimodal latency: fast presses 30-80ms, slow
 * presses 700-2600ms (measured in e2e/audio-prefetch.spec.ts).
 *
 * This rewrite: pool owns EVERY <audio> element. The visible player IS
 * one of the pool elements. `setActive(current, neighbours)` promotes
 * the target element to the visible slot (adds `native-player` class +
 * controls, moves into the page's audio-slot div) and demotes the
 * previous one back to the hidden host. One element per utterance, ever.
 * No duplicate fetches; no socket competition.
 *
 * Page-side integration:
 *   - On mount, the page calls `attachHiddenHost(div)` (typically once
 *     in +layout) and `attachVisibleHost(div)` (in +page.svelte).
 *   - On each navigation, the page calls `setActive(current, neighbours)`.
 *   - The page reads `state.activeEl` (an $state field) to wire its own
 *     addEventListener-based handlers. The reference is stable for a
 *     given utterance: once an element is created for an id, every
 *     subsequent `setActive` for that id returns the same instance.
 *   - The page must clean up listeners on $effect cleanup since the same
 *     element is reused later.
 */

export interface Neighbour {
	utterance_id: string;
	url: string;
	/** Region start in seconds — seeked to once metadata lands so the warm fetches the right byte range. */
	start: number;
}

interface PoolItem {
	el: HTMLAudioElement;
	url: string;
	start: number;
}

// Cap. Big enough to hold current + 2×radius neighbours with headroom.
// At radius=10 that's 21 elements; 24 keeps a small buffer against churn.
const MAX_POOL = 24;

class AudioPool {
	private items = new Map<string, PoolItem>();
	private hiddenHost: HTMLDivElement | null = null;
	private visibleHost: HTMLElement | null = null;
	private activeId: string | null = null;

	// Reactive surface for the page. $state is a Svelte 5 rune; allowed in
	// .svelte.ts modules. The page does `audioPool.state.activeEl` in a
	// $derived/$effect to track the current element.
	state = $state<{ activeEl: HTMLAudioElement | null; activeId: string | null; generation: number }>({
		activeEl: null,
		activeId: null,
		generation: 0
	});

	/** Attach (or detach by passing null) the offscreen host used to park hidden pool elements. */
	attachHiddenHost(el: HTMLDivElement | null): void {
		if (el === this.hiddenHost) return;
		this.hiddenHost = el;
		if (!el) return;
		// Adopt any orphan elements (e.g., recreated host on HMR).
		for (const [id, item] of this.items) {
			if (id === this.activeId) continue;
			if (!item.el.isConnected) el.appendChild(item.el);
		}
	}

	/** Attach (or detach by passing null) the visible audio slot inside the page. */
	attachVisibleHost(el: HTMLElement | null): void {
		if (el === this.visibleHost) return;
		this.visibleHost = el;
		// If we already have an active element, place it in the new host.
		if (el && this.activeId) {
			const active = this.items.get(this.activeId);
			if (active) el.appendChild(active.el);
		}
	}

	/**
	 * Make `current` the visible/active audio. Ensure pool elements exist
	 * for every neighbour. Evict beyond MAX_POOL.
	 */
	setActive(current: Neighbour, neighbours: Neighbour[]): void {
		this.ensureHiddenHost();
		const host = this.hiddenHost;
		if (!host) return;

		// (1) Get-or-create the element for `current`.
		let active = this.items.get(current.utterance_id);
		if (!active) {
			active = this.createItem(current.url, current.start);
			this.items.set(current.utterance_id, active);
		} else if (active.url !== current.url) {
			// The URL changed for this id (rare — e.g. mirror reassignment). Reset.
			active.el.src = current.url;
			active.url = current.url;
			active.start = current.start;
		}

		// (2) Demote the previously-active element if it's a different one.
		if (this.activeId && this.activeId !== current.utterance_id) {
			const prev = this.items.get(this.activeId);
			if (prev) this.demote(prev.el);
		}

		// (3) Promote the new active. The element keeps its loaded bytes and
		// readyState — moving it across the DOM doesn't reset the media.
		this.promote(active.el);
		this.activeId = current.utterance_id;
		this.state.activeEl = active.el;
		this.state.activeId = current.utterance_id;
		this.state.generation += 1;
		// Test hook: e2e measures latency by waiting for this counter to
		// bump after a j press. The visible <audio> doesn't always fire a
		// fresh `loadstart` (when the pool already had the element loaded,
		// promotion is just class+host shuffle), so a DOM event isn't
		// reliable — the generation counter is.
		if (typeof window !== 'undefined') {
			(window as unknown as { __audioPoolGeneration?: number }).__audioPoolGeneration = this.state.generation;
		}

		// (4) Ensure neighbour elements exist (warm side).
		for (const n of neighbours) {
			if (n.utterance_id === current.utterance_id) continue;
			if (this.items.has(n.utterance_id)) continue;
			const item = this.createItem(n.url, n.start);
			this.items.set(n.utterance_id, item);
		}

		// (5) Evict.
		this.evict(current.utterance_id, neighbours);
	}

	/** Read-only accessor for the active element (used by .ts callers without rune support). */
	getActiveEl(): HTMLAudioElement | null {
		return this.state.activeEl;
	}

	private ensureHiddenHost(): void {
		if (this.hiddenHost && this.hiddenHost.isConnected) return;
		if (typeof document === 'undefined') return;
		const div = document.createElement('div');
		div.setAttribute('aria-hidden', 'true');
		div.style.position = 'absolute';
		div.style.left = '-9999px';
		div.style.top = '0';
		div.style.width = '0';
		div.style.height = '0';
		div.style.overflow = 'hidden';
		document.body.appendChild(div);
		this.hiddenHost = div;
	}

	private createItem(url: string, start: number): PoolItem {
		const el = document.createElement('audio');
		el.preload = 'auto';
		// crossOrigin is intentionally NOT set — see audio-source.ts for the
		// no-CORS rationale; opencouncil's CDN doesn't return ACAO.
		if (url) el.src = url;
		// Seek to the utterance start once metadata lands. The browser then
		// issues a Range request for the byte offset matching `start`, so the
		// warm targets the segment the user will actually play.
		if (url && start > 0) {
			const onMeta = () => {
				try {
					el.currentTime = start;
				} catch {
					/* element may have been evicted between meta and now */
				}
			};
			el.addEventListener('loadedmetadata', onMeta, { once: true });
		}
		if (this.hiddenHost) this.hiddenHost.appendChild(el);
		return { el, url, start };
	}

	private promote(el: HTMLAudioElement): void {
		el.classList.add('native-player');
		el.controls = true;
		el.style.display = '';
		if (this.visibleHost) this.visibleHost.appendChild(el);
	}

	private demote(el: HTMLAudioElement): void {
		try {
			el.pause();
		} catch {
			/* ignore */
		}
		el.classList.remove('native-player');
		el.controls = false;
		// Keep the element alive — its bytes are why we're here. Just park
		// it back in the hidden host.
		if (this.hiddenHost) this.hiddenHost.appendChild(el);
	}

	private evict(currentId: string, neighbours: Neighbour[]): void {
		const keep = new Set<string>([currentId, ...neighbours.map((n) => n.utterance_id)]);
		// Walk insertion order — Map preserves it, so oldest non-keep first.
		for (const [id, item] of [...this.items.entries()]) {
			if (this.items.size <= MAX_POOL) break;
			if (keep.has(id)) continue;
			if (id === this.activeId) continue;
			item.el.src = '';
			item.el.removeAttribute('src');
			item.el.load(); // cancels in-flight Range requests
			item.el.remove();
			this.items.delete(id);
		}
	}

	/** Test/debug only. */
	_stats() {
		return {
			size: this.items.size,
			activeId: this.activeId,
			generation: this.state.generation
		};
	}
}

export const audioPool = new AudioPool();
