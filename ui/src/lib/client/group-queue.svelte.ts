/**
 * Client-side group queue cache for the file-backed branch.
 *
 * Holds a rolling cache of seeded groups so j/k navigation is served from
 * memory while a background fetch tops the queue up to TARGET_AHEAD groups
 * past the current one.
 */

import { SvelteMap } from 'svelte/reactivity';
import type { Group, QueueResponse, GroupPatchBody } from '$lib/domain/groups';

const TARGET_AHEAD = 11; // keep ±5 neighbours warm around the current item
// Generous cap: each Group is ~2 KB so 256 entries ≈ 0.5 MB browser memory.
// The bigger ceiling means a k-press well back in a long review session is
// still served from cache instead of round-tripping to /api/review/queue.
const MAX_CACHE = 256;

// SvelteMap is reactive on .set/.delete — plain Map inside $state isn't, so
// PATCH-driven label updates would otherwise not re-trigger $derived consumers.
const cache = new SvelteMap<string, Group>();
const order = $state<string[]>([]);
let cacheHash = $state<string | null>(null);
let nextCursor = $state<number | null>(null);
let currentSeed = $state<number>(1);

const inFlight = new Map<string, Promise<void>>();

function insert(item: Group) {
	if (!cache.has(item.utterance_id)) order.push(item.utterance_id);
	cache.set(item.utterance_id, item);
}

function evictIfNeeded() {
	while (order.length > MAX_CACHE) {
		const id = order.shift();
		if (id) cache.delete(id);
	}
}

export function get(id: string): Group | undefined {
	return cache.get(id);
}

export function nextOf(id: string): Group | undefined {
	const idx = order.indexOf(id);
	if (idx < 0 || idx + 1 >= order.length) return undefined;
	return cache.get(order[idx + 1]);
}

export function prevOf(id: string): Group | undefined {
	const idx = order.indexOf(id);
	if (idx <= 0) return undefined;
	return cache.get(order[idx - 1]);
}

/**
 * Return up to `radius` neighbours on each side of `id` (no duplicates, in
 * cached order, current item omitted). Used by the audio pool to warm players.
 */
export function neighborsAround(id: string, radius: number): Group[] {
	const idx = order.indexOf(id);
	if (idx < 0) return [];
	const out: Group[] = [];
	for (let i = idx + 1; i < Math.min(order.length, idx + 1 + radius); i++) {
		const g = cache.get(order[i]);
		if (g) out.push(g);
	}
	for (let i = idx - 1; i >= Math.max(0, idx - radius); i--) {
		const g = cache.get(order[i]);
		if (g) out.push(g);
	}
	return out;
}

export function seed(): number {
	return currentSeed;
}

export function setSeed(s: number) {
	if (s === currentSeed) return;
	currentSeed = s;
	cache.clear();
	order.length = 0;
	nextCursor = null;
}

export function patchLocal(id: string, updates: Partial<Group>): void {
	const existing = cache.get(id);
	if (!existing) return;
	cache.set(id, { ...existing, ...updates });
}

export function patchLocalLabel(id: string, updates: GroupPatchBody): void {
	const existing = cache.get(id);
	if (!existing) return;
	const label = { ...existing.label };
	for (const [k, v] of Object.entries(updates) as Array<[keyof GroupPatchBody, unknown]>) {
		if (v === undefined) continue;
		(label as unknown as Record<string, unknown>)[k] = v;
	}
	cache.set(id, { ...existing, label });
}

/**
 * Fetch a seeded slice and merge into the cache. If we have a known
 * `nextCursor`, fetch from there; otherwise start at 0 and find the desired
 * group on the seeded permutation.
 */
async function fetchSlice(from: number, n: number): Promise<void> {
	// Dedup by (seed, from) only — NOT by `n`. Two near-simultaneous top-ups
	// at the same `from` shouldn't fire two parallel /api/review/queue
	// requests just because they computed slightly different `n` from
	// transiently-different `order.length`. The second call awaits the
	// first's promise.
	//
	// Edge case: if the second call wanted *more* rows than the first
	// (e.g. the user navigated further ahead during the race), the
	// coalesced result is the smaller slice. That's harmless because
	// topUp is fire-and-forget and the next navigation triggers another
	// topUp that fills any remaining gap. We accept this minor under-fetch
	// in exchange for never duplicating an in-flight network round-trip.
	const key = `${currentSeed}|${from}`;
	const existing = inFlight.get(key);
	if (existing) return existing;
	const p = (async () => {
		try {
			const resp = await fetch(`/api/review/queue?seed=${currentSeed}&from=${from}&n=${n}`);
			if (!resp.ok) return;
			const data = (await resp.json()) as QueueResponse;
			if (cacheHash && data.cache_hash !== cacheHash) {
				// Cache regenerated under us — drop everything so cursors realign.
				cache.clear();
				order.length = 0;
			}
			cacheHash = data.cache_hash;
			for (const g of data.groups) insert(g);
			nextCursor = data.next_cursor;
			evictIfNeeded();
			// Audio prefetch is intentionally disabled. Each prefetch was a
			// full-meeting MP3 (~150–250 MB) per cached group, which on the
			// seeded-random order added up to several GB per page load.
			// See ~/.claude/plans/joyful-stirring-crystal.md for the
			// segment-prefetch follow-up.
		} finally {
			inFlight.delete(key);
		}
	})();
	inFlight.set(key, p);
	return p;
}

export async function ensureLoaded(id: string): Promise<Group | undefined> {
	if (cache.has(id)) return cache.get(id);
	// Walk the seeded order from 0 until we find `id`. Bounded by total —
	// O(N/page) requests at worst, but a small loop in practice because the
	// reviewer arrives via the queue.
	let from = 0;
	for (let i = 0; i < 50; i++) {
		await fetchSlice(from, 20);
		if (cache.has(id)) return cache.get(id);
		if (nextCursor === null) break;
		from = nextCursor;
	}
	// Fallback — direct group fetch (label-only, no queue context).
	const resp = await fetch(`/api/review/group/${encodeURIComponent(id)}`);
	if (!resp.ok) return undefined;
	const g = (await resp.json()) as Group;
	insert(g);
	return g;
}

export function topUp(currentId: string): Promise<void> | void {
	const idx = order.indexOf(currentId);
	if (idx < 0) return;
	const have = order.length - 1 - idx;
	if (have >= TARGET_AHEAD) return;
	if (nextCursor === null) return;
	return fetchSlice(nextCursor, TARGET_AHEAD - have);
}
