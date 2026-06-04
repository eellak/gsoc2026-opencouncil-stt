/**
 * Client-side group queue cache.
 *
 * Two modes share the same cache + order array:
 *
 *   - 'seeded' (default): server-side mulberry32 shuffle keyed by `seed`.
 *     `topUp` pages /api/review/queue, walking nextCursor forward.
 *   - 'filter' : `order` is an explicit id list returned by /api/review/ids
 *     (e.g. all utterances with include_status === 'include'). `topUp`
 *     prefetches ±5 group payloads via /api/review/group/{id}.
 *
 * Mode switches reset the cache so the two orderings can't get mixed.
 */

import { SvelteMap } from 'svelte/reactivity';
import type { Group, QueueResponse, GroupPatchBody } from '$lib/domain/groups';

const TARGET_AHEAD = 11; // keep ±5 neighbours warm around the current item
const MAX_CACHE = 256;

type Mode = 'seeded' | 'filter';

const cache = new SvelteMap<string, Group>();
const order = $state<string[]>([]);
let cacheHash = $state<string | null>(null);
let nextCursor = $state<number | null>(null);
let currentSeed = $state<number>(1);
let mode = $state<Mode>('seeded');
// Canonical filter identity, e.g. "status:include" / "category:akronymio" /
// "errorCategory:homophone". Null in seeded mode.
let filterKey = $state<string | null>(null);
let filterRevision = $state<number | null>(null);

const inFlight = new Map<string, Promise<void>>();

function insert(item: Group) {
	if (!cache.has(item.utterance_id)) order.push(item.utterance_id);
	cache.set(item.utterance_id, item);
}

function evictIfNeeded() {
	// In filter mode `order` is the canonical list — we don't evict from it.
	// We still cap the cache (group payloads) by evicting LRU group bodies
	// while leaving the id slots in `order` intact for navigation.
	if (mode === 'filter') {
		if (cache.size <= MAX_CACHE) return;
		// Drop the first cached entries that aren't near the current view.
		// Cheap heuristic: drop oldest-inserted cached ids until under cap.
		const it = cache.keys();
		while (cache.size > MAX_CACHE) {
			const next = it.next();
			if (next.done) break;
			cache.delete(next.value);
		}
		return;
	}
	while (order.length > MAX_CACHE) {
		const id = order.shift();
		if (id) cache.delete(id);
	}
}

function resetQueue() {
	cache.clear();
	order.length = 0;
	nextCursor = null;
	inFlight.clear();
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

/** Like nextOf/prevOf but returns just the id even if the group isn't cached. */
export function nextIdOf(id: string): string | undefined {
	const idx = order.indexOf(id);
	if (idx < 0 || idx + 1 >= order.length) return undefined;
	return order[idx + 1];
}
export function prevIdOf(id: string): string | undefined {
	const idx = order.indexOf(id);
	if (idx <= 0) return undefined;
	return order[idx - 1];
}

/** A group is "classified" once it carries a terminal review decision. */
function isClassified(g: Group | undefined): boolean {
	return !!g && g.label.include_status !== 'unreviewed';
}

/** True while the seeded queue still has unpaged items behind `nextCursor`. */
export function hasMoreSeeded(): boolean {
	return mode === 'seeded' && nextCursor !== null;
}

/**
 * Next unreviewed id using only the currently-loaded window — synchronous, no
 * fetch. Drives the nav-button href/gating. In seeded mode `order` and `cache`
 * are evicted together (see evictIfNeeded), so every id in `order` has a cached
 * group; an unknown lookup therefore can't happen mid-window. Filter-mode lists
 * are already pre-filtered, so we just hand back the immediate neighbour.
 */
export function nextUnreviewedIdLoaded(id: string): string | undefined {
	if (mode !== 'seeded') return nextIdOf(id);
	const start = order.indexOf(id);
	if (start < 0) return undefined;
	for (let i = start + 1; i < order.length; i++) {
		if (!isClassified(cache.get(order[i]))) return order[i];
	}
	return undefined;
}

/**
 * Next unreviewed id, paging forward through the seeded order as needed so the
 * search isn't bounded by the warm window. Returns undefined when the queue is
 * exhausted — callers must NOT fall back to the immediate (classified)
 * neighbour, or the skip is defeated at the tail.
 */
export async function nextUnreviewedId(id: string): Promise<string | undefined> {
	if (mode !== 'seeded') return nextIdOf(id);
	let idx = order.indexOf(id);
	if (idx < 0) return undefined;
	// Guard against a pathological loop; the real bound is order.length.
	for (let guard = 0; guard < 1_000_000; guard++) {
		if (idx + 1 >= order.length) {
			if (nextCursor === null) return undefined;
			const before = order.length;
			await fetchSlice(nextCursor, 50);
			if (order.length === before) return undefined; // no growth → give up
		}
		idx += 1;
		if (!isClassified(cache.get(order[idx]))) return order[idx];
	}
	return undefined;
}

/**
 * Previous unreviewed id within the retained window. Prev never pages backward
 * (the seeded cursor only walks forward), so this skips only as far back as the
 * cache still holds — acceptable: you came forward through those items.
 */
export function prevUnreviewedId(id: string): string | undefined {
	if (mode !== 'seeded') return prevIdOf(id);
	const start = order.indexOf(id);
	if (start < 0) return undefined;
	for (let i = start - 1; i >= 0; i--) {
		if (!isClassified(cache.get(order[i]))) return order[i];
	}
	return undefined;
}

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

export function currentMode(): Mode {
	return mode;
}

export function currentFilter(): string | null {
	return filterKey;
}

export function setSeed(s: number) {
	if (mode === 'seeded' && s === currentSeed) return;
	currentSeed = s;
	mode = 'seeded';
	filterKey = null;
	filterRevision = null;
	resetQueue();
}

/**
 * Switch to filter mode with an explicit id list. The caller is responsible
 * for fetching the list (see fetchFilterIds). A no-op if filter + revision
 * matches the current state.
 */
export function setFilterOrder(
	key: string,
	ids: readonly string[],
	revision: number,
	hash: string
): void {
	if (
		mode === 'filter' &&
		filterKey === key &&
		filterRevision === revision &&
		cacheHash === hash
	) {
		return;
	}
	mode = 'filter';
	filterKey = key;
	filterRevision = revision;
	cacheHash = hash;
	// Reset cache so we don't mix entries from the previous mode/revision.
	cache.clear();
	order.length = 0;
	nextCursor = null;
	inFlight.clear();
	for (const id of ids) order.push(id);
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

async function fetchSlice(from: number, n: number): Promise<void> {
	const key = `seed|${currentSeed}|${from}`;
	const existing = inFlight.get(key);
	if (existing) return existing;
	const p = (async () => {
		try {
			const resp = await fetch(`/api/review/queue?seed=${currentSeed}&from=${from}&n=${n}`);
			if (!resp.ok) return;
			const data = (await resp.json()) as QueueResponse;
			if (cacheHash && data.cache_hash !== cacheHash) {
				cache.clear();
				order.length = 0;
			}
			cacheHash = data.cache_hash;
			for (const g of data.groups) insert(g);
			nextCursor = data.next_cursor;
			evictIfNeeded();
		} finally {
			inFlight.delete(key);
		}
	})();
	inFlight.set(key, p);
	return p;
}

async function fetchGroupById(id: string): Promise<Group | undefined> {
	const key = `group|${id}`;
	const existing = inFlight.get(key);
	if (existing) {
		await existing;
		return cache.get(id);
	}
	const p = (async () => {
		try {
			const resp = await fetch(`/api/review/group/${encodeURIComponent(id)}`);
			if (!resp.ok) return;
			const g = (await resp.json()) as Group;
			cache.set(id, g);
			evictIfNeeded();
		} finally {
			inFlight.delete(key);
		}
	})();
	inFlight.set(key, p);
	await p;
	return cache.get(id);
}

export async function ensureLoaded(id: string): Promise<Group | undefined> {
	if (cache.has(id)) return cache.get(id);
	if (mode === 'filter') {
		return fetchGroupById(id);
	}
	// Seeded mode: walk the seeded order until we find `id`.
	let from = 0;
	for (let i = 0; i < 50; i++) {
		await fetchSlice(from, 20);
		if (cache.has(id)) return cache.get(id);
		if (nextCursor === null) break;
		from = nextCursor;
	}
	return fetchGroupById(id);
}

export function topUp(currentId: string): Promise<void> | void {
	if (mode === 'filter') {
		const idx = order.indexOf(currentId);
		if (idx < 0) return;
		const lo = Math.max(0, idx - 5);
		const hi = Math.min(order.length, idx + 6);
		const tasks: Promise<unknown>[] = [];
		for (let i = lo; i < hi; i++) {
			const id = order[i];
			if (!cache.has(id) && !inFlight.has(`group|${id}`)) {
				tasks.push(fetchGroupById(id));
			}
		}
		return tasks.length ? Promise.all(tasks).then(() => undefined) : undefined;
	}
	const idx = order.indexOf(currentId);
	if (idx < 0) return;
	const have = order.length - 1 - idx;
	if (have >= TARGET_AHEAD) return;
	if (nextCursor === null) return;
	return fetchSlice(nextCursor, TARGET_AHEAD - have);
}

/**
 * Fetch the id list for a filtered queue. `query` is the /api/review/ids query
 * string, e.g. "status=include", "category=akronymio", "errorCategory=homophone".
 * Returns the canonical filter key, the ids, plus cache hash + revision so
 * callers can detect staleness. The result is mirrored to sessionStorage keyed
 * by (filter, cache_hash, revision) so a reload within the same revision is fast.
 */
const SS_PREFIX = 'oc.ids.';

interface IdsResponse {
	filter: string;
	ids: string[];
	cache_hash: string;
	revision: number;
}

/** Test-only: reset all module state to a clean seeded queue. */
export function _resetForTest(): void {
	mode = 'seeded';
	filterKey = null;
	filterRevision = null;
	cacheHash = null;
	currentSeed = 1;
	resetQueue();
}

/**
 * Test-only: load an explicit seeded order + cached groups without hitting the
 * network. `nextCursor` controls whether the queue looks paged-out (null) or
 * has more behind it (a number, though fetchSlice is never called in tests).
 */
export function _loadSeededForTest(groups: Group[], cursor: number | null): void {
	mode = 'seeded';
	for (const g of groups) insert(g);
	nextCursor = cursor;
}

export async function fetchFilterIds(query: string): Promise<IdsResponse> {
	const resp = await fetch(`/api/review/ids?${query}`);
	if (!resp.ok) throw new Error(`fetchFilterIds: HTTP ${resp.status}`);
	const data = (await resp.json()) as IdsResponse;
	try {
		const key = `${SS_PREFIX}${data.filter}|${data.cache_hash}|${data.revision}`;
		sessionStorage.setItem(key, JSON.stringify(data.ids));
	} catch {
		/* quota exceeded — fine, it's a perf-only cache */
	}
	return data;
}
