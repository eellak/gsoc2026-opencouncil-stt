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
import type { IncludeStatus } from '$lib/domain/types';

const TARGET_AHEAD = 11; // keep ±5 neighbours warm around the current item
const MAX_CACHE = 256;

type Mode = 'seeded' | 'filter';

const cache = new SvelteMap<string, Group>();
const order = $state<string[]>([]);
let cacheHash = $state<string | null>(null);
let nextCursor = $state<number | null>(null);
let currentSeed = $state<number>(1);
let mode = $state<Mode>('seeded');
let filterStatus = $state<IncludeStatus | null>(null);
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

export function currentFilterStatus(): IncludeStatus | null {
	return filterStatus;
}

export function setSeed(s: number) {
	if (mode === 'seeded' && s === currentSeed) return;
	currentSeed = s;
	mode = 'seeded';
	filterStatus = null;
	filterRevision = null;
	resetQueue();
}

/**
 * Switch to filter mode with an explicit id list. The caller is responsible
 * for fetching the list (see fetchStatusIds). A no-op if status + revision
 * matches the current state.
 */
export function setStatusOrder(
	status: IncludeStatus,
	ids: readonly string[],
	revision: number,
	hash: string
): void {
	if (
		mode === 'filter' &&
		filterStatus === status &&
		filterRevision === revision &&
		cacheHash === hash
	) {
		return;
	}
	mode = 'filter';
	filterStatus = status;
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
 * Fetch (or reuse a cached) ids list for a given status. Stores it in
 * sessionStorage keyed by (status, cache_hash, revision) so a page reload
 * within the same revision serves the list synchronously. Returns the array
 * plus the revision so callers can detect staleness.
 */
const SS_PREFIX = 'oc.ids.';

interface IdsResponse {
	status: IncludeStatus;
	ids: string[];
	cache_hash: string;
	revision: number;
}

export async function fetchStatusIds(status: IncludeStatus): Promise<IdsResponse> {
	const resp = await fetch(`/api/review/ids?status=${status}`);
	if (!resp.ok) throw new Error(`fetchStatusIds: HTTP ${resp.status}`);
	const data = (await resp.json()) as IdsResponse;
	try {
		const key = `${SS_PREFIX}${status}|${data.cache_hash}|${data.revision}`;
		sessionStorage.setItem(key, JSON.stringify(data.ids));
	} catch {
		/* quota exceeded — fine, it's a perf-only cache */
	}
	return data;
}

export function peekCachedStatusIds(
	status: IncludeStatus,
	hash: string,
	revision: number
): string[] | null {
	try {
		const raw = sessionStorage.getItem(`${SS_PREFIX}${status}|${hash}|${revision}`);
		if (!raw) return null;
		return JSON.parse(raw) as string[];
	} catch {
		return null;
	}
}
