/**
 * Server-side scan-filter over the seeded queue.
 *
 * Walks the seeded permutation in whole batches and returns only the groups
 * that pass the review filter (and an optional extra predicate). Filtered-out
 * groups are never sent to the client, so they are never prefetched — the whole
 * point of doing this server-side instead of via a separate per-id queue.
 *
 * Cursor contract (see Codex review): `next_cursor` distinguishes "advanced but
 * found nothing yet" from "truly exhausted":
 *   - `next_cursor: number, exhausted: false` — keep paging from this offset.
 *     May carry an EMPTY `groups` array when a scan cap was hit before any match.
 *   - `next_cursor: null, exhausted: true` — the seeded order is fully consumed.
 * The client treats `next_cursor === null` as the only end condition, so the two
 * fields stay consistent (null cursor ⟺ exhausted).
 *
 * Batches advance by `page.next_cursor` (never a mid-batch offset), so a repo
 * that drops missing ids from a slice can't desync the cursor.
 */

import type { Group, QueueResponse } from '$lib/domain/groups';
import { matchesReviewFilter, type ReviewFilterSpec } from '$lib/shared/review-filters';

/** Minimal repo surface this helper needs. */
export interface SeededQueueSource {
	queue(seed: number, from: number, n: number): QueueResponse;
}

export interface FilteredQueueResult {
	cache_hash: string;
	total: number;
	groups: Group[];
	next_cursor: number | null;
	exhausted: boolean;
}

const BATCH = 50;
/** Max groups scanned per request, to bound worst-case latency on selective filters. */
const DEFAULT_SCAN_CAP = 2500;

export interface ScanOpts {
	cap?: number;
	/** Extra per-group gate (e.g. skip already-classified). Applied after the filter. */
	accept?: (g: Group) => boolean;
}

export interface FirstIdOpts extends ScanOpts {
	/**
	 * Total groups to scan before giving up (returns null). Bounds worst-case
	 * cost when the `accept` gate matches nothing late in the order (e.g. every
	 * matching item already reviewed). Default: unbounded.
	 */
	maxScan?: number;
}

/**
 * Collect up to `n` groups passing `spec` starting at seeded offset `from`.
 * Scans at most `cap` groups per call; if the cap is hit before `n` matches and
 * before exhaustion, returns what was collected with a non-null `next_cursor`
 * and `exhausted: false` so the caller pages on.
 */
export function scanFilteredQueue(
	repo: SeededQueueSource,
	seed: number,
	from: number,
	n: number,
	spec: ReviewFilterSpec,
	opts: ScanOpts = {}
): FilteredQueueResult {
	const cap = opts.cap ?? DEFAULT_SCAN_CAP;
	const accept = opts.accept;
	const want = Math.max(1, Math.min(50, Math.floor(n)));

	let cursor = Math.max(0, Math.floor(from));
	const collected: Group[] = [];
	let scanned = 0;
	let cacheHash = '';
	let total = 0;
	let exhausted = false;

	// Stops when `collected` first reaches `want`. We never split a batch, so the
	// result may overshoot `want` by up to one batch (BATCH-1) — that's
	// intentional: the cursor advances by whole batches, so trimming the extra
	// matches would silently DROP them (the next page starts past them). The
	// overshoot is bounded and harmless — the client de-dups groups by id.
	while (collected.length < want && scanned < cap) {
		const pageFrom = cursor;
		const page = repo.queue(seed, cursor, BATCH);
		cacheHash = page.cache_hash;
		total = page.total;
		for (const g of page.groups) {
			if (matchesReviewFilter(g, spec) && (!accept || accept(g))) collected.push(g);
		}
		if (page.next_cursor === null) {
			// Count the actual span consumed (a repo may drop missing ids from the
			// batch, so page.groups.length under-counts the cursor advance).
			scanned += Math.max(0, total - pageFrom);
			exhausted = true;
			break;
		}
		scanned += Math.max(0, page.next_cursor - pageFrom);
		cursor = page.next_cursor;
	}

	return {
		cache_hash: cacheHash,
		total,
		groups: collected,
		next_cursor: exhausted ? null : cursor,
		exhausted
	};
}

/**
 * First seeded id passing `spec` (and `accept`), or null if none. Used by the
 * landing redirect / skip-classified walk.
 */
export function firstFilteredId(
	repo: SeededQueueSource,
	seed: number,
	spec: ReviewFilterSpec,
	opts: FirstIdOpts = {}
): string | null {
	const maxScan = opts.maxScan ?? Infinity;
	const passCap = opts.cap ?? DEFAULT_SCAN_CAP;
	// Walk in capped passes until a match, true exhaustion, or the scan budget.
	let from = 0;
	let scanned = 0;
	// Hard bound on passes (cap-sized) as a runaway guard.
	for (let guard = 0; guard < 100_000; guard++) {
		// Never let a single pass scan past the remaining budget, so maxScan is a
		// true total bound and not just a per-pass-boundary check.
		const remaining = maxScan === Infinity ? passCap : Math.min(passCap, maxScan - scanned);
		const res = scanFilteredQueue(repo, seed, from, 1, spec, { ...opts, cap: remaining });
		if (res.groups.length > 0) return res.groups[0].utterance_id;
		if (res.next_cursor === null) return null;
		scanned += res.next_cursor - from;
		from = res.next_cursor;
		if (scanned >= maxScan) return null;
	}
	return null;
}
