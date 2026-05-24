/**
 * GET /api/review/ids?status=include|exclude|uncertain|unreviewed
 *
 * Returns the full id list for a given review status, plus the cache hash and
 * label revision so clients can detect staleness before walking the queue.
 *
 * Response: { status, ids: string[], cache_hash, revision }
 *
 * The result is memoised per (status, cache_hash, revision) so repeat hits
 * during a review session are O(1). Cache invalidates the moment the sidecar
 * appends a new event.
 */

import { json, error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import type { IncludeStatus } from '$lib/domain/types';
import type { RequestHandler } from './$types';

const STATUSES: ReadonlySet<IncludeStatus> = new Set([
	'include',
	'exclude',
	'uncertain',
	'unreviewed'
]);

type CacheKey = string; // `${status}|${cache_hash}|${revision}`
const memo = new Map<CacheKey, { ids: string[] }>();
const MEMO_MAX = 8;

export const GET: RequestHandler = async ({ url }) => {
	const status = url.searchParams.get('status') as IncludeStatus | null;
	if (!status || !STATUSES.has(status)) {
		throw error(400, 'status must be one of include|exclude|uncertain|unreviewed');
	}
	const repo = await getRepo();
	const revision = repo.labelsRevision;
	const cacheHash = repo.hash;
	const key: CacheKey = `${status}|${cacheHash}|${revision}`;
	let entry = memo.get(key);
	if (!entry) {
		entry = { ids: repo.idsByStatus(status) };
		// Tiny bounded LRU — we don't expect more than 4 keys per revision.
		if (memo.size >= MEMO_MAX) {
			const oldest = memo.keys().next().value;
			if (oldest !== undefined) memo.delete(oldest);
		}
		memo.set(key, entry);
	}
	return json(
		{ status, ids: entry.ids, cache_hash: cacheHash, revision },
		{
			headers: {
				'Cache-Control': 'no-store'
			}
		}
	);
};
