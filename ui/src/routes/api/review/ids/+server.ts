/**
 * GET /api/review/ids — id list for a filtered review queue, plus cache hash
 * and label revision so clients can detect staleness before walking it.
 *
 * Exactly one filter must be given:
 *   ?status=include|exclude|uncertain|unreviewed   (by review decision)
 *   ?category=<ingest_category>                      (by CSV ingest classifier)
 *   ?errorCategory=<taxonomy_id>                     (by human-assigned category)
 *
 * Response: { filter, ids: string[], cache_hash, revision }
 * `filter` is the canonical key, e.g. "status:include" / "category:akronymio".
 *
 * Results are memoised per (filter, cache_hash, revision), so repeat hits in a
 * review session are O(1). The memo invalidates the moment the sidecar appends
 * a new event (revision bumps).
 */

import { json, error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { getCategoryCache } from '$lib/server/state/category-cache';
import { normalizeTaxonomyId } from '$lib/shared/taxonomy';
import type { IncludeStatus } from '$lib/domain/types';
import type { RequestHandler } from './$types';

const STATUSES: ReadonlySet<IncludeStatus> = new Set([
	'include',
	'exclude',
	'uncertain',
	'unreviewed'
]);

const memo = new Map<string, { ids: string[] }>();
const MEMO_MAX = 12;

export const GET: RequestHandler = async ({ url }) => {
	const status = url.searchParams.get('status');
	const category = url.searchParams.get('category');
	const errorCategory = url.searchParams.get('errorCategory');

	const repo = await getRepo();
	const revision = repo.labelsRevision;
	const cacheHash = repo.hash;

	let filter: string;
	let computeIds: () => string[] | Promise<string[]>;
	if (status) {
		if (!STATUSES.has(status as IncludeStatus)) {
			throw error(400, 'status must be one of include|exclude|uncertain|unreviewed');
		}
		filter = `status:${status}`;
		computeIds = () => repo.idsByStatus(status as IncludeStatus);
	} else if (category) {
		filter = `category:${category}`;
		computeIds = () => getCategoryCache().ids(repo, category);
	} else if (errorCategory) {
		const norm = normalizeTaxonomyId(errorCategory) ?? errorCategory;
		filter = `errorCategory:${norm}`;
		computeIds = () => repo.idsByErrorCategory(norm);
	} else {
		throw error(400, 'one of status, category, errorCategory is required');
	}

	const key = `${filter}|${cacheHash}|${revision}`;
	let entry = memo.get(key);
	if (!entry) {
		entry = { ids: await computeIds() };
		// Tiny bounded FIFO — a handful of filters per revision at most.
		if (memo.size >= MEMO_MAX) {
			const oldest = memo.keys().next().value;
			if (oldest !== undefined) memo.delete(oldest);
		}
		memo.set(key, entry);
	}

	return json(
		{ filter, ids: entry.ids, cache_hash: cacheHash, revision },
		{ headers: { 'Cache-Control': 'no-store' } }
	);
};
