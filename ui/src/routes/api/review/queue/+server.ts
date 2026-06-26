/**
 * GET /api/review/queue?seed=<seed>&from=<index>&n=<count>
 *
 * Returns a seeded slice of utterance groups. `from` is the offset into the
 * seeded permutation (not an utterance_id), so paging is a simple integer
 * cursor.
 *
 * Optional start-time filters (`punct=drop`, `src=task,both,user`) narrow which
 * groups the seeded queue surfaces. When present, the endpoint scan-filters the
 * seeded order server-side and returns only matching groups, so filtered-out
 * items are never sent and never prefetched. Absent → byte-identical to before.
 *
 * Response: { cache_hash, total, groups, next_cursor, exhausted? }
 */

import { json, error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { parseSeedParam } from '$lib/shared/urls';
import { parseReviewFilter, isFilterActive } from '$lib/shared/review-filters';
import { scanFilteredQueue } from '$lib/server/state/review-filter-queue';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
	const seedStr = url.searchParams.get('seed') ?? '1';
	const fromStr = url.searchParams.get('from') ?? '0';
	const nStr = url.searchParams.get('n') ?? '5';

	const seed = parseSeedParam(seedStr) ?? 1;
	const from = Number.parseInt(fromStr, 10);
	const n = Number.parseInt(nStr, 10);
	if (!Number.isInteger(from) || !Number.isInteger(n)) {
		throw error(400, 'from, n must be integers');
	}
	if (from < 0 || n < 1 || n > 100) {
		throw error(400, 'from must be ≥ 0; n must be in [1, 100]');
	}

	const repo = await getRepo();

	const filter = parseReviewFilter(url.searchParams);
	if (isFilterActive(filter)) {
		return json(scanFilteredQueue(repo, seed, from, n, filter));
	}

	// Unfiltered path: passthrough, with `exhausted` made explicit so the client
	// end-condition is uniform across both paths.
	const res = repo.queue(seed, from, n);
	return json({ ...res, exhausted: res.next_cursor === null });
};
