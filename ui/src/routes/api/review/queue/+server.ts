/**
 * GET /api/review/queue?seed=<seed>&from=<index>&n=<count>
 *
 * Returns a seeded slice of utterance groups. `from` is the offset into the
 * seeded permutation (not an utterance_id), so paging is a simple integer
 * cursor.
 *
 * Response: { cache_hash, total, groups, next_cursor }
 */

import { json, error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async ({ url }) => {
	const seedStr = url.searchParams.get('seed') ?? '1';
	const fromStr = url.searchParams.get('from') ?? '0';
	const nStr = url.searchParams.get('n') ?? '5';

	const seed = Number.parseInt(seedStr, 10);
	const from = Number.parseInt(fromStr, 10);
	const n = Number.parseInt(nStr, 10);
	if (!Number.isFinite(seed) || !Number.isFinite(from) || !Number.isFinite(n)) {
		throw error(400, 'seed, from, n must be integers');
	}

	const repo = await getRepo();
	return json(repo.queue(seed, from, n));
};
