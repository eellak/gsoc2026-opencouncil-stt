/**
 * POST /api/stats/refresh — force an immediate full-corpus recompute.
 *
 * The GET endpoint serves a 5-minute cache (see ../+server.ts), so a reviewer
 * who just labelled items can otherwise wait up to 5 minutes before /stats
 * reflects their work. This endpoint backs the manual "refresh now" button:
 * it pays the ~17 s aggregation cost on demand and returns the fresh stats.
 */

import { json } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { getStatsCache } from '$lib/server/state/stats-cache';

export async function POST() {
	const repo = await getRepo();
	const cached = await getStatsCache().forceRecompute(repo);
	return json(cached.stats, {
		headers: {
			'X-Stats-Computed-At': String(cached.computedAt),
			'X-Stats-Age-Ms': '0',
			'Cache-Control': 'no-store'
		}
	});
}
