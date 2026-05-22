/**
 * GET /api/stats — full-corpus aggregations served from a 5-minute cache.
 *
 * The raw aggregation is ~17 s on the real corpus, so we precompute it and
 * persist to `ui/.state/stats.snapshot.json`. Subsequent requests inside the
 * TTL return in a handful of ms. After the TTL we serve stale while a
 * background recompute runs. See `lib/server/state/stats-cache.ts`.
 *
 * Clients can use the `X-Stats-Computed-At` response header (epoch ms) to
 * show a "last updated N min ago" indicator.
 */

import { json } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { getStatsCache } from '$lib/server/state/stats-cache';

export async function GET() {
	const repo = await getRepo();
	const cached = await getStatsCache().get(repo);
	const ageMs = Date.now() - cached.computedAt;
	return json(cached.stats, {
		headers: {
			'X-Stats-Computed-At': String(cached.computedAt),
			'X-Stats-Age-Ms': String(ageMs),
			'Cache-Control': 'public, max-age=30'
		}
	});
}
