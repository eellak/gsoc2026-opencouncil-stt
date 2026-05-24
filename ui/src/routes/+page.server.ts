import { redirect } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { getStatsCache } from '$lib/server/state/stats-cache';
import { parseSeedParam, randomSeed, reviewHref } from '$lib/shared/urls';
import type { IncludeStatus } from '$lib/domain/types';
import type { PageServerLoad } from './$types';

const STATUSES: ReadonlySet<IncludeStatus> = new Set([
	'include',
	'exclude',
	'uncertain',
	'unreviewed'
]);

function parseStatus(raw: string | null): IncludeStatus | null {
	if (!raw) return null;
	return STATUSES.has(raw as IncludeStatus) ? (raw as IncludeStatus) : null;
}

/**
 * Landing page behaviour:
 *   - `/?status=include` (or exclude/uncertain) → jump into the first item
 *     of that status-filtered queue. Walks the in-memory id list.
 *   - `/?seed=N` → redirect straight into the first item of that seed.
 *   - `/` (no seed, no status) → render the seed input UI + a condensed
 *     distribution row.
 */
export const load: PageServerLoad = async ({ url }) => {
	const statusParam = parseStatus(url.searchParams.get('status'));
	if (statusParam && statusParam !== 'unreviewed') {
		const repo = await getRepo();
		const ids = repo.idsByStatus(statusParam);
		if (!ids.length) throw redirect(302, `/stats/by-status/${statusParam}`);
		// `seed` is intentionally omitted in status mode — the queue is the
		// id list itself, not a seeded permutation. The review page reads
		// `?status=` and switches to filtered navigation.
		const target = new URL(`/review/${encodeURIComponent(ids[0])}`, url);
		target.searchParams.set('status', statusParam);
		throw redirect(302, target.pathname + target.search);
	}

	const seedParam = parseSeedParam(url.searchParams.get('seed'));
	if (seedParam !== null) {
		const repo = await getRepo();
		const { groups } = repo.queue(seedParam, 0, 1);
		if (!groups.length) throw redirect(302, '/stats');
		throw redirect(302, reviewHref({ utterance_id: groups[0].utterance_id, seed: seedParam }));
	}

	// No seed → show the landing page. Pre-generate a random seed so the user
	// can submit-without-typing for a fresh exploration. Also load the
	// by-status distribution so the home page shows a condensed summary.
	const repo = await getRepo();
	const cached = await getStatsCache().get(repo);
	return {
		suggestedSeed: randomSeed(),
		distribution: cached.stats.by_status,
		distributionTotal: cached.stats.total
	};
};
