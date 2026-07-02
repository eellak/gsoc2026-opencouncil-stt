import { redirect } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import { getStatsCache } from '$lib/server/state/stats-cache';
import { parseSeedParam, randomSeed, reviewHref } from '$lib/shared/urls';
import {
	parseReviewFilter,
	serializeReviewFilter,
	isFilterActive,
	matchesReviewFilter
} from '$lib/shared/review-filters';
import { firstFilteredId } from '$lib/server/state/review-filter-queue';
import { queueIdSet } from '$lib/server/state/nb2-ids';
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
	// Fixed review queues (?queue=nb2 | nb2audio). Same shape as the status queue:
	// jump into the first eligible item; the review page re-applies ?queue=<name>
	// for filtered j/k navigation.
	const queueName = url.searchParams.get('queue');
	if (queueName) {
		const set = queueIdSet(queueName);
		if (set) {
			const repo = await getRepo();
			const ids = repo.eligibleOrderedIds().filter((id) => set.has(id));
			if (!ids.length) throw redirect(302, '/stats');
			const target = new URL(`/review/${encodeURIComponent(ids[0])}`, url);
			target.searchParams.set('queue', queueName);
			throw redirect(302, target.pathname + target.search);
		}
	}

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
		const skipClassified = url.searchParams.get('skip') === '1';
		// Start-time review filters (punct/src). When active, every landing path
		// below scopes to matching groups so the queue we drop the reviewer into
		// already excludes filtered-out corrections. `filterQuery` is the
		// canonical query preserved on the redirect so the review page re-applies it.
		const filterSpec = parseReviewFilter(url.searchParams);
		const filterQuery = serializeReviewFilter(filterSpec);
		const active = isFilterActive(filterSpec);

		// Resume at a specific item if the client passed one and it still
		// belongs to the eligible (navigable) universe AND passes the active
		// filter. Falls through to the walk when it's stale/ineligible/filtered.
		const resume = url.searchParams.get('resume');
		if (resume && repo.eligibleOrderedIds().includes(resume)) {
			const g = repo.getGroup(resume);
			if (g && (!active || matchesReviewFilter(g, filterSpec))) {
				throw redirect(302, reviewHref({ utterance_id: resume, seed: seedParam, filter: filterQuery }));
			}
		}
		if (skipClassified) {
			// First unreviewed group that also passes the filter. firstFilteredId
			// pages the seeded order in capped passes (no full materialisation).
			let landed = firstFilteredId(repo, seedParam, filterSpec, {
				accept: (g) => g.label.include_status === 'unreviewed',
				// Bound the unreviewed search so the all-reviewed end-state stays cheap;
				// falls back to the first matching item below.
				maxScan: 20_000
			});
			// Everything reviewed (or none unreviewed left) → land on the first
			// filter-matching item rather than bouncing to stats.
			if (!landed) landed = firstFilteredId(repo, seedParam, filterSpec);
			if (!landed) throw redirect(302, '/stats');
			throw redirect(302, reviewHref({ utterance_id: landed, seed: seedParam, filter: filterQuery }));
		}
		const landed = firstFilteredId(repo, seedParam, filterSpec);
		if (!landed) throw redirect(302, '/stats');
		throw redirect(302, reviewHref({ utterance_id: landed, seed: seedParam, filter: filterQuery }));
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
