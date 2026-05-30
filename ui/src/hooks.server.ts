/**
 * SvelteKit server hook — runs once when the Node server starts (adapter-node
 * imports this module at boot). We use it to schedule the background stats
 * recompute so the snapshot is always fresh when a reviewer hits /stats.
 *
 * Without this the only refresh path was "first request after the TTL
 * expired" (stale-while-revalidate). That made the user-facing stats page
 * occasionally pay the ~17 s aggregation, which the project's primary
 * reviewer asked us to eliminate.
 *
 * Skipped under NODE_ENV=test so vitest runs don't accidentally spin up a
 * real-process timer.
 */

import { getRepo } from '$lib/server/repo';
import { getStatsCache } from '$lib/server/state/stats-cache';
import { getCategoryCache } from '$lib/server/state/category-cache';

if (process.env.NODE_ENV !== 'test') {
	try {
		getStatsCache().startBackgroundRefresh(() => getRepo());
		// Precompute the category index too, so the first /category visit
		// doesn't pay the ~17 s build inside a page load. Both builds yield to
		// the event loop, so they don't freeze concurrent requests.
		getCategoryCache().startBackgroundBuild(() => getRepo());
	} catch (err) {
		// Don't fail server startup on an init error — pages still serve, just
		// paying the aggregation cost on first request.
		console.error('[hooks] background precompute init failed', err);
	}
}

// SvelteKit doesn't require a `handle` export but linters expect at least one
// public symbol from a hooks module. A pass-through is fine.
import type { Handle } from '@sveltejs/kit';
export const handle: Handle = ({ event, resolve }) => resolve(event);
