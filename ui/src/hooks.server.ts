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

if (process.env.NODE_ENV !== 'test') {
	try {
		getStatsCache().startBackgroundRefresh(() => getRepo());
	} catch (err) {
		// Don't fail server startup on a stats-refresh init error — the page
		// will still serve, just paying the aggregation cost on first request.
		console.error('[hooks] startBackgroundRefresh failed', err);
	}
}

// SvelteKit doesn't require a `handle` export but linters expect at least one
// public symbol from a hooks module. A pass-through is fine.
import type { Handle } from '@sveltejs/kit';
export const handle: Handle = ({ event, resolve }) => resolve(event);
