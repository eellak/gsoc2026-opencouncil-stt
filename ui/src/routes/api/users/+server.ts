/**
 * GET /api/users → sorted reviewer usernames plus each reviewer's decision
 * tallies (include / exclude / uncertain / total). Counts come from the
 * sidecar's in-memory per-user map, so this stays O(users) with no log scan.
 */

import { json } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	const repo = await getRepo();
	const usernames = repo.listUsernames();
	const counts = repo.userCounts();
	const ZERO = { include: 0, exclude: 0, uncertain: 0, total: 0 };
	return json({
		usernames,
		counts,
		users: usernames.map((name) => ({ name, counts: counts[name] ?? ZERO }))
	});
};
