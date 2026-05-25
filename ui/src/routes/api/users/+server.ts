/**
 * GET /api/users → sorted list of distinct reviewer usernames seen in the event log.
 */

import { json } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import type { RequestHandler } from './$types';

export const GET: RequestHandler = async () => {
	const repo = await getRepo();
	return json({ usernames: repo.listUsernames() });
};
