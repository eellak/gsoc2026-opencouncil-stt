import { getRepo } from '$lib/server/repo';
import type { StatsResponse } from '$lib/domain/types';

export async function load({ fetch }) {
	await getRepo();
	const resp = await fetch('/api/stats');
	const stats = (await resp.json()) as StatsResponse;
	return { stats };
}
