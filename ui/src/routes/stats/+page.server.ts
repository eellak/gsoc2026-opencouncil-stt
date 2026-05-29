import { error } from '@sveltejs/kit';
import { getRepo } from '$lib/server/repo';
import type { StatsResponse } from '$lib/domain/types';

export async function load({ fetch }) {
	try {
		await getRepo();
	} catch (err) {
		console.error('[stats] getRepo failed', err);
		throw error(500, 'review repo unavailable');
	}

	let resp: Response;
	try {
		resp = await fetch('/api/stats');
	} catch (err) {
		console.error('[stats] /api/stats fetch failed', err);
		throw error(502, 'stats endpoint unreachable');
	}
	if (!resp.ok) {
		throw error(502, `stats endpoint ${resp.status}`);
	}

	try {
		const stats = (await resp.json()) as StatsResponse;
		const computedAtHeader = resp.headers.get('X-Stats-Computed-At');
		const computedAt = computedAtHeader ? Number(computedAtHeader) : null;
		return { stats, computedAt };
	} catch (err) {
		console.error('[stats] parse failed', err);
		throw error(502, 'stats payload invalid');
	}
}
