import { json, error } from '@sveltejs/kit';
import { resolve } from 'node:path';
import { getRepo } from '$lib/server/repo';
import { buildBatch } from '$lib/server/llm-batch/service';
import type { RequestHandler } from './$types';

function stateDir(): string {
	return process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');
}

export const POST: RequestHandler = async ({ request }) => {
	let body: unknown;
	try {
		body = await request.json();
	} catch {
		throw error(400, 'body must be JSON');
	}
	if (!body || typeof body !== 'object') throw error(400, 'body must be object');
	const b = body as Record<string, unknown>;
	const n = Number(b.n ?? 200);
	const model = typeof b.model === 'string' && b.model.trim() ? b.model.trim() : 'unknown';
	const include_empty = b.include_empty === true;
	if (!Number.isFinite(n) || n < 1 || n > 500) throw error(400, 'n must be in [1, 500]');

	const repo = await getRepo();
	const batch = await buildBatch(repo, { n, model, stateDir: stateDir(), include_empty });
	return json(batch);
};
