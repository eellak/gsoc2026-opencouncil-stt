import { json, error } from '@sveltejs/kit';
import { resolve } from 'node:path';
import { getRepo } from '$lib/server/repo';
import { ingestBatch } from '$lib/server/llm-batch/service';
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
	if (typeof b.batch_id !== 'string' || !b.batch_id) throw error(400, 'batch_id required');
	if (typeof b.raw !== 'string') throw error(400, 'raw must be string');

	const repo = await getRepo();
	try {
		const result = await ingestBatch(repo, {
			batch_id: b.batch_id,
			raw: b.raw,
			stateDir: stateDir()
		});
		return json(result);
	} catch (err) {
		throw error(400, (err as Error).message);
	}
};
