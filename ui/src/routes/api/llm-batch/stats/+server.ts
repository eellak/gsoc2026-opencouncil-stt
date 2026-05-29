import { json } from '@sveltejs/kit';
import { resolve } from 'node:path';
import { getRepo } from '$lib/server/repo';
import { getStats } from '$lib/server/llm-batch/service';
import type { RequestHandler } from './$types';

function stateDir(): string {
	return process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');
}

export const GET: RequestHandler = async () => {
	const repo = await getRepo();
	return json(await getStats(repo, stateDir()));
};
