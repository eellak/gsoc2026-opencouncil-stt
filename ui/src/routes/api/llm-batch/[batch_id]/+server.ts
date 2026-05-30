import { json, error } from '@sveltejs/kit';
import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';
import type { RequestHandler } from './$types';

function stateDir(): string {
	return process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');
}

function isSafeId(id: string): boolean {
	return /^[a-f0-9]{4,32}$/.test(id);
}

/** DELETE /api/llm-batch/<id> — discard a batch's issued-id file (e.g. after ingest). */
export const DELETE: RequestHandler = async ({ params }) => {
	const id = params.batch_id ?? '';
	if (!isSafeId(id)) throw error(400, 'invalid batch_id');
	const path = resolve(stateDir(), 'llm-batches', `${id}.json`);
	try {
		await fs.unlink(path);
	} catch (err: unknown) {
		const e = err as NodeJS.ErrnoException;
		if (e.code !== 'ENOENT') throw error(500, e.message);
	}
	return json({ deleted: id });
};
