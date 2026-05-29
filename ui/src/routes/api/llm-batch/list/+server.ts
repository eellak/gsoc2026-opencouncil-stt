import { json } from '@sveltejs/kit';
import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';
import type { RequestHandler } from './$types';

function stateDir(): string {
	return process.env.REVIEW_STATE_DIR ?? resolve(process.cwd(), '.state');
}

interface BatchSummary {
	batch_id: string;
	model: string;
	created_at: string;
	size: number;
}

export const GET: RequestHandler = async () => {
	const dir = resolve(stateDir(), 'llm-batches');
	let names: string[];
	try {
		names = await fs.readdir(dir);
	} catch {
		return json([] as BatchSummary[]);
	}
	const out: BatchSummary[] = [];
	for (const name of names) {
		if (!name.endsWith('.json')) continue;
		try {
			const raw = await fs.readFile(resolve(dir, name), 'utf8');
			const j = JSON.parse(raw) as { batch_id: string; model: string; created_at: string; ids: string[] };
			out.push({ batch_id: j.batch_id, model: j.model, created_at: j.created_at, size: j.ids.length });
		} catch {
			/* skip */
		}
	}
	out.sort((a, b) => b.created_at.localeCompare(a.created_at));
	return json(out);
};
