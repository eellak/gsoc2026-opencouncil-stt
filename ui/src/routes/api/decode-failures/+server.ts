import { json } from '@sveltejs/kit';
import { appendFile, readFile } from 'fs/promises';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';
import type { RequestHandler } from './$types';

// Anchor the data path to this module's location instead of process.cwd(),
// which is unreliable under SvelteKit/Vercel adapters.
const HERE = dirname(fileURLToPath(import.meta.url));
// HERE = ui/src/routes/api/decode-failures → ../../../../../data lives at <repo-root>/data
const FAILURES_FILE = resolve(HERE, '..', '..', '..', '..', '..', 'data', 'decode-failures.jsonl');

// Bounded de-dup cache: keep only the last MAX_SEEN URLs so a long-lived
// server process doesn't accumulate unbounded memory.
const MAX_SEEN = 10_000;
const seen = new Map<string, true>(); // insertion-ordered → cheap FIFO eviction

function recordSeen(url: string): void {
	if (seen.has(url)) {
		// refresh recency
		seen.delete(url);
		seen.set(url, true);
		return;
	}
	seen.set(url, true);
	while (seen.size > MAX_SEEN) {
		const oldest = seen.keys().next().value;
		if (oldest === undefined) break;
		seen.delete(oldest);
	}
}

let seeded = false;
async function seedSeen() {
	if (seeded) return;
	seeded = true;
	try {
		const text = await readFile(FAILURES_FILE, 'utf8');
		for (const line of text.split('\n')) {
			if (!line.trim()) continue;
			try {
				const u = JSON.parse(line).originalUrl;
				if (typeof u === 'string') recordSeen(u);
			} catch { /* skip malformed */ }
		}
	} catch { /* file doesn't exist yet */ }
}

export const POST: RequestHandler = async ({ request }) => {
	let body: Record<string, unknown>;
	try { body = await request.json(); } catch { return json({ ok: false }, { status: 400 }); }

	const { originalUrl } = body;
	if (typeof originalUrl !== 'string') return json({ ok: false }, { status: 400 });

	await seedSeen();
	if (seen.has(originalUrl)) return json({ ok: true, duplicate: true });

	recordSeen(originalUrl);
	const entry = JSON.stringify({ ...body, recordedAt: new Date().toISOString() }) + '\n';
	try {
		await appendFile(FAILURES_FILE, entry, 'utf8');
	} catch (e) {
		console.error('[decode-failures] write error', e);
		return json({ ok: false }, { status: 500 });
	}
	console.info('[decode-failures] recorded', originalUrl);
	return json({ ok: true });
};
