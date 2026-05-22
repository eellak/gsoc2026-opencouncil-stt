#!/usr/bin/env bun
/**
 * Emit the next N unlabeled queue entries as a JSON array, ready to paste
 * into a subagent prompt. "Unlabeled" = no label in current events stream.
 *
 * Usage:
 *   bun scripts/llm-batch-prep.ts <batchSize> [--skip N] [--out path]
 *
 * Default: 50 items, written to stdout.
 */

import { promises as fs } from 'node:fs';
import { resolve } from 'node:path';

const QUEUE_PATH = resolve('.state/llm-queue.jsonl');
const EVENTS_PATH = resolve('.state/review-events.jsonl');

interface QueueEntry {
	utterance_id: string;
	before: string;
	after: string;
}

interface ReviewEvent {
	id: number;
	source: string;
	utterance_id: string;
	patch: { error_categories?: string[] };
}

async function loadLabeledSet(): Promise<Set<string>> {
	const set = new Set<string>();
	try {
		const raw = await fs.readFile(EVENTS_PATH, 'utf8');
		for (const line of raw.split('\n')) {
			if (!line) continue;
			try {
				const ev = JSON.parse(line) as ReviewEvent;
				if (ev.patch?.error_categories && ev.patch.error_categories.length > 0) {
					set.add(ev.utterance_id);
				}
			} catch {
				/* skip */
			}
		}
	} catch {
		/* no events yet */
	}
	return set;
}

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	let n = 50;
	let skip = 0;
	let out: string | null = null;
	for (let i = 0; i < args.length; i++) {
		const a = args[i];
		if (a === '--skip') skip = parseInt(args[++i], 10);
		else if (a === '--out') out = args[++i];
		else if (!a.startsWith('--')) n = parseInt(a, 10);
	}
	const labeled = await loadLabeledSet();
	const queue = (await fs.readFile(QUEUE_PATH, 'utf8'))
		.split('\n')
		.filter((l) => l.length > 0)
		.map((l) => JSON.parse(l) as QueueEntry);

	const remaining = queue.filter((q) => !labeled.has(q.utterance_id));
	console.error(`[prep] queue=${queue.length}, labeled=${labeled.size}, remaining=${remaining.length}`);
	const slice = remaining.slice(skip, skip + n);
	const payload = JSON.stringify(slice);
	if (out) {
		await fs.writeFile(out, payload);
		console.error(`[prep] wrote ${slice.length} items to ${out}`);
	} else {
		process.stdout.write(payload);
	}
}

main().catch((err) => {
	console.error('[fatal]', err);
	process.exit(1);
});
