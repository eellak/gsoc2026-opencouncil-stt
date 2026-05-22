#!/usr/bin/env bun
/**
 * Read two judge JSON files for the same batch, intersect categories per
 * utterance, and append ReviewEvents to review-events.jsonl with
 * source='llm-v1'. Only labels with non-empty intersection are written.
 *
 * Usage:
 *   bun scripts/llm-merge-batch.ts <judge1.json> <judge2.json> [--apply]
 *
 * Both files must be JSON arrays of {utterance_id, categories} entries.
 * Without --apply, prints what would be written (counts + samples).
 */

import { promises as fs } from 'node:fs';
import { resolve, dirname } from 'node:path';
import type { GroupLabel } from '../src/lib/domain/groups';
import { DEFAULT_LABEL } from '../src/lib/domain/groups';
import { TAXONOMY_MAP } from '../src/lib/shared/taxonomy';

const EVENTS_PATH = resolve('.state/review-events.jsonl');
const SNAPSHOT_PATH = resolve('.state/review-labels.snapshot.json');
const AUTO_SOURCE = 'llm-v1';

interface JudgeEntry {
	utterance_id: string;
	categories: string[];
}

interface ReviewEvent {
	id: number;
	ts: string;
	utterance_id: string;
	source: string;
	patch: { error_categories: string[] };
}

interface Snapshot {
	last_event_id: number;
	labels: Record<string, GroupLabel>;
}

async function loadLastEventId(): Promise<number> {
	try {
		const raw = await fs.readFile(SNAPSHOT_PATH, 'utf8');
		const snap = JSON.parse(raw) as Snapshot;
		let last = snap.last_event_id;
		try {
			const ev = await fs.readFile(EVENTS_PATH, 'utf8');
			for (const line of ev.split('\n')) {
				if (!line) continue;
				try {
					const e = JSON.parse(line) as ReviewEvent;
					if (e.id > last) last = e.id;
				} catch {
					/* skip */
				}
			}
		} catch {
			/* no events */
		}
		return last;
	} catch {
		return 0;
	}
}

async function alreadyLabeled(): Promise<Set<string>> {
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
		/* no events file yet */
	}
	return set;
}

function validateCategories(cats: string[]): string[] {
	const out: string[] = [];
	for (const c of cats) {
		if (typeof c !== 'string') continue;
		if (c in TAXONOMY_MAP) out.push(c);
	}
	return [...new Set(out)].sort();
}

async function main(): Promise<void> {
	const args = process.argv.slice(2);
	const apply = args.includes('--apply');
	const files = args.filter((a) => !a.startsWith('--'));
	if (files.length !== 2) {
		console.error('Usage: bun scripts/llm-merge-batch.ts <j1.json> <j2.json> [--apply]');
		process.exit(1);
	}
	const j1 = JSON.parse(await fs.readFile(files[0], 'utf8')) as JudgeEntry[];
	const j2 = JSON.parse(await fs.readFile(files[1], 'utf8')) as JudgeEntry[];

	const getCats = (x: JudgeEntry & { labels?: string[] }) =>
		Array.isArray(x.categories) ? x.categories : Array.isArray((x as { labels?: string[] }).labels) ? (x as { labels?: string[] }).labels! : [];
	const map1 = new Map(j1.map((x) => [x.utterance_id, new Set(validateCategories(getCats(x)))]));
	const map2 = new Map(j2.map((x) => [x.utterance_id, new Set(validateCategories(getCats(x)))]));

	const seen = await alreadyLabeled();

	const intersected: { utterance_id: string; categories: string[] }[] = [];
	let bothEmpty = 0;
	let disagreed = 0;
	let agreedNonEmpty = 0;
	let alreadySeen = 0;

	for (const uid of map1.keys()) {
		if (seen.has(uid)) { alreadySeen++; continue; }
		const s1 = map1.get(uid)!;
		const s2 = map2.get(uid);
		if (!s2) continue; // j2 missing this id
		const inter = [...s1].filter((c) => s2.has(c)).sort();
		if (inter.length === 0) {
			if (s1.size === 0 && s2.size === 0) bothEmpty++;
			else disagreed++;
			continue;
		}
		agreedNonEmpty++;
		intersected.push({ utterance_id: uid, categories: inter });
	}

	console.error(`Total in batch: ${map1.size}`);
	console.error(`  Already labeled (skipped): ${alreadySeen}`);
	console.error(`  Agreed non-empty:          ${agreedNonEmpty}`);
	console.error(`  Both empty:                ${bothEmpty}`);
	console.error(`  Disagreed (→ empty):       ${disagreed}`);
	console.error(`Would write: ${intersected.length} events`);

	if (!apply) {
		console.error('--- samples ---');
		for (const e of intersected.slice(0, 5)) {
			console.error(`  ${e.utterance_id}: ${JSON.stringify(e.categories)}`);
		}
		console.error('(dry-run — pass --apply to persist)');
		return;
	}

	if (intersected.length === 0) {
		console.error('Nothing to write.');
		return;
	}

	await fs.mkdir(dirname(EVENTS_PATH), { recursive: true });
	const startId = await loadLastEventId();
	const ts = new Date().toISOString();
	const events: ReviewEvent[] = intersected.map((e, i) => ({
		id: startId + i + 1,
		ts,
		utterance_id: e.utterance_id,
		source: AUTO_SOURCE,
		patch: { error_categories: e.categories }
	}));
	const payload = events.map((e) => JSON.stringify(e)).join('\n') + '\n';
	await fs.appendFile(EVENTS_PATH, payload);
	console.error(`Appended ${events.length} events; last id now ${startId + events.length}`);
}

main().catch((err) => {
	console.error('[fatal]', err);
	process.exit(1);
});
