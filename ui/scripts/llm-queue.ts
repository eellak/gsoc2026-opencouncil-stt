#!/usr/bin/env bun
/**
 * Build the LLM-judging queue: every group that currently has no
 * error_categories (auto-v1 or human) gets emitted as one JSONL line with
 * the minimum context needed for an LLM to classify.
 *
 * Output: ui/.state/llm-queue.jsonl
 *   { "utterance_id": "...", "before": "...", "after": "..." }
 *
 * Each line corresponds to one decision the LLM must make.
 */

import { promises as fs } from 'node:fs';
import { resolve, dirname } from 'node:path';
import type { Group, GroupLabel } from '../src/lib/domain/groups';
import { DEFAULT_LABEL } from '../src/lib/domain/groups';

const CACHE_PATH = resolve('.cache/groups.v1.json');
const SNAPSHOT_PATH = resolve('.state/review-labels.snapshot.json');
const EVENTS_PATH = resolve('.state/review-events.jsonl');
const OUT_PATH = resolve('.state/llm-queue.jsonl');

interface ReviewEvent {
	id: number;
	ts: string;
	utterance_id: string;
	source: string;
	patch: { error_categories?: string[]; error_category?: string | null };
}

interface Snapshot {
	last_event_id: number;
	labels: Record<string, GroupLabel>;
}

async function loadLabels(): Promise<Map<string, GroupLabel>> {
	const labels = new Map<string, GroupLabel>();
	try {
		const raw = await fs.readFile(SNAPSHOT_PATH, 'utf8');
		const snap = JSON.parse(raw) as Snapshot;
		for (const [id, lbl] of Object.entries(snap.labels)) {
			labels.set(id, { ...DEFAULT_LABEL, ...lbl });
		}
	} catch {
		/* empty */
	}
	try {
		const raw = await fs.readFile(EVENTS_PATH, 'utf8');
		for (const line of raw.split('\n')) {
			if (!line) continue;
			let ev: ReviewEvent;
			try {
				ev = JSON.parse(line);
			} catch {
				continue;
			}
			const cur = labels.get(ev.utterance_id) ?? { ...DEFAULT_LABEL };
			const next = { ...cur, error_categories: [...cur.error_categories] };
			const p = ev.patch ?? {};
			if (Array.isArray(p.error_categories)) next.error_categories = [...p.error_categories];
			else if (typeof p.error_category === 'string' && p.error_category) next.error_categories = [p.error_category];
			else if (p.error_category === null) next.error_categories = [];
			labels.set(ev.utterance_id, next);
		}
	} catch {
		/* empty */
	}
	return labels;
}

async function main(): Promise<void> {
	console.error('[queue] loading cache…');
	const groups = JSON.parse(await fs.readFile(CACHE_PATH, 'utf8')) as Group[];
	console.error(`[queue] ${groups.length} groups`);

	const labels = await loadLabels();
	console.error(`[queue] ${labels.size} labels currently in state`);

	await fs.mkdir(dirname(OUT_PATH), { recursive: true });
	const fh = await fs.open(OUT_PATH, 'w');
	let kept = 0;
	let skipped = 0;
	for (const g of groups) {
		const lbl = labels.get(g.utterance_id);
		if (lbl && lbl.error_categories.length > 0) {
			skipped++;
			continue;
		}
		const before = g.initial_before_text ?? '';
		const after = g.final_after_text ?? '';
		if (!before || !after) {
			skipped++;
			continue;
		}
		await fh.write(JSON.stringify({ utterance_id: g.utterance_id, before, after }) + '\n');
		kept++;
	}
	await fh.close();
	console.error(`[queue] wrote ${kept} entries to ${OUT_PATH} (skipped ${skipped})`);
}

main().catch((err) => {
	console.error('[fatal]', err);
	process.exit(1);
});
