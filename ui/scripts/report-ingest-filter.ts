#!/usr/bin/env bun
/**
 * Audit report for the degenerate-bin ingest filter.
 *
 * Reads the built SQLite index + the review-label sidecar and reports, per
 * ingest category (judged by each utterance's LATEST edit):
 *   - how many utterances fall in it,
 *   - whether it is in the active DROP set (DROP_INGEST_CATEGORIES),
 *   - how many DROPPED utterances already carry a human review label
 *     (include/exclude/uncertain) — i.e. review work that the filter now hides
 *     from navigation (nothing is deleted; getGroup still resolves them),
 *   - a sample of affected utterance_ids so a rollback is operational.
 *
 * Writes a Markdown report + a JSON sidecar under ../data/reports/.
 * This is read-only over the index and the .state sidecar — it changes nothing.
 *
 * Usage:
 *   bun scripts/report-ingest-filter.ts
 *   DROP_INGEST_CATEGORIES=noop_edit,empty_after bun scripts/report-ingest-filter.ts
 */

import Database from 'better-sqlite3';
import { promises as fs } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { SidecarStore } from '../src/lib/server/state/sidecar';
import { degenerateCategories, ALL_INGEST_CATEGORIES } from '../src/lib/server/state/ingest-filter';

const HERE = dirname(fileURLToPath(import.meta.url));
const UI_ROOT = resolve(HERE, '..');
const REPO_ROOT = resolve(UI_ROOT, '..');

const cacheDir = process.env.REVIEW_CACHE_DIR ?? resolve(UI_ROOT, '.cache');
const stateDir = process.env.REVIEW_STATE_DIR ?? resolve(UI_ROOT, '.state');
const dbPath = resolve(cacheDir, 'groups.v1.sqlite');
const reportsDir = resolve(REPO_ROOT, 'data', 'reports');

const SAMPLE_SIZE = 25;
const drop = degenerateCategories();

console.log(`Index : ${dbPath}`);
console.log(`State : ${stateDir}`);
console.log(`Drop  : ${drop.size ? [...drop].join(', ') : '(none — filter disabled)'}`);

const db = new Database(dbPath, { readonly: true, fileMustExist: true });
db.pragma('query_only = true');

// Latest-edit ingest_category per utterance, in one pass. Mirrors the repos'
// json_extract on the last edit so the counts match what the filter actually does.
const rows = db
	.prepare(
		`SELECT utterance_id,
		        json_extract(json, '$.edits[' || (json_array_length(json,'$.edits') - 1) || '].ingest_category') AS cat
		 FROM groups`
	)
	.all() as Array<{ utterance_id: string; cat: string | null }>;
db.close();

const sidecar = await SidecarStore.load(stateDir);
const labels = sidecar.all();

interface Bucket {
	category: string;
	dropped: boolean;
	total: number;
	labeled: number; // utterances with a non-unreviewed include_status
	include: number;
	exclude: number;
	uncertain: number;
	withErrorCategories: number;
	sample: string[];
}

const buckets = new Map<string, Bucket>();
function bucketFor(cat: string): Bucket {
	let b = buckets.get(cat);
	if (!b) {
		b = {
			category: cat,
			dropped: drop.has(cat),
			total: 0,
			labeled: 0,
			include: 0,
			exclude: 0,
			uncertain: 0,
			withErrorCategories: 0,
			sample: []
		};
		buckets.set(cat, b);
	}
	return b;
}

for (const { utterance_id, cat } of rows) {
	const key = cat ?? '(null)';
	const b = bucketFor(key);
	b.total++;
	if (b.sample.length < SAMPLE_SIZE) b.sample.push(utterance_id);
	const lbl = labels.get(utterance_id);
	if (lbl) {
		if (lbl.include_status && lbl.include_status !== 'unreviewed') {
			b.labeled++;
			if (lbl.include_status === 'include') b.include++;
			else if (lbl.include_status === 'exclude') b.exclude++;
			else if (lbl.include_status === 'uncertain') b.uncertain++;
		}
		if (lbl.error_categories && lbl.error_categories.length > 0) b.withErrorCategories++;
	}
}

// Stable order: all known categories first (config order), then any unexpected.
const ordered = [
	...ALL_INGEST_CATEGORIES.filter((c) => buckets.has(c)),
	...[...buckets.keys()].filter((c) => !ALL_INGEST_CATEGORIES.includes(c as never))
];

const total = rows.length;
const droppedBuckets = ordered.map(bucketFor).filter((b) => b.dropped);
const droppedTotal = droppedBuckets.reduce((a, b) => a + b.total, 0);
const droppedLabeled = droppedBuckets.reduce((a, b) => a + b.labeled, 0);

const stamp = new Date().toISOString().slice(0, 10);
const jsonPath = resolve(reportsDir, `ingest-filter-${stamp}.json`);
const mdPath = resolve(reportsDir, `ingest-filter-${stamp}.md`);

await fs.mkdir(reportsDir, { recursive: true });
await fs.writeFile(
	jsonPath,
	JSON.stringify(
		{
			generated_at: new Date().toISOString(),
			index: dbPath,
			drop_set: [...drop],
			total_utterances: total,
			dropped_total: droppedTotal,
			dropped_with_existing_label: droppedLabeled,
			buckets: ordered.map(bucketFor)
		},
		null,
		2
	)
);

const fmt = (n: number) => n.toLocaleString('en-US');
const lines: string[] = [];
lines.push(`# Ingest-filter audit — ${stamp}`);
lines.push('');
lines.push(`Read-only report. The filter is query-time and reversible — nothing in`);
lines.push(`the index or \`.state/\` is modified; \`getGroup(id)\` still resolves every`);
lines.push(`utterance below.`);
lines.push('');
lines.push(`- **Index:** \`${dbPath}\``);
lines.push(`- **Drop set (\`DROP_INGEST_CATEGORIES\`):** ${drop.size ? [...drop].map((c) => `\`${c}\``).join(', ') : '_(none — filter disabled)_'}`);
lines.push(`- **Total utterances:** ${fmt(total)}`);
lines.push(`- **Dropped from review + export:** ${fmt(droppedTotal)} (${((droppedTotal / total) * 100).toFixed(2)}%)`);
lines.push(`- **Of those, already carrying a human decision:** ${fmt(droppedLabeled)}`);
lines.push('');
lines.push(`## Per-category breakdown (by latest edit)`);
lines.push('');
lines.push(`| Category | Dropped? | Utterances | Labeled | include | exclude | uncertain | w/ error cats |`);
lines.push(`|---|---|--:|--:|--:|--:|--:|--:|`);
for (const cat of ordered) {
	const b = bucketFor(cat);
	lines.push(
		`| \`${b.category}\` | ${b.dropped ? '**yes**' : 'no'} | ${fmt(b.total)} | ${fmt(b.labeled)} | ${fmt(b.include)} | ${fmt(b.exclude)} | ${fmt(b.uncertain)} | ${fmt(b.withErrorCategories)} |`
	);
}
lines.push('');
if (droppedLabeled > 0) {
	lines.push(`> ⚠️ ${fmt(droppedLabeled)} dropped utterances already have a human decision.`);
	lines.push(`> They are hidden from navigation but NOT deleted. To review them, set`);
	lines.push(`> \`DROP_INGEST_CATEGORIES=""\` (disables the filter) or query them by id.`);
	lines.push('');
}
lines.push(`## Sample dropped utterance_ids (for rollback / spot-check)`);
lines.push('');
for (const cat of ordered) {
	const b = bucketFor(cat);
	if (!b.dropped || b.total === 0) continue;
	lines.push(`### \`${b.category}\` (${fmt(b.total)} total, showing ${Math.min(SAMPLE_SIZE, b.sample.length)})`);
	lines.push('');
	lines.push('```');
	lines.push(b.sample.join('\n'));
	lines.push('```');
	lines.push('');
}
lines.push(`Full machine-readable data: \`${jsonPath.replace(REPO_ROOT + '/', '')}\``);
lines.push('');

await fs.writeFile(mdPath, lines.join('\n'));

console.log(`\nWrote:\n  ${mdPath}\n  ${jsonPath}`);
console.log(
	`\nDropped ${fmt(droppedTotal)}/${fmt(total)} utterances ` +
		`(${fmt(droppedLabeled)} already labeled).`
);
