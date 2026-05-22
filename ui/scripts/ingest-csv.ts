#!/usr/bin/env bun
/**
 * Ingest corrections CSV into Turso (or local libsql file).
 * Usage:
 *   TURSO_DATABASE_URL=... TURSO_AUTH_TOKEN=... bun run scripts/ingest-csv.ts [path/to/csv]
 *   bun run scripts/ingest-csv.ts   # uses local file:./data/corrections.sqlite + default CSV
 *
 * Idempotent: INSERT OR REPLACE on edit_id.
 * Reads schema from server/db.ts (same as the app), then adds ingest_category + cleaning_applied columns.
 */

import { createClient } from '@libsql/client';
import { parse } from 'csv-parse';
import { createReadStream } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { categorise, type RawCsvRow } from './lib/csv-clean';
import { INGEST_CATEGORIES } from '../src/lib/domain/categories';

const __dirname = dirname(fileURLToPath(import.meta.url));
const csvPath = resolve(process.argv[2] ?? join(__dirname, '..', '..', 'utterance-edits-may12-26.csv'));
const dbUrl = process.env.TURSO_DATABASE_URL ?? `file:${join(__dirname, '..', 'data', 'corrections.sqlite')}`;
const authToken = process.env.TURSO_AUTH_TOKEN;

console.log(`CSV:  ${csvPath}`);
console.log(`DB:   ${dbUrl}`);

const client = createClient({ url: dbUrl, authToken });

// Apply base schema (same as the app's applySchema) + ingest-specific columns.
await client.executeMultiple(`
	CREATE TABLE IF NOT EXISTS corrections (
		edit_id          TEXT PRIMARY KEY,
		edit_timestamp   TEXT NOT NULL,
		edit_updated_at  TEXT,
		before_text      TEXT NOT NULL,
		after_text       TEXT NOT NULL,
		edited_by        TEXT,
		utterance_start  REAL NOT NULL,
		utterance_end    REAL NOT NULL,
		audio_url        TEXT NOT NULL,
		youtube_url      TEXT,
		meeting_name     TEXT,
		meeting_date     TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_corrections_ts      ON corrections(edit_timestamp);
	CREATE INDEX IF NOT EXISTS idx_corrections_meeting ON corrections(meeting_name, meeting_date);
	CREATE INDEX IF NOT EXISTS idx_corrections_editor  ON corrections(edited_by);

	CREATE TABLE IF NOT EXISTS review_labels (
		edit_id          TEXT PRIMARY KEY REFERENCES corrections(edit_id),
		error_category   TEXT,
		include_status   TEXT NOT NULL DEFAULT 'unreviewed',
		adjusted_start   REAL,
		adjusted_end     REAL,
		reviewer_notes   TEXT,
		human_updated_at TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_labels_status   ON review_labels(include_status);
	CREATE INDEX IF NOT EXISTS idx_labels_category ON review_labels(error_category);

	CREATE TABLE IF NOT EXISTS events (
		id       INTEGER PRIMARY KEY AUTOINCREMENT,
		ts       TEXT NOT NULL,
		edit_id  TEXT NOT NULL,
		field    TEXT NOT NULL,
		old_val  TEXT,
		new_val  TEXT,
		actor    TEXT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_events_edit_id ON events(edit_id);

	CREATE TABLE IF NOT EXISTS category_descriptions (
		category    TEXT PRIMARY KEY,
		label_el    TEXT NOT NULL,
		reason_el   TEXT NOT NULL,
		is_rejected INTEGER NOT NULL DEFAULT 0
	);
`);

// Add ingest columns if missing (idempotent via error swallow)
for (const col of ['ingest_category', 'cleaning_applied']) {
	try {
		await client.execute({ sql: `ALTER TABLE corrections ADD COLUMN ${col} TEXT`, args: [] });
	} catch { /* column already exists */ }
}
await client.execute({
	sql: `CREATE INDEX IF NOT EXISTS idx_corrections_ingest_category ON corrections(ingest_category)`,
	args: []
});

// Upsert category descriptions from the static taxonomy
for (const cat of INGEST_CATEGORIES) {
	await client.execute({
		sql: `INSERT OR REPLACE INTO category_descriptions (category, label_el, reason_el, is_rejected) VALUES (?, ?, ?, ?)`,
		args: [cat.key, cat.label_el, cat.reason_el, cat.is_rejected ? 1 : 0]
	});
}
console.log('Schema ready, category_descriptions seeded.');

const BATCH_SIZE = 500;
let batch: Array<{ row: RawCsvRow; clean: ReturnType<typeof categorise> }> = [];
let total = 0;
const catCounts = new Map<string, number>();

async function flushBatch() {
	if (!batch.length) return;
	const statements = batch.flatMap(({ row, clean }) => [
		{
			sql: `INSERT OR REPLACE INTO corrections
				(edit_id, edit_timestamp, edit_updated_at, before_text, after_text, edited_by,
				 utterance_start, utterance_end, audio_url, youtube_url, meeting_name, meeting_date,
				 ingest_category, cleaning_applied)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			args: [
				row.edit_id, row.edit_timestamp, row.edit_updated_at || null,
				clean.before_text, clean.after_text, row.edited_by || null,
				clean.utterance_start, clean.utterance_end,
				row.audio_url, row.youtube_url || null, row.meeting_name || null, row.meeting_date || null,
				clean.ingest_category, clean.cleaning_applied || null
			]
		},
		{
			sql: `INSERT OR IGNORE INTO review_labels (edit_id, include_status) VALUES (?, 'unreviewed')`,
			args: [row.edit_id]
		}
	]);
	await client.batch(statements, 'write');
	batch = [];
}

const parser = parse({ columns: true, skip_empty_lines: true, relax_column_count: true, bom: true });
const startMs = Date.now();

parser.on('readable', () => {
	let row: RawCsvRow;
	while ((row = parser.read() as RawCsvRow) !== null) {
		const clean = categorise(row);
		batch.push({ row, clean });
		total++;
		catCounts.set(clean.ingest_category, (catCounts.get(clean.ingest_category) ?? 0) + 1);

		if (batch.length >= BATCH_SIZE) {
			// Synchronous push trick: use the async version properly via a queue
			parser.pause();
			flushBatch().then(() => {
				if (total % 10000 === 0) process.stdout.write(`\r  ${total.toLocaleString()} rows…`);
				parser.resume();
			}).catch(err => {
				console.error('\nBatch error:', err);
				process.exit(1);
			});
		}
	}
});

parser.on('end', async () => {
	await flushBatch();
	const elapsed = ((Date.now() - startMs) / 1000).toFixed(1);
	console.log(`\r  Ingested ${total.toLocaleString()} rows in ${elapsed}s`);
	console.log('\nKατηγορίες:');
	for (const [cat, n] of [...catCounts.entries()].sort((a, b) => b[1] - a[1])) {
		console.log(`  ${cat.padEnd(24)} ${n.toLocaleString().padStart(8)}`);
	}
	await client.close();
});

parser.on('error', (err) => {
	console.error('CSV parse error:', err);
	process.exit(1);
});

createReadStream(csvPath).pipe(parser);
