#!/usr/bin/env bun
/**
 * One-shot migration: copy rows from local SQLite → Turso.
 * Run once after provisioning a Turso DB (or after re-ingesting locally).
 *
 * Usage:
 *   TURSO_DATABASE_URL=libsql://... TURSO_AUTH_TOKEN=... bun scripts/migrate-to-turso.ts [--db path]
 *
 * Safe to re-run: uses INSERT OR REPLACE on corrections, INSERT OR IGNORE on labels/events.
 */

import { Database } from 'bun:sqlite';
import { createClient } from '@libsql/client';
import { join, resolve } from 'path';
import { INGEST_CATEGORIES } from '../src/lib/domain/categories';

const argv = process.argv.slice(2);
const getArg = (flag: string, def: string) => {
	const i = argv.indexOf(flag);
	return i !== -1 && argv[i + 1] ? argv[i + 1] : def;
};

const dbPath = resolve(getArg('--db', join(import.meta.dir, '..', 'data', 'corrections.sqlite')));
const tursoUrl = process.env.TURSO_DATABASE_URL;
const authToken = process.env.TURSO_AUTH_TOKEN;

if (!tursoUrl) {
	console.error('TURSO_DATABASE_URL is required');
	process.exit(1);
}

console.log(`Source: ${dbPath}`);
console.log(`Target: ${tursoUrl}`);

const local = new Database(dbPath, { readonly: true });
const client = createClient({ url: tursoUrl, authToken });

// Apply full schema
await client.executeMultiple(`
	CREATE TABLE IF NOT EXISTS corrections (
		edit_id TEXT PRIMARY KEY, edit_timestamp TEXT NOT NULL, edit_updated_at TEXT,
		before_text TEXT NOT NULL, after_text TEXT NOT NULL, edited_by TEXT,
		utterance_start REAL NOT NULL, utterance_end REAL NOT NULL, audio_url TEXT NOT NULL,
		youtube_url TEXT, meeting_name TEXT, meeting_date TEXT,
		ingest_category TEXT, cleaning_applied TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_corrections_ts       ON corrections(edit_timestamp);
	CREATE INDEX IF NOT EXISTS idx_corrections_meeting  ON corrections(meeting_name, meeting_date);
	CREATE INDEX IF NOT EXISTS idx_corrections_editor   ON corrections(edited_by);
	CREATE INDEX IF NOT EXISTS idx_corrections_ingest_category ON corrections(ingest_category);

	CREATE TABLE IF NOT EXISTS review_labels (
		edit_id TEXT PRIMARY KEY REFERENCES corrections(edit_id),
		error_category TEXT, include_status TEXT NOT NULL DEFAULT 'unreviewed',
		adjusted_start REAL, adjusted_end REAL, reviewer_notes TEXT, human_updated_at TEXT
	);
	CREATE INDEX IF NOT EXISTS idx_labels_status   ON review_labels(include_status);
	CREATE INDEX IF NOT EXISTS idx_labels_category ON review_labels(error_category);

	CREATE TABLE IF NOT EXISTS events (
		id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, edit_id TEXT NOT NULL,
		field TEXT NOT NULL, old_val TEXT, new_val TEXT, actor TEXT NOT NULL
	);
	CREATE INDEX IF NOT EXISTS idx_events_edit_id ON events(edit_id);

	CREATE TABLE IF NOT EXISTS category_descriptions (
		category TEXT PRIMARY KEY, label_el TEXT NOT NULL,
		reason_el TEXT NOT NULL, is_rejected INTEGER NOT NULL DEFAULT 0
	);
`);

// Add ingest columns to existing Turso table if missing
for (const col of ['ingest_category', 'cleaning_applied']) {
	try {
		await client.execute({ sql: `ALTER TABLE corrections ADD COLUMN ${col} TEXT`, args: [] });
	} catch { /* already exists */ }
}

// Seed category descriptions
for (const cat of INGEST_CATEGORIES) {
	await client.execute({
		sql: `INSERT OR REPLACE INTO category_descriptions (category, label_el, reason_el, is_rejected) VALUES (?, ?, ?, ?)`,
		args: [cat.key, cat.label_el, cat.reason_el, cat.is_rejected ? 1 : 0]
	});
}
console.log('Schema + category_descriptions ready.');

const corrections = local.query('SELECT * FROM corrections ORDER BY edit_timestamp').all() as Record<string, unknown>[];
const labels = local.query('SELECT * FROM review_labels').all() as Record<string, unknown>[];

console.log(`Migrating ${corrections.length.toLocaleString()} corrections, ${labels.length.toLocaleString()} labels…`);

const BATCH = 50;

for (let i = 0; i < corrections.length; i += BATCH) {
	const chunk = corrections.slice(i, i + BATCH);
	await client.batch(
		chunk.map(c => ({
			sql: `INSERT OR REPLACE INTO corrections
				(edit_id, edit_timestamp, edit_updated_at, before_text, after_text, edited_by,
				 utterance_start, utterance_end, audio_url, youtube_url, meeting_name, meeting_date,
				 ingest_category, cleaning_applied)
				VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
			args: [
				c.edit_id, c.edit_timestamp, c.edit_updated_at,
				c.before_text, c.after_text, c.edited_by,
				c.utterance_start, c.utterance_end,
				c.audio_url, c.youtube_url, c.meeting_name, c.meeting_date,
				c.ingest_category ?? null, c.cleaning_applied ?? null
			]
		})),
		'write'
	);
	if ((i + BATCH) % 5000 === 0 || i + BATCH >= corrections.length) {
		process.stdout.write(`  corrections: ${Math.min(i + BATCH, corrections.length).toLocaleString()}/${corrections.length.toLocaleString()}\r`);
	}
}

console.log(`\n  corrections done.`);

for (let i = 0; i < labels.length; i += BATCH) {
	const chunk = labels.slice(i, i + BATCH);
	await client.batch(
		chunk.map(l => ({
			sql: `INSERT OR IGNORE INTO review_labels
				(edit_id, error_category, include_status, adjusted_start, adjusted_end, reviewer_notes, human_updated_at)
				VALUES (?, ?, ?, ?, ?, ?, ?)`,
			args: [
				l.edit_id, l.error_category, l.include_status,
				l.adjusted_start, l.adjusted_end, l.reviewer_notes, l.human_updated_at
			]
		})),
		'write'
	);
	if ((i + BATCH) % 5000 === 0 || i + BATCH >= labels.length) {
		process.stdout.write(`  labels: ${Math.min(i + BATCH, labels.length).toLocaleString()}/${labels.length.toLocaleString()}\r`);
	}
}

local.close();
await client.close();
console.log(`\nDone. ${corrections.length.toLocaleString()} corrections, ${labels.length.toLocaleString()} labels migrated to ${tursoUrl}`);
