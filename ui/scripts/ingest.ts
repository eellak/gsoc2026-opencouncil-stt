#!/usr/bin/env bun
/**
 * Ingest utterance-edits CSV into corrections.sqlite (bun:sqlite).
 * Usage: bun run scripts/ingest.ts <path/to/csv>
 *
 * Idempotent: re-running does INSERT OR REPLACE on edit_id.
 */

import { Database } from 'bun:sqlite';
import { parse } from 'csv-parse';
import { createReadStream } from 'fs';
import { join, resolve } from 'path';

// Anchor default paths to the script location so behavior is independent of
// the caller's working directory. Explicit argv[2] still wins.
const SCRIPT_DIR = import.meta.dir;
const UI_DIR = resolve(SCRIPT_DIR, '..');
const REPO_ROOT = resolve(UI_DIR, '..');

const csvPath = resolve(process.argv[2] ?? join(REPO_ROOT, 'utterance-edits-may12-26.csv'));
const dbPath = join(UI_DIR, 'data', 'corrections.sqlite');

console.log(`Reading CSV: ${csvPath}`);
console.log(`Writing DB:  ${dbPath}`);

const db = new Database(dbPath, { create: true });
db.exec('PRAGMA journal_mode = WAL');
db.exec('PRAGMA synchronous = NORMAL');
db.exec('PRAGMA foreign_keys = OFF'); // faster ingest, re-enable after

db.exec(`
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
  CREATE INDEX IF NOT EXISTS idx_corrections_meeting ON corrections(meeting_name, meeting_date);
  CREATE INDEX IF NOT EXISTS idx_corrections_audio   ON corrections(audio_url, utterance_start);
  CREATE INDEX IF NOT EXISTS idx_corrections_editor  ON corrections(edited_by);

  CREATE TABLE IF NOT EXISTS review_labels (
    edit_id          TEXT PRIMARY KEY,
    error_category   TEXT,
    include_status   TEXT NOT NULL DEFAULT 'unreviewed',
    adjusted_start   REAL,
    adjusted_end     REAL,
    reviewer_notes   TEXT,
    human_updated_at TEXT
  );
  CREATE INDEX IF NOT EXISTS idx_labels_status   ON review_labels(include_status);
  CREATE INDEX IF NOT EXISTS idx_labels_category ON review_labels(error_category);
`);

const insertCorrection = db.prepare(`
  INSERT OR REPLACE INTO corrections
    (edit_id, edit_timestamp, edit_updated_at, before_text, after_text, edited_by,
     utterance_start, utterance_end, audio_url, youtube_url, meeting_name, meeting_date)
  VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
`);

const insertLabel = db.prepare(`
  INSERT OR IGNORE INTO review_labels (edit_id, include_status)
  VALUES (?, 'unreviewed')
`);

const BATCH = 5000;
let batch: string[][] = [];
let total = 0;
const meetings = new Set<string>();
const audioUrls = new Set<string>();
const editors = new Map<string, number>();

function flushBatch(rows: string[][]) {
	db.exec('BEGIN');
	try {
		for (const r of rows) {
			const start = parseFloat(r[6]);
			const end = parseFloat(r[7]);
			if (!Number.isFinite(start) || !Number.isFinite(end)) {
				console.warn(`[ingest] skip edit_id=${r[0]} invalid timestamps`, { start: r[6], end: r[7] });
				continue;
			}
			insertCorrection.run(
				r[0], r[1], r[2] || null, r[3], r[4], r[5] || null,
				start, end,
				r[8], r[9] || null, r[10] || null, r[11] || null
			);
			insertLabel.run(r[0]);
		}
		db.exec('COMMIT');
	} catch (e) {
		db.exec('ROLLBACK');
		throw e;
	}
}

const parser = parse({ columns: true, skip_empty_lines: true, relax_column_count: true });

parser.on('readable', () => {
	let row: Record<string, string>;
	while ((row = parser.read()) !== null) {
		batch.push([
			row.edit_id, row.edit_timestamp, row.edit_updated_at,
			row.before_text, row.after_text, row.edited_by,
			row.utterance_start, row.utterance_end,
			row.audio_url, row.youtube_url, row.meeting_name, row.meeting_date
		]);
		total++;

		meetings.add(`${row.meeting_name}|${row.meeting_date}`);
		if (row.audio_url) audioUrls.add(row.audio_url);
		const e = row.edited_by || 'unknown';
		editors.set(e, (editors.get(e) ?? 0) + 1);

		if (batch.length >= BATCH) {
			flushBatch(batch);
			batch = [];
			process.stdout.write(`\r  ingested ${total.toLocaleString()} rows…`);
		}
	}
});

parser.on('end', () => {
	if (batch.length) flushBatch(batch);
	db.exec('PRAGMA foreign_keys = ON');
	console.log(`\r  ingested ${total.toLocaleString()} rows total.    `);
	console.log(`  unique meetings:  ${meetings.size}`);
	console.log(`  unique audio URLs: ${audioUrls.size}`);
	console.log(`  by editor:`);
	for (const [e, n] of [...editors.entries()].sort((a, b) => b[1] - a[1])) {
		console.log(`    ${e}: ${n.toLocaleString()}`);
	}
	db.close();
});

parser.on('error', (err) => {
	console.error('CSV parse error:', err);
	process.exit(1);
});

createReadStream(csvPath).pipe(parser);
