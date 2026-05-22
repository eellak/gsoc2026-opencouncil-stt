#!/usr/bin/env bun
/**
 * Seed corrections.sqlite from fixtures/dummy-corrections.json (deterministic, no CSV required).
 * Usage: bun run seed  OR  bun scripts/seed-dummy.ts [--db path] [--fixture path]
 */

import { Database } from 'bun:sqlite';
import { join, resolve } from 'path';
import type { Correction } from '../src/lib/domain/types';

const argv = process.argv.slice(2);
const getArg = (flag: string, def: string) => {
	const i = argv.indexOf(flag);
	return i !== -1 && argv[i + 1] ? argv[i + 1] : def;
};

const dbPath = resolve(getArg('--db', join(import.meta.dir, '..', 'data', 'corrections.sqlite')));
const fixturePath = resolve(getArg('--fixture', join(import.meta.dir, '..', 'fixtures', 'dummy-corrections.json')));

const corrections: Correction[] = JSON.parse(await Bun.file(fixturePath).text());

const db = new Database(dbPath);
db.run('PRAGMA journal_mode = WAL');
db.run('PRAGMA synchronous = NORMAL');
db.run('PRAGMA foreign_keys = ON');

db.run(`
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
	)
`);
db.run(`CREATE INDEX IF NOT EXISTS idx_corrections_ts      ON corrections(edit_timestamp)`);
db.run(`CREATE INDEX IF NOT EXISTS idx_corrections_meeting ON corrections(meeting_name, meeting_date)`);
db.run(`CREATE INDEX IF NOT EXISTS idx_corrections_editor  ON corrections(edited_by)`);
db.run(`
	CREATE TABLE IF NOT EXISTS review_labels (
		edit_id          TEXT PRIMARY KEY REFERENCES corrections(edit_id),
		error_category   TEXT,
		include_status   TEXT NOT NULL DEFAULT 'unreviewed',
		adjusted_start   REAL,
		adjusted_end     REAL,
		reviewer_notes   TEXT,
		human_updated_at TEXT
	)
`);
db.run(`CREATE INDEX IF NOT EXISTS idx_labels_status   ON review_labels(include_status)`);
db.run(`CREATE INDEX IF NOT EXISTS idx_labels_category ON review_labels(error_category)`);

db.run('DELETE FROM review_labels');
db.run('DELETE FROM corrections');

const insertC = db.prepare(`
	INSERT INTO corrections
		(edit_id, edit_timestamp, edit_updated_at, before_text, after_text, edited_by,
		 utterance_start, utterance_end, audio_url, youtube_url, meeting_name, meeting_date)
	VALUES ($edit_id, $edit_timestamp, $edit_updated_at, $before_text, $after_text, $edited_by,
	        $utterance_start, $utterance_end, $audio_url, $youtube_url, $meeting_name, $meeting_date)
`);
const insertL = db.prepare(`INSERT INTO review_labels (edit_id, include_status) VALUES ($edit_id, 'unreviewed')`);

db.transaction(() => {
	for (const c of corrections) {
		insertC.run({
			$edit_id: c.edit_id, $edit_timestamp: c.edit_timestamp, $edit_updated_at: c.edit_updated_at,
			$before_text: c.before_text, $after_text: c.after_text, $edited_by: c.edited_by,
			$utterance_start: c.utterance_start, $utterance_end: c.utterance_end,
			$audio_url: c.audio_url, $youtube_url: c.youtube_url,
			$meeting_name: c.meeting_name, $meeting_date: c.meeting_date
		});
		insertL.run({ $edit_id: c.edit_id });
	}
})();

db.close();

console.log(`Seeded ${corrections.length} corrections into ${dbPath}`);
