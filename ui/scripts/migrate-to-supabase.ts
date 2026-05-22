#!/usr/bin/env bun
/**
 * One-shot migration: copy local SQLite → Supabase Postgres via Drizzle/postgres-js.
 *
 * Usage:
 *   DATABASE_URL=postgresql://... bun scripts/migrate-to-supabase.ts [--db path]
 *
 * Idempotent: ON CONFLICT DO UPDATE on corrections, ON CONFLICT DO NOTHING on labels.
 * Reads `audio_cdn_url` if the local schema includes it; otherwise inserts null.
 */

import { Database } from 'bun:sqlite';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { join, resolve } from 'path';
import { sql } from 'drizzle-orm';
import { corrections as correctionsTbl, reviewLabels, categoryDescriptions } from '../drizzle/schema';
import { INGEST_CATEGORIES } from '../src/lib/domain/categories';

const argv = process.argv.slice(2);
const getArg = (flag: string, def: string) => {
	const i = argv.indexOf(flag);
	return i !== -1 && argv[i + 1] ? argv[i + 1] : def;
};

const url = process.env.DATABASE_URL;
if (!url) {
	console.error('DATABASE_URL is required (Supabase pooler connection string)');
	process.exit(1);
}

const dbPath = resolve(getArg('--db', join(import.meta.dir, '..', 'data', 'corrections.sqlite')));

console.log(`Source: ${dbPath}`);
console.log(`Target: ${url.replace(/:[^:@/]+@/, ':****@')}`);

const local = new Database(dbPath, { readonly: true });

// Detect optional column in local schema.
const hasAudioCdn =
	(local.query("SELECT 1 FROM pragma_table_info('corrections') WHERE name = 'audio_cdn_url'").get() as
		| { 1: number }
		| undefined) !== undefined;
console.log(`Local has audio_cdn_url: ${hasAudioCdn}`);

const client = postgres(url, { prepare: false, max: 1 });
const db = drizzle(client);

// 1. Seed category_descriptions
console.log('Seeding category_descriptions…');
for (const cat of INGEST_CATEGORIES) {
	await db
		.insert(categoryDescriptions)
		.values({
			category: cat.key,
			labelEl: cat.label_el,
			reasonEl: cat.reason_el,
			isRejected: cat.is_rejected ? 1 : 0
		})
		.onConflictDoUpdate({
			target: categoryDescriptions.category,
			set: {
				labelEl: cat.label_el,
				reasonEl: cat.reason_el,
				isRejected: cat.is_rejected ? 1 : 0
			}
		});
}

// 2. Counts
const correctionsCount = (local.query('SELECT count(*) as n FROM corrections').get() as { n: number }).n;
const labelsCount = (local.query('SELECT count(*) as n FROM review_labels').get() as { n: number }).n;
console.log(`Migrating ${correctionsCount.toLocaleString()} corrections, ${labelsCount.toLocaleString()} labels…`);

// 3. Stream corrections in chunks
const BATCH = 1000;
const selectCorrections = `
	SELECT edit_id, edit_timestamp, edit_updated_at, before_text, after_text, edited_by,
	       utterance_start, utterance_end, audio_url, ${hasAudioCdn ? 'audio_cdn_url' : 'NULL as audio_cdn_url'},
	       youtube_url, meeting_name, meeting_date, ingest_category, cleaning_applied
	FROM corrections
	ORDER BY edit_timestamp
	LIMIT ? OFFSET ?
`;

let migrated = 0;
const t0 = Date.now();

for (let offset = 0; offset < correctionsCount; offset += BATCH) {
	const rows = local.query(selectCorrections).all(BATCH, offset) as Array<{
		edit_id: string;
		edit_timestamp: string;
		edit_updated_at: string | null;
		before_text: string;
		after_text: string;
		edited_by: string | null;
		utterance_start: number;
		utterance_end: number;
		audio_url: string;
		audio_cdn_url: string | null;
		youtube_url: string | null;
		meeting_name: string | null;
		meeting_date: string | null;
		ingest_category: string | null;
		cleaning_applied: string | null;
	}>;
	if (!rows.length) break;

	await db
		.insert(correctionsTbl)
		.values(
			rows.map((r) => ({
				editId: r.edit_id,
				editTimestamp: r.edit_timestamp,
				editUpdatedAt: r.edit_updated_at,
				beforeText: r.before_text,
				afterText: r.after_text,
				editedBy: r.edited_by,
				utteranceStart: r.utterance_start,
				utteranceEnd: r.utterance_end,
				audioUrl: r.audio_url,
				audioCdnUrl: r.audio_cdn_url,
				youtubeUrl: r.youtube_url,
				meetingName: r.meeting_name,
				meetingDate: r.meeting_date,
				ingestCategory: r.ingest_category,
				cleaningApplied: r.cleaning_applied
			}))
		)
		.onConflictDoUpdate({
			target: correctionsTbl.editId,
			set: {
				editTimestamp: sql`excluded.edit_timestamp`,
				editUpdatedAt: sql`excluded.edit_updated_at`,
				beforeText: sql`excluded.before_text`,
				afterText: sql`excluded.after_text`,
				editedBy: sql`excluded.edited_by`,
				utteranceStart: sql`excluded.utterance_start`,
				utteranceEnd: sql`excluded.utterance_end`,
				audioUrl: sql`excluded.audio_url`,
				audioCdnUrl: sql`excluded.audio_cdn_url`,
				youtubeUrl: sql`excluded.youtube_url`,
				meetingName: sql`excluded.meeting_name`,
				meetingDate: sql`excluded.meeting_date`,
				ingestCategory: sql`excluded.ingest_category`,
				cleaningApplied: sql`excluded.cleaning_applied`
			}
		});

	migrated += rows.length;
	const pct = ((migrated / correctionsCount) * 100).toFixed(1);
	const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
	process.stdout.write(`\r  corrections: ${migrated.toLocaleString()}/${correctionsCount.toLocaleString()} (${pct}%, ${elapsed}s)`);
}

console.log('\n  corrections done.');

// 4. Labels (only non-default rows are interesting, but copy all for parity)
let lmigrated = 0;
for (let offset = 0; offset < labelsCount; offset += BATCH) {
	const rows = local
		.query(
			`SELECT edit_id, error_category, include_status, adjusted_start, adjusted_end, reviewer_notes, human_updated_at
			 FROM review_labels ORDER BY edit_id LIMIT ? OFFSET ?`
		)
		.all(BATCH, offset) as Array<{
		edit_id: string;
		error_category: string | null;
		include_status: string;
		adjusted_start: number | null;
		adjusted_end: number | null;
		reviewer_notes: string | null;
		human_updated_at: string | null;
	}>;
	if (!rows.length) break;

	await db
		.insert(reviewLabels)
		.values(
			rows.map((l) => ({
				editId: l.edit_id,
				errorCategory: l.error_category,
				includeStatus: l.include_status,
				adjustedStart: l.adjusted_start,
				adjustedEnd: l.adjusted_end,
				reviewerNotes: l.reviewer_notes,
				humanUpdatedAt: l.human_updated_at
			}))
		)
		.onConflictDoNothing({ target: reviewLabels.editId });

	lmigrated += rows.length;
	process.stdout.write(`\r  labels: ${lmigrated.toLocaleString()}/${labelsCount.toLocaleString()}`);
}

console.log('\n  labels done.');

local.close();
await client.end();
console.log(`\nDone. ${migrated.toLocaleString()} corrections, ${lmigrated.toLocaleString()} labels migrated.`);
