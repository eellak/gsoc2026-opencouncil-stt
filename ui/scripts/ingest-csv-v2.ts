#!/usr/bin/env bun
/**
 * Ingest the second-generation corrections CSV (with stable IDs) into Supabase
 * Postgres via Drizzle / postgres-js.
 *
 * New columns vs v1: utterance_id, meeting_id, city_id.
 *
 * Usage:
 *   DATABASE_URL=postgresql://... bun scripts/ingest-csv-v2.ts <csv-path>
 *
 * Idempotent: ON CONFLICT (edit_id) DO UPDATE SET … for corrections; labels are
 * left untouched (existing review state is preserved).
 */

import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { parse } from 'csv-parse';
import { createReadStream } from 'fs';
import { sql } from 'drizzle-orm';
import { resolve } from 'path';
import { corrections as correctionsTbl, reviewLabels, meetings as meetingsTbl } from '../drizzle/schema';
import { categorise } from './lib/csv-clean';

const url = process.env.DATABASE_URL;
if (!url) {
	console.error('DATABASE_URL is required');
	process.exit(1);
}

const csvPath = resolve(
	process.argv[2] ?? resolve(import.meta.dir, '..', '..', 'data-1779206108158.csv')
);
console.log(`Reading CSV: ${csvPath}`);
console.log(`Target: ${url.replace(/:[^:@/]+@/, ':****@')}`);

const pg = postgres(url, { prepare: false, max: 5 });
const db = drizzle(pg);

const BATCH = 1000;
let totalSeen = 0;
let totalInserted = 0;
const t0 = Date.now();

interface Row {
	edit_id: string;
	utterance_id: string;
	edit_timestamp: string;
	edit_updated_at: string;
	before_text: string;
	after_text: string;
	edited_by: string;
	utterance_start: string;
	utterance_end: string;
	audio_url: string;
	youtube_url: string;
	meeting_name: string;
	meeting_date: string;
	meeting_id: string;
	city_id: string;
}

async function flush(batch: Row[]): Promise<void> {
	if (!batch.length) return;
	const rows = batch.flatMap((r) => {
		const out = categorise({
			edit_id: r.edit_id,
			edit_timestamp: r.edit_timestamp,
			edit_updated_at: r.edit_updated_at,
			before_text: r.before_text,
			after_text: r.after_text,
			edited_by: r.edited_by,
			utterance_start: r.utterance_start,
			utterance_end: r.utterance_end,
			audio_url: r.audio_url,
			youtube_url: r.youtube_url,
			meeting_name: r.meeting_name,
			meeting_date: r.meeting_date
		});
		if (!Number.isFinite(out.utterance_start) || !Number.isFinite(out.utterance_end)) {
			console.warn(`[ingest] skip edit_id=${r.edit_id} invalid timestamps`);
			return [];
		}
		return [{
			editId: r.edit_id,
			utteranceId: r.utterance_id || null,
			meetingId: r.meeting_id || null,
			editTimestamp: r.edit_timestamp,
			editUpdatedAt: r.edit_updated_at || null,
			beforeText: out.before_text,
			afterText: out.after_text,
			editedBy: r.edited_by || null,
			utteranceStart: out.utterance_start,
			utteranceEnd: out.utterance_end,
			ingestCategory: out.ingest_category,
			cleaningApplied: out.cleaning_applied || null,
			// Sidecar fields for the meetings upsert below — not in the corrections schema.
			_meetingName: r.meeting_name || null,
			_meetingDate: r.meeting_date || null,
			_cityId: r.city_id || null,
			_audioUrl: r.audio_url || null,
			_youtubeUrl: r.youtube_url || null
		}];
	});
	if (!rows.length) return;

	// Upsert distinct meetings first so the FK on corrections.meeting_id is satisfied.
	const uniqueMeetings = new Map<string, {
		meetingId: string;
		meetingName: string | null;
		meetingDate: string | null;
		cityId: string | null;
		audioUrl: string | null;
		youtubeUrl: string | null;
	}>();
	for (const r of rows) {
		if (!r.meetingId) continue;
		if (uniqueMeetings.has(r.meetingId)) continue;
		uniqueMeetings.set(r.meetingId, {
			meetingId: r.meetingId,
			meetingName: r._meetingName,
			meetingDate: r._meetingDate,
			cityId: r._cityId,
			audioUrl: r._audioUrl,
			youtubeUrl: r._youtubeUrl
		});
	}
	if (uniqueMeetings.size) {
		await db
			.insert(meetingsTbl)
			.values([...uniqueMeetings.values()])
			.onConflictDoUpdate({
				target: meetingsTbl.meetingId,
				set: {
					meetingName: sql`excluded.meeting_name`,
					meetingDate: sql`excluded.meeting_date`,
					cityId: sql`excluded.city_id`,
					audioUrl: sql`excluded.audio_url`,
					// audioCdnUrl deliberately not overwritten — only apply-audio-cdn-map sets it
					youtubeUrl: sql`excluded.youtube_url`
				}
			});
	}

	const correctionRows = rows.map(({ _meetingName, _meetingDate, _cityId, _audioUrl, _youtubeUrl, ...rest }) => rest);
	await db
		.insert(correctionsTbl)
		.values(correctionRows)
		.onConflictDoUpdate({
			target: correctionsTbl.editId,
			set: {
				utteranceId: sql`excluded.utterance_id`,
				meetingId: sql`excluded.meeting_id`,
				editTimestamp: sql`excluded.edit_timestamp`,
				editUpdatedAt: sql`excluded.edit_updated_at`,
				beforeText: sql`excluded.before_text`,
				afterText: sql`excluded.after_text`,
				editedBy: sql`excluded.edited_by`,
				utteranceStart: sql`excluded.utterance_start`,
				utteranceEnd: sql`excluded.utterance_end`,
				ingestCategory: sql`excluded.ingest_category`,
				cleaningApplied: sql`excluded.cleaning_applied`
			}
		});

	// Ensure a review_labels row exists for every correction.
	await db
		.insert(reviewLabels)
		.values(rows.map((r) => ({ editId: r.editId, includeStatus: 'unreviewed' })))
		.onConflictDoNothing({ target: reviewLabels.editId });

	totalInserted += rows.length;
	const pct = ((totalSeen / 397556) * 100).toFixed(1);
	const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
	process.stdout.write(
		`\r  upsert: ${totalInserted.toLocaleString()}/~397k (${pct}%, ${elapsed}s)`
	);
}

const parser = parse({ columns: true, skip_empty_lines: true, relax_column_count: true });
let batch: Row[] = [];
const pending: Promise<void>[] = [];

parser.on('readable', () => {
	let row: Row;
	while ((row = parser.read() as Row) !== null) {
		batch.push(row);
		totalSeen++;
		if (batch.length >= BATCH) {
			const chunk = batch;
			batch = [];
			parser.pause();
			pending.push(
				flush(chunk).then(() => {
					parser.resume();
				})
			);
		}
	}
});

parser.on('end', async () => {
	if (batch.length) await flush(batch);
	await Promise.all(pending);
	console.log(`\n  done. ${totalInserted.toLocaleString()} rows upserted.`);
	await pg.end({ timeout: 5 });
});

parser.on('error', (err) => {
	console.error('CSV parse error:', err);
	process.exit(1);
});

createReadStream(csvPath).pipe(parser);
