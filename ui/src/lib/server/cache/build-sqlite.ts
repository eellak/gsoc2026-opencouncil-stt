/**
 * Materialise grouped corrections as a SQLite database.
 *
 * Runtime trade-off: the JSON cache is ~432 MB and forces every request path
 * to live with a fully-loaded in-memory map (>1 GB heap on real data). The
 * SQLite layout lets the server query a single row per request and keeps the
 * resident set under ~80 MB, which matters on the 1 GB Oracle VM.
 *
 * Schema (kept narrow on purpose — most query patterns hit `utterance_id`
 * directly; `ord` preserves the deterministic insertion order from
 * `buildGroups()` so seeded shuffles are stable across rebuilds):
 *
 *   groups(utterance_id PK, meeting_id, city_id, ord, json TEXT)
 *   edits(edit_id PK, utterance_id)
 *   meta(key PK, value)
 *
 * Notes:
 * - The serialised `json` strips the `label` field — labels live in the
 *   sidecar JSONL. Tests enforce this; if a label leaks in it can silently
 *   override the sidecar's source-of-truth.
 * - WAL is enabled while we write (faster bulk insert) and explicitly
 *   checkpointed + closed before the atomic rename. Runtime opens the DB
 *   read-only, so we never want WAL/SHM sidecar files following the rename.
 */

import Database from 'better-sqlite3';
import { promises as fs } from 'node:fs';
import type { CacheMeta, Group } from '$lib/domain/groups';
import type { TranscriptRow } from './transcript-extract';

/** Per-meeting presence + the separate transcript versioning, written to meta. */
export interface TranscriptManifest {
	/** Successfully-indexed meetings (drives per-meeting local-vs-fallback). */
	meetings: Array<{ city_id: string; meeting_id: string; utt_count: number }>;
	/** Independent of the CSV/label snapshot hashing — see context-in-index spec. */
	schema_version: number;
	manifest_hash: string;
	build_status: 'complete' | 'partial' | 'absent';
	failed_count: number;
}

export interface BuildSqliteInput {
	groups: Group[];
	meta: CacheMeta;
	/** Final on-disk path (the file we'll `fs.rename` into at the end). */
	outPath: string;
	/**
	 * Optional surrounding-context index. When present, rows + manifest are
	 * written to the `transcript`/`transcript_meeting` tables and transcript meta
	 * keys. Absent → no transcript tables (runtime falls back to live upstream).
	 */
	transcript?: {
		rows: TranscriptRow[];
		manifest: TranscriptManifest;
	};
}

export async function buildSqlite(input: BuildSqliteInput): Promise<void> {
	const { groups, meta, outPath, transcript } = input;
	const tmpPath = `${outPath}.tmp`;

	// Clean any leftover tmp from a previous failed build before opening.
	await fs.rm(tmpPath, { force: true });

	const db = new Database(tmpPath);
	try {
		db.pragma('journal_mode = WAL');
		db.pragma('synchronous = NORMAL');

		db.exec(`
			CREATE TABLE groups (
				utterance_id TEXT PRIMARY KEY,
				meeting_id   TEXT NOT NULL,
				city_id      TEXT NOT NULL,
				ord          INTEGER NOT NULL,
				json         TEXT NOT NULL
			);
			CREATE INDEX idx_groups_ord     ON groups(ord);
			CREATE INDEX idx_groups_meeting ON groups(meeting_id);

			CREATE TABLE edits (
				edit_id      TEXT PRIMARY KEY,
				utterance_id TEXT NOT NULL REFERENCES groups(utterance_id)
			);
			CREATE INDEX idx_edits_utterance ON edits(utterance_id);

			CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
		`);

		if (transcript) {
			db.exec(`
				CREATE TABLE transcript (
					utterance_id TEXT NOT NULL,
					city_id      TEXT NOT NULL,
					meeting_id   TEXT NOT NULL,
					seq          INTEGER NOT NULL,
					text         TEXT NOT NULL,
					start        REAL,
					"end"        REAL,
					speaker_tag  TEXT,
					PRIMARY KEY (city_id, meeting_id, utterance_id)
				);
				CREATE UNIQUE INDEX idx_transcript_meeting_seq ON transcript(city_id, meeting_id, seq);
				CREATE INDEX idx_transcript_utterance ON transcript(utterance_id);

				CREATE TABLE transcript_meeting (
					city_id    TEXT NOT NULL,
					meeting_id TEXT NOT NULL,
					utt_count  INTEGER NOT NULL,
					PRIMARY KEY (city_id, meeting_id)
				);
			`);
		}

		const insertGroup = db.prepare(
			'INSERT INTO groups (utterance_id, meeting_id, city_id, ord, json) VALUES (?, ?, ?, ?, ?)'
		);
		const insertEdit = db.prepare('INSERT INTO edits (edit_id, utterance_id) VALUES (?, ?)');
		const insertMeta = db.prepare('INSERT INTO meta (key, value) VALUES (?, ?)');

		const seenEditIds = new Set<string>();
		const tx = db.transaction((items: Group[]) => {
			let ord = 0;
			for (const g of items) {
				// Strip the label before serialisation so the sidecar remains
				// the only source of truth. Spreading like this avoids mutating
				// the caller's group object.
				const { label: _label, ...rest } = g;
				void _label; // discard
				const json = JSON.stringify(rest);
				insertGroup.run(g.utterance_id, g.meeting_id ?? '', g.city_id ?? '', ord, json);
				for (const e of g.edits) {
					if (seenEditIds.has(e.edit_id)) {
						throw new Error(
							`duplicate edit_id ${JSON.stringify(e.edit_id)} ` +
								`(utterance_id=${g.utterance_id}, csv_row=${e.csv_row})`
						);
					}
					seenEditIds.add(e.edit_id);
					insertEdit.run(e.edit_id, g.utterance_id);
				}
				ord++;
			}
		});
		tx(groups);

		const metaRows: Array<[string, string]> = [
			['cache_version', String(meta.cache_version)],
			['source_csv_path', meta.source_csv_path],
			['source_size', String(meta.source_size)],
			['source_mtime_ms', String(meta.source_mtime_ms)],
			['source_hash', meta.source_hash],
			['generated_at', meta.generated_at],
			['group_count', String(meta.group_count)],
			['edit_count', String(meta.edit_count)],
			['missing_utterance_id_count', String(meta.missing_utterance_id_count)]
		];
		// Filtered rebuild: persist the exclusion digest so the runtime cache_hash
		// reflects the index CONTENT (not just the source CSV), forcing dependent
		// snapshots (stats/category/eligibility) to invalidate when exclusions change.
		if (meta.exclusions) {
			metaRows.push(['exclusions_hash', meta.exclusions.exclusions_hash]);
		}
		// Transcript versioning is intentionally SEPARATE from the CSV source_hash /
		// exclusions_hash so it never invalidates the (CSV-derived) label snapshots.
		if (transcript) {
			const m = transcript.manifest;
			metaRows.push(
				['transcript_schema_version', String(m.schema_version)],
				['transcript_manifest_hash', m.manifest_hash],
				['transcript_build_status', m.build_status],
				['transcript_failed_count', String(m.failed_count)]
			);
		}
		const metaTx = db.transaction((rows: Array<[string, string]>) => {
			for (const [k, v] of rows) insertMeta.run(k, v);
		});
		metaTx(metaRows);

		if (transcript) {
			const insertTranscript = db.prepare(
				'INSERT INTO transcript (utterance_id, city_id, meeting_id, seq, text, start, "end", speaker_tag) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
			);
			const insertTrMeeting = db.prepare(
				'INSERT INTO transcript_meeting (city_id, meeting_id, utt_count) VALUES (?, ?, ?)'
			);
			const trTx = db.transaction(() => {
				for (const r of transcript.rows) {
					insertTranscript.run(
						r.utterance_id,
						r.city_id,
						r.meeting_id,
						r.seq,
						r.text,
						r.start,
						r.end,
						r.speaker_tag
					);
				}
				for (const mm of transcript.manifest.meetings) {
					insertTrMeeting.run(mm.city_id, mm.meeting_id, mm.utt_count);
				}
			});
			trTx();
		}

		// Force a full checkpoint so WAL/SHM are merged before close.
		// Runtime opens read-only and any leftover -wal/-shm files would be
		// invalid relative to the renamed db file.
		db.pragma('wal_checkpoint(TRUNCATE)');
	} catch (err) {
		db.close();
		await fs.rm(tmpPath, { force: true });
		await fs.rm(`${tmpPath}-journal`, { force: true });
		await fs.rm(`${tmpPath}-wal`, { force: true });
		await fs.rm(`${tmpPath}-shm`, { force: true });
		throw err;
	}
	db.close();

	// Final sanity: any sidecar files at this point are stale (we truncated
	// the WAL above and closed), but if they exist remove them so the rename
	// produces a clean single-file DB.
	await fs.rm(`${tmpPath}-wal`, { force: true });
	await fs.rm(`${tmpPath}-shm`, { force: true });

	await fs.rename(tmpPath, outPath);
}
